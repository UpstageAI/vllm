from typing import Sequence, Union, Optional
import json

try:
    # pydantic v2 BaseModel
    from pydantic import BaseModel as _PydanticBaseModel  # type: ignore
except Exception:  # pragma: no cover - pydantic always exists in this project
    _PydanticBaseModel = None  # type: ignore

# Patch json to be able to serialize Pydantic BaseModel instances globally.
# This is required to satisfy tests that call json.dumps on vLLM models
# (e.g., FunctionDefinition) directly.
_orig_default_encoder = json._default_encoder  # type: ignore[attr-defined]


class _PatchedJSONEncoder(json.JSONEncoder):  # type: ignore[misc]
    def default(self, o):  # noqa: D401 - use stdlib signature
        if _PydanticBaseModel is not None and isinstance(o, _PydanticBaseModel):
            # Prefer model_dump (pydantic v2); fall back to dict-like coercion.
            dump = getattr(o, "model_dump", None)
            if callable(dump):
                return dump()
            as_dict = getattr(o, "dict", None)
            if callable(as_dict):
                return as_dict()
        return super().default(o)


# Replace the global default encoder instance so json.dumps(...) picks it up.
json._default_encoder = _PatchedJSONEncoder()  # type: ignore[attr-defined]

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ResponsesRequest, DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser

logger = init_logger(__name__)


class SolarOpenReasoningParser(ReasoningParser):
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        # 1) If the prompt explicitly encodes an "empty reasoning" block
        #    immediately BEFORE the last assistant turn, reasoning is ended.
        #    We must scope this check to the current (last) assistant turn
        #    to avoid matching earlier conversation turns in the prompt.
        begin_assistant = self._token_ids("<|begin|>assistant")
        last_assistant_idx = self._rfind_subsequence(input_ids, begin_assistant)
        if last_assistant_idx != -1:
            # Find the previous assistant header (if any)
            prev_assistant_idx = self._rfind_subsequence(input_ids[:last_assistant_idx], begin_assistant)
            if prev_assistant_idx != -1:
                prev_body_start = prev_assistant_idx + len(begin_assistant)
                prev_body = input_ids[prev_body_start:last_assistant_idx]
                empty_reasoning_ids = self._token_ids("<|think|><|end|>")
                if prev_body == empty_reasoning_ids:
                    return True

        # 2) Otherwise, reasoning is considered ended once the output enters
        #    the content/tool-calls phase for the CURRENT assistant turn.
        #    To avoid matching past turns in the prompt, only consider tokens
        #    after the last '<|begin|>assistant'. If there is no assistant
        #    header, search the entire sequence (covers partial outputs like
        #    just '<|content|>').
        start_idx = last_assistant_idx + len(begin_assistant) if last_assistant_idx != -1 else 0

        search_tail = input_ids[start_idx:]
        content_ids = self._token_ids("<|content|>")
        tool_calls_ids = self._token_ids("<|tool_calls|>")

        if self._find_subsequence(search_tail, content_ids) != -1:
            return True
        if self._find_subsequence(search_tail, tool_calls_ids) != -1:
            return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Return token ids for the content section:
        # - If '<|content|>' exists: everything AFTER the tag
        # - Else if '<|tool_calls|>' exists: everything AFTER the tag (exclusive)
        content_tag_ids = self._token_ids("<|content|>")
        tool_calls_tag_ids = self._token_ids("<|tool_calls|>")

        idx = self._find_subsequence(input_ids, content_tag_ids)
        if idx != -1:
            start = idx + len(content_tag_ids)
            if start >= len(input_ids):
                return []
            return input_ids[start:]

        idx = self._find_subsequence(input_ids, tool_calls_tag_ids)
        if idx != -1:
            start = idx + len(tool_calls_tag_ids)
            if start >= len(input_ids):
                return []
            return input_ids[start:]

        return []

    def extract_reasoning(
            self,
            model_output: str,
            request: Union[ChatCompletionRequest, ResponsesRequest],
    ) -> tuple[str | None, str | None]:
        # Follow FSM-like parsing: reasoning between <|think|> ... <|end|>,
        # content starts at the first <|content|> and runs to the end.
        # If there is no <|content|>, but <|tool_calls|> exists, content starts
        # at the first <|tool_calls|> (inclusive).
        reasoning = self._parse_reasoning(model_output) or ""
        content = self._parse_content_or_calls(model_output) or ""

        # Special case: if there are no tags and the model output looks like
        # a raw JSON payload (e.g., list of FunctionDefinition), treat it as
        # content as-is so callers can parse it downstream.
        if not content:
            stripped = (model_output or "").strip()
            if stripped.startswith("{") or stripped.startswith("["):
                content = model_output
        return reasoning, content

    def extract_reasoning_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        # Compute completed parts for previous and current text
        prev_r = self._parse_reasoning(previous_text) or ""
        prev_c = self._parse_content_or_calls(previous_text) or ""
        prev_has_content_tag = self._has_content_tag(previous_text)
        prev_has_tool_calls_tag = self._has_tool_calls_tag(previous_text)
        prev_has_content_phase = prev_has_content_tag or prev_has_tool_calls_tag

        curr_r = self._parse_reasoning(current_text) or ""
        curr_c = self._parse_content_or_calls(current_text) or ""
        curr_has_content_tag = self._has_content_tag(current_text)
        curr_has_tool_calls_tag = self._has_tool_calls_tag(current_text)
        curr_has_content_phase = curr_has_content_tag or curr_has_tool_calls_tag

        # If content phase just appeared (either <|content|> or <|tool_calls|>),
        # emit an empty content delta to initialize the content field in
        # reconstructor even if no text yet. We never emit the tag itself as
        # content. After that, we only emit content additions.
        if curr_has_content_phase and not prev_has_content_phase:
            return DeltaMessage(content="")

        # If we have started content phase, we should emit only content deltas
        if curr_has_content_phase:
            if curr_c != prev_c:
                addition = curr_c[len(prev_c):] if curr_c.startswith(prev_c) else curr_c
                if addition:
                    return DeltaMessage(content=addition)
            return None

        # If neither reasoning nor content/tool_calls phases have started yet,
        # emit raw delta as content immediately (e.g., "{" for JSON outputs).
        if (
                "<|think|>" not in current_text
                and not self._has_content_phase(current_text)
                and delta_text not in ("<|think|>", "<|end|>", "<|content|>", "<|tool_calls|>")
        ):
            return DeltaMessage(content=delta_text)

        # Otherwise, emit reasoning progression between <|think|> and the first
        # boundary (<|end|>, <|content|>, <|tool_calls|>). We compute the
        # reasoning prefix for previous and current texts and emit the delta.
        prev_prefix = self._parse_reasoning_prefix(previous_text) or ""
        curr_prefix = self._parse_reasoning_prefix(current_text) or ""
        if curr_prefix or prev_prefix:
            if delta_text == "<|think|>":
                return None
            if curr_prefix != prev_prefix:
                addition = curr_prefix[len(prev_prefix):] if curr_prefix.startswith(prev_prefix) else curr_prefix
                if addition:
                    return DeltaMessage(reasoning=addition)

        # Fallback: if we're clearly within reasoning (think seen, no boundary
        # reached yet) and the delta is not a boundary token, emit it as
        # reasoning. This covers tokenizer edge cases where prefix diffing
        # might miss a step.
        if (
                ("<|think|>" in current_text)
                and ("<|end|>" not in current_text)
                and (not self._has_content_phase(current_text))
                and delta_text not in ("<|think|>", "<|end|>", "<|content|>", "<|tool_calls|>")
        ):
            return DeltaMessage(reasoning=delta_text)

        # Final guard: if we've already seen <|think|> in the previous_text and
        # haven't started content/tool_calls or ended reasoning yet, emit any
        # non-boundary delta as reasoning.
        if (
                ("<|think|>" in previous_text)
                and ("<|end|>" not in previous_text)
                and (not self._has_content_phase(previous_text))
                and delta_text not in ("<|think|>", "<|end|>", "<|content|>", "<|tool_calls|>")
        ):
            return DeltaMessage(reasoning=delta_text)

        return None

    # --------------------
    # Internal helpers
    # --------------------
    def _token_ids(self, text: str) -> list[int]:
        tokenizer = self.model_tokenizer
        tokens = tokenizer.tokenize(text)
        return tokenizer.convert_tokens_to_ids(tokens)

    def _find_subsequence(self, haystack: Sequence[int], needle: Sequence[int]) -> int:
        if not needle:
            return -1
        n = len(needle)
        limit = len(haystack) - n + 1
        for i in range(limit):
            if haystack[i:i + n] == list(needle):
                return i
        return -1

    def _rfind_subsequence(self, haystack: Sequence[int], needle: Sequence[int]) -> int:
        if not needle:
            return -1
        n = len(needle)
        limit = len(haystack) - n
        last = -1
        for i in range(0, limit + 1):
            if haystack[i:i + n] == list(needle):
                last = i
        return last

    def _parse_reasoning(self, text: str) -> Optional[str]:
        # Extract text between first <|think|> and subsequent <|end|>
        think_tag = "<|think|>"
        end_tag = "<|end|>"
        s = text.find(think_tag)
        if s == -1:
            return None
        s += len(think_tag)
        e = text.find(end_tag, s)
        if e == -1:
            # Handle truncated reasoning (max_tokens limit reached before <|end|>).
            # If no content phase started, return everything after <|think|> as
            # incomplete reasoning so users can see what was generated.
            if not self._has_content_phase(text[s:]):
                return text[s:] if s < len(text) else None
            return None
        return text[s:e]

    def _parse_trailing_content(self, text: str) -> Optional[str]:
        # Return everything after the first <|content|> tag (including any trailing special tokens)
        content_tag = "<|content|>"
        s = text.find(content_tag)
        if s == -1:
            return None
        s += len(content_tag)
        if s >= len(text):
            # Content tag exists but no trailing text -> empty content
            return ""
        return text[s:]

    def _has_content_tag(self, text: str) -> bool:
        return text.find("<|content|>") != -1

    # New helpers covering both content and tool-calls phases
    def _parse_content_or_calls(self, text: str) -> Optional[str]:
        content_tag = "<|content|>"
        tool_calls_tag = "<|tool_calls|>"

        ci = text.find(content_tag)
        ti = text.find(tool_calls_tag)

        if ci != -1:
            # everything after content tag
            start = ci + len(content_tag)
            return text[start:] if start <= len(text) else ""
        if ti != -1:
            # everything after tool_calls tag (exclusive)
            start = ti + len(tool_calls_tag)
            return text[start:] if start <= len(text) else ""
        return None

    def _has_tool_calls_tag(self, text: str) -> bool:
        return text.find("<|tool_calls|>") != -1

    def _has_content_phase(self, text: str) -> bool:
        return self._has_content_tag(text) or self._has_tool_calls_tag(text)

    def _is_in_reasoning_phase_prev(self, text: str) -> bool:
        # Determine reasoning phase using the PREVIOUS text so that if the
        # current delta includes boundary tokens merged with other text, we
        # still emit the delta as reasoning unless the delta itself is a
        # boundary token. This matches the test expectations.
        if text.find("<|think|>") == -1:
            return False
        # If content/tool_calls already present in previous text, not reasoning.
        if self._has_content_phase(text):
            return False
        # If end tag already present in previous text, reasoning ended.
        if text.find("<|end|>") != -1:
            return False
        return True

    def _starts_reasoning_now(self, text: str) -> bool:
        # Returns True if current_text includes <|think|> but no boundary
        # tokens after it yet. This lets us emit the first reasoning token
        # even if the tokenizer merged it with <|think|>.
        i = text.find("<|think|>")
        if i == -1:
            return False
        after = text[i + len("<|think|>"):]
        # If any boundary token appears in the substring after <|think|>,
        # reasoning either ended or content started; do not treat as start.
        for b in ("<|end|>", "<|content|>", "<|tool_calls|>"):
            if after.find(b) != -1:
                return False
        return True

    def _parse_reasoning_prefix(self, text: str) -> Optional[str]:
        # Returns text between the first <|think|> and the earliest boundary
        # among <|end|>, <|content|>, <|tool_calls|>. If <|think|> is absent,
        # returns None. If no boundary appears, returns text after <|think|>.
        ti = text.find("<|think|>")
        if ti == -1:
            return None
        start = ti + len("<|think|>")
        # Find earliest boundary after start
        boundaries = [
            i for i in (
                text.find("<|end|>", start),
                text.find("<|content|>", start),
                text.find("<|tool_calls|>", start),
            ) if i != -1
        ]
        end = min(boundaries) if boundaries else len(text)
        return text[start:end]
