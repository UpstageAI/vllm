import random
import re
import string
import ast
import json
from collections.abc import Sequence
from typing import Union, Tuple, List, Optional

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    DeltaFunctionCall,
    DeltaToolCall,
    ExtractedToolCallInformation,
    ToolCall,
    FunctionCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser
)
from vllm.logger import init_logger

import pyjson5

class ToolCallID:
    _LENGTH = 10

    def __init__(self, id_val: str, validation: bool = False):
        self._id = id_val
        if validation:
            self._validate()

    @classmethod
    def random(cls, validation=False) -> 'ToolCallID':
        chars = string.ascii_lowercase + string.digits
        return cls(''.join(random.choice(chars) for _ in range(ToolCallID._LENGTH)), validation=validation)

    def _validate(self):
        assert len(self._id) == ToolCallID._LENGTH
        pattern = r'^[a-z0-9]{10}$'
        assert re.match(pattern, self._id) is not None

    def to_string(self) -> str:
        return self._id

    def __str__(self) -> str:
        return self.to_string()


logger = init_logger(__name__)


class SolarOpenToolParser(ToolParser):

    def extract_tool_calls(
            self,
            model_output: str,
            request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        content, tool_calls = self._parse_text(model_output)
        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=content if content else None,
        )

    def extract_tool_calls_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
            request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        # 1) Emit plain content tokens immediately until content terminator
        # tags or tool_calls section begins. Be careful when tokenizer groups
        # multiple special tags into a single delta (e.g., "<|tool_calls|><|tool_call:begin|>").
        # Only emit as content if BOTH:
        #  - previous_text has not seen any special markers, and
        #  - delta_text does NOT contain any of those markers as a substring.
        if delta_text:
            # Do NOT emit content if we have already started any special section
            # including tool call tags. Content should only be emitted at the
            # very beginning before any markers show up.
            special_markers = (
                "<|flush|>",
                "<|end|>",
                "<|begin|>",
                "<|tool_calls|>",
                "<|tool_call:begin|>",
                "<|tool_call:name|>",
                "<|tool_call:args|>",
                "<|tool_call:end|>",
                "<|calls|>",
            )
            if not any(tag in previous_text for tag in special_markers):
                if not any(tag in delta_text for tag in special_markers):
                    return DeltaMessage(content=delta_text, tool_calls=[])

        tool_call_deltas: list[DeltaToolCall] = []

        # Helper lambdas to analyze current_text state
        def _completed_calls_count(txt: str) -> int:
            return len(self._parse_tool_calls(txt))

        # Detect if a new tool_call started streaming its args just now.
        if delta_text and "<|tool_call:args|>" in delta_text:
            # Extract id and name for the latest tool call block present so far.
            begin_tag = "<|tool_call:begin|>"
            name_tag = "<|tool_call:name|>"
            args_tag = "<|tool_call:args|>"

            latest_args = current_text.rfind(args_tag)
            latest_name = current_text.rfind(name_tag, 0, latest_args if latest_args != -1 else None)
            latest_begin = current_text.rfind(begin_tag, 0, latest_name if latest_name != -1 else None)
            if latest_begin != -1 and latest_name != -1 and latest_args != -1 and latest_begin < latest_name < latest_args:
                tool_id = current_text[latest_begin + len(begin_tag):latest_name]
                func_name = current_text[latest_name + len(name_tag):latest_args]
                # Index equals number of args tags seen before this delta
                index = previous_text.count(args_tag)
                tool_call_deltas.append(
                    DeltaToolCall(
                        id=tool_id,
                        type="function",
                        index=index,
                        function=DeltaFunctionCall(name=func_name, arguments=""),
                    )
                )

        # If we are inside args (after last args tag without end), stream arg chunk
        begin_tag = "<|tool_call:begin|>"
        args_tag = "<|tool_call:args|>"
        end_tag = "<|tool_call:end|>"
        last_args_pos = current_text.rfind(args_tag)
        last_end_pos = current_text.rfind(end_tag)
        if last_args_pos != -1 and (last_end_pos == -1 or last_args_pos > last_end_pos):
            # Currently within args for the latest tool call
            # Determine previous args text and current args text to compute delta
            prev_last_args = previous_text.rfind(args_tag)
            prev_last_end = previous_text.rfind(end_tag)
            if prev_last_args != -1 and (prev_last_end == -1 or prev_last_args > prev_last_end):
                # Already inside args previously: emit only the delta_text
                if delta_text and delta_text not in (begin_tag, args_tag, end_tag):
                    # Stream into the most recently started (but not yet ended) call
                    index = max(previous_text.count(args_tag) - 1, 0)
                    tool_call_deltas.append(
                        DeltaToolCall(
                            id=None,
                            type=None,
                            index=index,
                            function=DeltaFunctionCall(name=None, arguments=delta_text),
                        )
                    )

        if not tool_call_deltas:
            return None

        return DeltaMessage(content=None, tool_calls=tool_call_deltas)

    # --------------------
    # Internal helpers
    # --------------------
    def _parse_text(self, text: str) -> Tuple[Optional[str], List[ToolCall]]:
        """Parse the completed segments from the given text.

        Returns (content, tool_calls) where content is extracted as the leading
        text up to the first '<|flush|>' or '<|end|>' marker, and tool_calls is
        a list of fully parsed tool calls inside '<|tool_calls|> ... <|calls|>'.
        """
        content = self._parse_content(text)
        tool_calls = self._parse_tool_calls(text)
        return content, tool_calls

    def _parse_content(self, text: str) -> Optional[str]:
        """Extract assistant content from the text.

        Rule: take the leading content before the first '<|flush|>' or
        '<|end|>' marker. If neither marker exists, return None.
        """
        end_tags = ["<|flush|>", "<|end|>"]

        # Take leading content before the first end tag
        end_positions = [pos for tag in end_tags if (pos := text.find(tag)) != -1]
        if not end_positions:
            return None
        end = min(end_positions)
        # Trim only the extracted portion; tests expect exact substring
        return text[:end]

    def _parse_tool_call_args(self, text: str) -> str:
        try:
            # Try to parse as JSON
            args = json.loads(text)
        except json.JSONDecodeError:
            try:
                # Try to parse as JSON5
                args = pyjson5.decode(text)
            except pyjson5.Json5DecoderException:
                try:
                    # Try to parse as Python literal
                    args = ast.literal_eval(text)
                except Exception:
                    # Fallback: return the original string
                    args = text
        if not isinstance(args, str):
            # Always convert back to JSON string
            args = json.dumps(args)
        return args

    def _parse_tool_calls(self, text: str) -> List[ToolCall]:
        tool_calls: list[ToolCall] = []
        # Parse globally; wrapper '<|tool_calls|>' may or may not be present.
        section_start = 0
        # section ends at <|calls|> if present, else use end of text
        section_end = text.find("<|calls|>")
        if section_end == -1:
            section_end = len(text)
        i = section_start
        while True:
            begin_tag = "<|tool_call:begin|>"
            name_tag = "<|tool_call:name|>"
            args_tag = "<|tool_call:args|>"
            end_tag = "<|tool_call:end|>"

            b = text.find(begin_tag, i, section_end)
            if b == -1:
                break
            b += len(begin_tag)
            n = text.find(name_tag, b, section_end)
            if n == -1:
                break
            tool_id = text[b:n]
            n += len(name_tag)
            a = text.find(args_tag, n, section_end)
            if a == -1:
                break
            name = text[n:a]
            a += len(args_tag)
            e = text.find(end_tag, a, section_end)
            if e == -1:
                break
            args = text[a:e]
            tool_calls.append(
                ToolCall(
                    id=tool_id,
                    function=FunctionCall(name=name, arguments=self._parse_tool_call_args(args)),
                ))
            i = e + len(end_tag)

        return tool_calls
