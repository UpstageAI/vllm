import os
from enum import Enum
from typing import TYPE_CHECKING

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Hardcoded token IDs for Solar tokenizer

# Special token IDs for chat template
BEGIN_TOKEN_ID = 20  # <|begin|>
END_TOKEN_ID = 21  # <|end|>
THINK_TOKEN_ID = 22  # <|think|>
CONTENT_TOKEN_ID = 23  # <|content|>
FLUSH_TOKEN_ID = 24  # <|flush|> (eos token)
ASSISTANT_TOKEN_ID = 163444  # assistant
'''
'assistant' is not a special token exactly, but is treated as one in the logits
processing.
'''

# Tool call related tokens
CALLS_TOKEN_ID = 25  # <|calls|> (eos token for tool calls)
TOOL_CALLS_TOKEN_ID = 30  # <|tool_calls|>
TOOL_CALL_BEGIN_TOKEN_ID = 31  # <|tool_call:begin|>
TOOL_CALL_END_TOKEN_ID = 32  # <|tool_call:end|>
TOOL_CALL_NAME_TOKEN_ID = 33  # <|tool_call:name|>
TOOL_CALL_ARGS_TOKEN_ID = 34  # <|tool_call:args|>

# =============================================================================
# Dynamic Reasoning Budget Configuration
# =============================================================================
# budget = min(max_budget, max(min_budget, max_tokens * ratio / 100))
# Priority: max_budget > min_budget > ratio
#
# Available environment variables:
#   HIGH effort:
#     SOLAR_REASONING_BUDGET_HIGH_MAX    (default: 32768)  - max_budget
#     SOLAR_REASONING_BUDGET_HIGH_MIN    (default: 8192)   - min_budget
#     SOLAR_REASONING_BUDGET_HIGH_RATIO  (default: 60)     - % of max_tokens
#
#   MEDIUM effort:
#     SOLAR_REASONING_BUDGET_MEDIUM_MAX    (default: 16384)  - max_budget
#     SOLAR_REASONING_BUDGET_MEDIUM_MIN    (default: 4096)   - min_budget
#     SOLAR_REASONING_BUDGET_MEDIUM_RATIO  (default: 30)     - % of max_tokens
#
#   Tool call:
#     SOLAR_TOOL_CALL_ID_BUDGET  (default: 10)  - Max tokens for tool call ID
# =============================================================================

DEFAULT_REASONING_EFFORT = "high"

# HIGH effort settings (1k = 1024 tokens)
DEFAULT_REASONING_BUDGET_HIGH_MAX = 32 * 1024
DEFAULT_REASONING_BUDGET_HIGH_MIN = 8 * 1024
DEFAULT_REASONING_BUDGET_HIGH_RATIO = 60

# MEDIUM effort settings
DEFAULT_REASONING_BUDGET_MEDIUM_MAX = 16 * 1024
DEFAULT_REASONING_BUDGET_MEDIUM_MIN = 4 * 1024
DEFAULT_REASONING_BUDGET_MEDIUM_RATIO = 30

# Tool call settings
DEFAULT_TOOL_CALL_ID_BUDGET = 10

# Pre-computed constant to avoid repeated string parsing
NEG_INF = float("-inf")


def is_reasoning_request(params: SamplingParams) -> bool:
    """Check if the request is a reasoning request based on reasoning_effort."""
    return (params.reasoning_effort is None) or (params.reasoning_effort in ("medium", "high"))


def is_structured_outputs(params: SamplingParams) -> bool:
    """Check if the request has structured outputs constraints."""
    return (
        params.structured_outputs is not None
        and not params.structured_outputs.all_constraints_none()
    )


class GenerationState(Enum):
    """Enum representing the current state of response generation."""

    # Initial state - no tokens generated yet
    INITIAL = "initial"

    # New message states (after think_end)
    NEW_MESSAGE_BEGIN = "new_message_begin"  # <|begin|> token was just generated
    NEW_MESSAGE_ASSISTANT = "new_message_assistant"  # assistant token after <|begin|>

    # Think mode states
    THINK_BEGIN = "think_begin"  # <|think|> token was just generated
    THINK_IN_PROGRESS = "think_in_progress"  # Generating think content
    THINK_END = "think_end"  # <|end|> after think content
    THINK_FLUSH = "think_flush"  # <|flush|> after think content

    # Content states
    CONTENT_BEGIN = "content_begin"  # <|content|> token was just generated
    CONTENT_IN_PROGRESS = "content_in_progress"  # Generating content
    CONTENT_END = "content_end"  # <|end|> or <|flush|> after content
    CONTENT_FLUSH = "content_flush"  # <|flush|> after content

    # Tool call states
    # Flow: <|tool_calls|> -> (<|tool_call:begin|> -> id -> <|tool_call:name|> -> name -> <|tool_call:args|> -> args -> <|tool_call:end|>)+ -> <|calls|>
    # Note: Think message can appear before <|tool_calls|>
    TOOL_CALLS_BEGIN = "tool_calls_begin"  # <|tool_calls|> token was just generated
    TOOL_CALL_BEGIN = "tool_call_begin"  # <|tool_call:begin|> token was just generated
    TOOL_CALL_ID_IN_PROGRESS = "tool_call_id_in_progress"  # Generating tool call ID
    TOOL_CALL_NAME_BEGIN = "tool_call_name_begin"  # <|tool_call:name|> token was just generated
    TOOL_CALL_NAME_IN_PROGRESS = "tool_call_name_in_progress"  # Generating tool name
    TOOL_CALL_ARGS_BEGIN = "tool_call_args_begin"  # <|tool_call:args|> token was just generated
    TOOL_CALL_ARGS_IN_PROGRESS = "tool_call_args_in_progress"  # Generating tool arguments (JSON)
    TOOL_CALL_END = "tool_call_end"  # <|tool_call:end|> token was just generated (can start another tool call or end)
    CALLS = "calls"  # <|calls|> token was just generated (eos token for tool calls)


def get_generation_state(
    output_token_ids: list[int],
    begin_token_id: int = BEGIN_TOKEN_ID,
    end_token_id: int = END_TOKEN_ID,
    flush_token_id: int = FLUSH_TOKEN_ID,
    think_token_id: int = THINK_TOKEN_ID,
    content_token_id: int = CONTENT_TOKEN_ID,
    tool_calls_token_id: int = TOOL_CALLS_TOKEN_ID,
    tool_call_begin_token_id: int = TOOL_CALL_BEGIN_TOKEN_ID,
    tool_call_name_token_id: int = TOOL_CALL_NAME_TOKEN_ID,
    tool_call_args_token_id: int = TOOL_CALL_ARGS_TOKEN_ID,
    tool_call_end_token_id: int = TOOL_CALL_END_TOKEN_ID,
    calls_token_id: int = CALLS_TOKEN_ID,
    assistant_token_id: int = ASSISTANT_TOKEN_ID,
) -> GenerationState:
    """Determine the current generation state based on output token IDs.

    Analyzes the sequence of generated tokens to determine which phase
    of the chat template the generation is currently in.

    Response format specs:
    - think mode: <|think|>{{think-tokens}}<|end|><|begin|>assistant<|content|>{{content-tokens}}<|flush|>
    - tool mode: <|begin|>assistant<|tool_calls|><|tool_call:begin|>{{id}}<|tool_call:name|>{{name}}<|tool_call:args|>{{args}}<|tool_call:end|><|calls|>
    - tool mode (with think): <|think|>{{think-tokens}}<|end|><|begin|>assistant<|tool_calls|>...<|calls|>
    - no-think mode: <|content|>{{content-tokens}}<|flush|>

    Args:
        output_token_ids: List of token IDs generated so far.
        begin_token_id: Token ID for <|begin|>.
        end_token_id: Token ID for <|end|>.
        flush_token_id: Token ID for <|flush|> (eos).
        think_token_id: Token ID for <|think|>.
        content_token_id: Token ID for <|content|>.
        tool_calls_token_id: Token ID for <|tool_calls|>.
        tool_call_begin_token_id: Token ID for <|tool_call:begin|>.
        tool_call_name_token_id: Token ID for <|tool_call:name|>.
        tool_call_args_token_id: Token ID for <|tool_call:args|>.
        tool_call_end_token_id: Token ID for <|tool_call:end|>.
        calls_token_id: Token ID for <|calls|> (eos).
        assistant_token_id: Token ID for assistant.

    Returns:
        GenerationState indicating the current phase of generation.
    """
    if not output_token_ids:
        return GenerationState.INITIAL

    # Track state by scanning through tokens
    state = GenerationState.INITIAL
    in_think = False
    in_content = False

    for token_id in output_token_ids:
        if token_id == think_token_id:
            state = GenerationState.THINK_BEGIN
            in_think = True
            in_content = False

        elif token_id == content_token_id:
            state = GenerationState.CONTENT_BEGIN
            in_content = True
            in_think = False

        elif token_id == tool_calls_token_id:
            state = GenerationState.TOOL_CALLS_BEGIN
            in_think = False
            in_content = False

        elif token_id == tool_call_begin_token_id:
            state = GenerationState.TOOL_CALL_BEGIN

        elif token_id == tool_call_name_token_id:
            state = GenerationState.TOOL_CALL_NAME_BEGIN

        elif token_id == tool_call_args_token_id:
            state = GenerationState.TOOL_CALL_ARGS_BEGIN

        elif token_id == tool_call_end_token_id:
            state = GenerationState.TOOL_CALL_END

        elif token_id == calls_token_id:
            state = GenerationState.CALLS

        elif token_id == begin_token_id:
            state = GenerationState.NEW_MESSAGE_BEGIN

        elif token_id == assistant_token_id:
            if state == GenerationState.NEW_MESSAGE_BEGIN:
                state = GenerationState.NEW_MESSAGE_ASSISTANT

        elif token_id == end_token_id:
            if in_think:
                state = GenerationState.THINK_END
                in_think = False
            elif in_content:
                state = GenerationState.CONTENT_END
                in_content = False

        elif token_id == flush_token_id:
            if in_think:
                state = GenerationState.THINK_FLUSH
                in_think = False
            elif in_content:
                state = GenerationState.CONTENT_FLUSH
                in_content = False

        else:
            # Regular token - update state based on current context
            if state == GenerationState.THINK_BEGIN:
                state = GenerationState.THINK_IN_PROGRESS
            elif state == GenerationState.THINK_IN_PROGRESS:
                pass  # Stay in think_in_progress
            elif state == GenerationState.CONTENT_BEGIN:
                state = GenerationState.CONTENT_IN_PROGRESS
            elif state == GenerationState.CONTENT_IN_PROGRESS:
                pass  # Stay in content_in_progress
            elif state == GenerationState.TOOL_CALL_BEGIN:
                state = GenerationState.TOOL_CALL_ID_IN_PROGRESS
            elif state == GenerationState.TOOL_CALL_ID_IN_PROGRESS:
                pass  # Stay in tool_call_id_in_progress
            elif state == GenerationState.TOOL_CALL_NAME_BEGIN:
                state = GenerationState.TOOL_CALL_NAME_IN_PROGRESS
            elif state == GenerationState.TOOL_CALL_NAME_IN_PROGRESS:
                pass  # Stay in tool_call_name_in_progress
            elif state == GenerationState.TOOL_CALL_ARGS_BEGIN:
                state = GenerationState.TOOL_CALL_ARGS_IN_PROGRESS
            elif state == GenerationState.TOOL_CALL_ARGS_IN_PROGRESS:
                pass  # Stay in tool_call_args_in_progress

    return state


# Pre-computed list of all special token IDs for batch indexing
_ALL_SPECIAL_TOKEN_IDS = [
    BEGIN_TOKEN_ID,
    END_TOKEN_ID,
    THINK_TOKEN_ID,
    CONTENT_TOKEN_ID,
    FLUSH_TOKEN_ID,
    CALLS_TOKEN_ID,
    TOOL_CALLS_TOKEN_ID,
    TOOL_CALL_BEGIN_TOKEN_ID,
    TOOL_CALL_END_TOKEN_ID,
    TOOL_CALL_NAME_TOKEN_ID,
    TOOL_CALL_ARGS_TOKEN_ID,
]

# Pre-computed lists for state-specific batch indexing (excluding allowed tokens)
_SPECIAL_EXCEPT_END = [  # For THINK states (allow END)
    BEGIN_TOKEN_ID, FLUSH_TOKEN_ID, THINK_TOKEN_ID, CONTENT_TOKEN_ID,
    TOOL_CALLS_TOKEN_ID, CALLS_TOKEN_ID, TOOL_CALL_BEGIN_TOKEN_ID,
    TOOL_CALL_END_TOKEN_ID, TOOL_CALL_NAME_TOKEN_ID, TOOL_CALL_ARGS_TOKEN_ID,
]

_SPECIAL_EXCEPT_CONTENT_TOOLCALLS = [  # For NEW_MESSAGE_ASSISTANT (allow CONTENT, TOOL_CALLS)
    THINK_TOKEN_ID, BEGIN_TOKEN_ID, END_TOKEN_ID, FLUSH_TOKEN_ID,
    CALLS_TOKEN_ID, TOOL_CALL_BEGIN_TOKEN_ID, TOOL_CALL_END_TOKEN_ID,
    TOOL_CALL_NAME_TOKEN_ID, TOOL_CALL_ARGS_TOKEN_ID,
]

_SPECIAL_EXCEPT_FLUSH = [  # For CONTENT states (allow FLUSH)
    BEGIN_TOKEN_ID, END_TOKEN_ID, THINK_TOKEN_ID, CONTENT_TOKEN_ID,
    TOOL_CALLS_TOKEN_ID, CALLS_TOKEN_ID, TOOL_CALL_BEGIN_TOKEN_ID,
    TOOL_CALL_END_TOKEN_ID, TOOL_CALL_NAME_TOKEN_ID, TOOL_CALL_ARGS_TOKEN_ID,
]

_SPECIAL_EXCEPT_TOOLCALL_NAME = [  # For TOOL_CALL_ID_IN_PROGRESS (allow TOOL_CALL_NAME)
    BEGIN_TOKEN_ID, END_TOKEN_ID, THINK_TOKEN_ID, CONTENT_TOKEN_ID,
    FLUSH_TOKEN_ID, CALLS_TOKEN_ID, TOOL_CALLS_TOKEN_ID,
    TOOL_CALL_BEGIN_TOKEN_ID, TOOL_CALL_END_TOKEN_ID, TOOL_CALL_ARGS_TOKEN_ID,
]

_SPECIAL_EXCEPT_TOOLCALL_ARGS = [  # For TOOL_CALL_NAME_IN_PROGRESS (allow TOOL_CALL_ARGS)
    BEGIN_TOKEN_ID, END_TOKEN_ID, THINK_TOKEN_ID, CONTENT_TOKEN_ID,
    FLUSH_TOKEN_ID, CALLS_TOKEN_ID, TOOL_CALLS_TOKEN_ID,
    TOOL_CALL_BEGIN_TOKEN_ID, TOOL_CALL_END_TOKEN_ID, TOOL_CALL_NAME_TOKEN_ID,
]

_SPECIAL_EXCEPT_TOOLCALL_END = [  # For TOOL_CALL_ARGS_IN_PROGRESS (allow TOOL_CALL_END)
    BEGIN_TOKEN_ID, END_TOKEN_ID, THINK_TOKEN_ID, CONTENT_TOKEN_ID,
    FLUSH_TOKEN_ID, CALLS_TOKEN_ID, TOOL_CALLS_TOKEN_ID,
    TOOL_CALL_BEGIN_TOKEN_ID, TOOL_CALL_NAME_TOKEN_ID, TOOL_CALL_ARGS_TOKEN_ID,
]


def _forbid_all_special_tokens(logits: torch.Tensor) -> None:
    """Set all special token logits to -inf."""
    logits[_ALL_SPECIAL_TOKEN_IDS] = NEG_INF


class SolarOpenTemplateEnforcer:
    """Request-level logits processor that enforces Solar Open chat template.

    Enforces the following generation rules:
    - think mode: <|think|>{{tokens}}<|end|><|begin|>assistant<|content|>{{tokens}}<|flush|>
    - tool mode: <|tool_calls|><|tool_call:begin|>{{id}}<|tool_call:name|>{{name}}<|tool_call:args|>{{args}}<|tool_call:end|><|calls|>
    - tool+think mode: <|think|>{{tokens}}<|end|><|begin|>assistant<|tool_calls|>...<|calls|>
    - no-think mode: <|content|>{{tokens}}<|flush|>

    Key constraints:
    - Think message can only appear first
    - Think message must be followed by another message
    - Content and tool messages cannot coexist
    - Maximum 2 messages (think + content/tool, or just content/tool)

    Performance optimization:
    - Uses incremental state tracking to avoid full token sequence scan on each call
    - Maintains local counters for budget tracking
    - Uses pre-computed constants to avoid repeated object creation
    """

    # Pre-computed frozenset for reasoning state check (avoids set creation per call)
    _REASONING_STATES = frozenset({
        GenerationState.INITIAL,
        GenerationState.THINK_BEGIN,
        GenerationState.THINK_IN_PROGRESS,
    })

    def __init__(
        self,
        is_reasoning_request: bool,
        is_structured_outputs: bool,
        reasoning_budget: int | None = None,
        tool_call_id_budget: int = DEFAULT_TOOL_CALL_ID_BUDGET,
    ):
        self._is_reasoning_request = is_reasoning_request
        self._is_structured_outputs = is_structured_outputs
        self._reasoning_budget = reasoning_budget
        self._tool_call_id_budget = tool_call_id_budget

        # Incremental state tracking
        self._state = GenerationState.INITIAL
        self._last_processed_len = 0
        self._in_think = False
        self._in_content = False

        # Budget counters
        self._think_token_count = 0
        self._tool_call_id_token_count = 0

    def _reset_state(self) -> None:
        """Reset all incremental state to initial values.

        Called when defensive reprocessing is needed (e.g., token sequence inconsistency).
        """
        self._state = GenerationState.INITIAL
        self._last_processed_len = 0
        self._in_think = False
        self._in_content = False
        self._think_token_count = 0
        self._tool_call_id_token_count = 0

    def _process_token(self, token_id: int) -> None:
        """Process a single token and update internal state incrementally.

        Args:
            token_id: The token ID to process.
        """
        if token_id == THINK_TOKEN_ID:
            self._state = GenerationState.THINK_BEGIN
            self._in_think = True
            self._in_content = False
            self._think_token_count = 0  # Reset counter for new think block

        elif token_id == CONTENT_TOKEN_ID:
            self._state = GenerationState.CONTENT_BEGIN
            self._in_content = True
            self._in_think = False

        elif token_id == TOOL_CALLS_TOKEN_ID:
            self._state = GenerationState.TOOL_CALLS_BEGIN
            self._in_think = False
            self._in_content = False

        elif token_id == TOOL_CALL_BEGIN_TOKEN_ID:
            self._state = GenerationState.TOOL_CALL_BEGIN
            self._tool_call_id_token_count = 0  # Reset counter for new tool call

        elif token_id == TOOL_CALL_NAME_TOKEN_ID:
            self._state = GenerationState.TOOL_CALL_NAME_BEGIN

        elif token_id == TOOL_CALL_ARGS_TOKEN_ID:
            self._state = GenerationState.TOOL_CALL_ARGS_BEGIN

        elif token_id == TOOL_CALL_END_TOKEN_ID:
            self._state = GenerationState.TOOL_CALL_END

        elif token_id == CALLS_TOKEN_ID:
            self._state = GenerationState.CALLS

        elif token_id == BEGIN_TOKEN_ID:
            self._state = GenerationState.NEW_MESSAGE_BEGIN

        elif token_id == ASSISTANT_TOKEN_ID:
            if self._state == GenerationState.NEW_MESSAGE_BEGIN:
                self._state = GenerationState.NEW_MESSAGE_ASSISTANT

        elif token_id == END_TOKEN_ID:
            if self._in_think:
                self._state = GenerationState.THINK_END
                self._in_think = False
            elif self._in_content:
                self._state = GenerationState.CONTENT_END
                self._in_content = False

        elif token_id == FLUSH_TOKEN_ID:
            if self._in_think:
                self._state = GenerationState.THINK_FLUSH
                self._in_think = False
            elif self._in_content:
                self._state = GenerationState.CONTENT_FLUSH
                self._in_content = False

        else:
            # Regular token - update state and counters based on current context
            if self._state == GenerationState.THINK_BEGIN:
                self._state = GenerationState.THINK_IN_PROGRESS
                self._think_token_count += 1
            elif self._state == GenerationState.THINK_IN_PROGRESS:
                self._think_token_count += 1
            elif self._state == GenerationState.CONTENT_BEGIN:
                self._state = GenerationState.CONTENT_IN_PROGRESS
            elif self._state == GenerationState.CONTENT_IN_PROGRESS:
                pass  # Stay in content_in_progress
            elif self._state == GenerationState.TOOL_CALL_BEGIN:
                self._state = GenerationState.TOOL_CALL_ID_IN_PROGRESS
                self._tool_call_id_token_count += 1
            elif self._state == GenerationState.TOOL_CALL_ID_IN_PROGRESS:
                self._tool_call_id_token_count += 1
            elif self._state == GenerationState.TOOL_CALL_NAME_BEGIN:
                self._state = GenerationState.TOOL_CALL_NAME_IN_PROGRESS
            elif self._state == GenerationState.TOOL_CALL_NAME_IN_PROGRESS:
                pass  # Stay in tool_call_name_in_progress
            elif self._state == GenerationState.TOOL_CALL_ARGS_BEGIN:
                self._state = GenerationState.TOOL_CALL_ARGS_IN_PROGRESS
            elif self._state == GenerationState.TOOL_CALL_ARGS_IN_PROGRESS:
                pass  # Stay in tool_call_args_in_progress

    def _update_state_incremental(self, output_token_ids: list[int]) -> None:
        """Update internal state by processing only new tokens.

        Args:
            output_token_ids: Full list of output token IDs.
        """
        current_len = len(output_token_ids)

        # Defensive check: if token sequence is shorter than expected, reset and reprocess
        if current_len < self._last_processed_len:
            self._reset_state()

        # Process only new tokens
        for i in range(self._last_processed_len, current_len):
            self._process_token(output_token_ids[i])

        self._last_processed_len = current_len

    @staticmethod
    def _count_think_tokens(output_token_ids: list[int]) -> int:
        """Count the number of tokens generated after <|think|> token.

        Returns 0 if <|think|> token is not found (defensive).
        Note: This static method is kept for backward compatibility and testing.
        The incremental version uses _think_token_count instead.
        """
        try:
            think_index = output_token_ids.index(THINK_TOKEN_ID)
            return len(output_token_ids) - think_index - 1
        except ValueError:
            return 0

    @staticmethod
    def _count_tool_call_id_tokens(output_token_ids: list[int]) -> int:
        """Count the number of tokens generated after the last <|tool_call:begin|> token.

        Returns 0 if <|tool_call:begin|> token is not found (defensive).
        Note: This static method is kept for backward compatibility and testing.
        The incremental version uses _tool_call_id_token_count instead.
        """
        # Find the last occurrence of <|tool_call:begin|> for multi-tool-call support
        try:
            # Reverse search for the last <|tool_call:begin|>
            reversed_index = output_token_ids[::-1].index(TOOL_CALL_BEGIN_TOKEN_ID)
            last_begin_index = len(output_token_ids) - 1 - reversed_index
            return len(output_token_ids) - last_begin_index - 1
        except ValueError:
            return 0

    def __call__(
        self,
        output_token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # Update state incrementally (only process new tokens)
        self._update_state_incremental(output_token_ids)
        state = self._state

        # Handle structured outputs mode
        if self._is_structured_outputs:
            if not self._is_reasoning_request:
                # Non-reasoning request with structured outputs: no logit control
                return logits
            else:
                # Reasoning request with structured outputs:
                # Control logits only during reasoning phase
                if state not in self._REASONING_STATES:
                    # Reasoning finished, let structured outputs handle it
                    return logits

        if state == GenerationState.INITIAL:
            if self._is_reasoning_request:
                # Force: <|think|> only (reasoning request must start with think)
                think_logit = logits[THINK_TOKEN_ID].clone()
                logits.fill_(NEG_INF)
                logits[THINK_TOKEN_ID] = think_logit
            else:
                # Allow: <|content|>, <|tool_calls|> only
                content_logit = logits[CONTENT_TOKEN_ID].clone()
                tool_calls_logit = logits[TOOL_CALLS_TOKEN_ID].clone()
                logits.fill_(NEG_INF)
                logits[CONTENT_TOKEN_ID] = content_logit
                logits[TOOL_CALLS_TOKEN_ID] = tool_calls_logit

        elif state in (GenerationState.THINK_BEGIN, GenerationState.THINK_IN_PROGRESS):
            # Check if reasoning budget is exceeded (using incremental counter)
            if (
                self._reasoning_budget is not None
                and state == GenerationState.THINK_IN_PROGRESS
            ):
                if self._think_token_count >= self._reasoning_budget:
                    # Force <|end|> token to terminate reasoning
                    logits.fill_(NEG_INF)
                    logits[END_TOKEN_ID] = 0.0
                    return logits

            # Transform: <|flush|> -> <|end|>
            # Think must be followed by another message, so prevent early termination
            logits[END_TOKEN_ID] = torch.maximum(logits[END_TOKEN_ID], logits[FLUSH_TOKEN_ID])
            # Forbid all special tokens except <|end|>
            logits[_SPECIAL_EXCEPT_END] = NEG_INF

        elif state == GenerationState.THINK_END:
            # Force: <|begin|> only
            # Think must be followed by another message
            logits.fill_(NEG_INF)
            logits[BEGIN_TOKEN_ID] = 0.0

        elif state == GenerationState.NEW_MESSAGE_BEGIN:
            # Force: assistant token only
            logits.fill_(NEG_INF)
            logits[ASSISTANT_TOKEN_ID] = 0.0

        elif state == GenerationState.NEW_MESSAGE_ASSISTANT:
            # Allow: <|content|>, <|tool_calls|>, regular tokens
            # Forbid: all other special tokens
            logits[_SPECIAL_EXCEPT_CONTENT_TOOLCALLS] = NEG_INF

        elif state in (GenerationState.CONTENT_BEGIN, GenerationState.CONTENT_IN_PROGRESS):
            # Transform: <|end|> -> <|flush|>
            # Content cannot be followed by another message
            logits[FLUSH_TOKEN_ID] = torch.maximum(logits[FLUSH_TOKEN_ID], logits[END_TOKEN_ID])
            # Forbid all special tokens except <|flush|>
            logits[_SPECIAL_EXCEPT_FLUSH] = NEG_INF

        elif state == GenerationState.TOOL_CALLS_BEGIN:
            # Force: <|tool_call:begin|> only
            tool_call_begin_logit = logits[TOOL_CALL_BEGIN_TOKEN_ID].clone()
            logits.fill_(NEG_INF)
            logits[TOOL_CALL_BEGIN_TOKEN_ID] = tool_call_begin_logit

        elif state == GenerationState.TOOL_CALL_BEGIN:
            # Allow: regular tokens only (ID generation)
            # Forbid: all special tokens
            _forbid_all_special_tokens(logits)

        elif state == GenerationState.TOOL_CALL_ID_IN_PROGRESS:
            # Check if tool call ID budget is exceeded (using incremental counter)
            if self._tool_call_id_token_count >= self._tool_call_id_budget:
                # Force <|tool_call:name|> token to terminate ID generation
                logits.fill_(NEG_INF)
                logits[TOOL_CALL_NAME_TOKEN_ID] = 0.0
                return logits

            # Allow: <|tool_call:name|>, regular tokens
            # Forbid: all other special tokens
            logits[_SPECIAL_EXCEPT_TOOLCALL_NAME] = NEG_INF

        elif state == GenerationState.TOOL_CALL_NAME_BEGIN:
            # Allow: regular tokens only (function name generation)
            # Forbid: all special tokens
            _forbid_all_special_tokens(logits)

        elif state == GenerationState.TOOL_CALL_NAME_IN_PROGRESS:
            # Allow: <|tool_call:args|>, regular tokens
            # Forbid: all other special tokens
            logits[_SPECIAL_EXCEPT_TOOLCALL_ARGS] = NEG_INF

        elif state == GenerationState.TOOL_CALL_ARGS_BEGIN:
            # Allow: regular tokens only (JSON args generation)
            # Forbid: all special tokens
            _forbid_all_special_tokens(logits)

        elif state == GenerationState.TOOL_CALL_ARGS_IN_PROGRESS:
            # Allow: <|tool_call:end|>, regular tokens
            # Forbid: all other special tokens
            logits[_SPECIAL_EXCEPT_TOOLCALL_END] = NEG_INF

        elif state == GenerationState.TOOL_CALL_END:
            # Allow: <|tool_call:begin|> (next tool call), <|calls|> (end)
            # Forbid: all other special tokens
            tool_call_begin_logit = logits[TOOL_CALL_BEGIN_TOKEN_ID].clone()
            calls_logit = logits[CALLS_TOKEN_ID].clone()
            logits.fill_(NEG_INF)
            logits[TOOL_CALL_BEGIN_TOKEN_ID] = tool_call_begin_logit
            logits[CALLS_TOKEN_ID] = calls_logit

        # CALLS state: no processing needed (EOS)

        return logits

class SolarOpenTemplateLogitsProcessor(AdapterLogitsProcessor):
    """
    Logits processor that enforces Solar Open chat template.
    This processor manages the generation flow according to the
    Solar Open chat template by tracking generation states.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ):
        super().__init__(vllm_config, device, is_pin_memory)

        # Dynamic reasoning budget settings for HIGH effort
        self._high_max = self._parse_env_int(
            "SOLAR_REASONING_BUDGET_HIGH_MAX", DEFAULT_REASONING_BUDGET_HIGH_MAX
        )
        self._high_min = self._parse_env_int(
            "SOLAR_REASONING_BUDGET_HIGH_MIN", DEFAULT_REASONING_BUDGET_HIGH_MIN
        )
        self._high_ratio = self._parse_env_int(
            "SOLAR_REASONING_BUDGET_HIGH_RATIO", DEFAULT_REASONING_BUDGET_HIGH_RATIO
        )

        # Dynamic reasoning budget settings for MEDIUM effort
        self._medium_max = self._parse_env_int(
            "SOLAR_REASONING_BUDGET_MEDIUM_MAX", DEFAULT_REASONING_BUDGET_MEDIUM_MAX
        )
        self._medium_min = self._parse_env_int(
            "SOLAR_REASONING_BUDGET_MEDIUM_MIN", DEFAULT_REASONING_BUDGET_MEDIUM_MIN
        )
        self._medium_ratio = self._parse_env_int(
            "SOLAR_REASONING_BUDGET_MEDIUM_RATIO", DEFAULT_REASONING_BUDGET_MEDIUM_RATIO
        )

        self._tool_call_id_budget: int = self._parse_env_int(
            "SOLAR_TOOL_CALL_ID_BUDGET", DEFAULT_TOOL_CALL_ID_BUDGET
        )

    @staticmethod
    def _parse_env_int(env_var: str, default: int) -> int:
        """Parse environment variable as integer, return default if not set or invalid."""
        value = os.environ.get(env_var)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def _calculate_reasoning_budget(self, effort: str, max_tokens: int) -> int:
        """Calculate dynamic reasoning budget based on effort level and max_tokens.

        Priority (higher priority conditions are applied first):
        1. max_budget: Upper limit for reasoning tokens
        2. min_budget: Lower limit for reasoning tokens
        3. ratio: Percentage of max_tokens allocated for reasoning (e.g., 60 means 60%)

        budget = min(max_budget, max(min_budget, max_tokens * ratio / 100))
        """
        if effort == "high":
            max_budget = self._high_max
            min_budget = self._high_min
            ratio = self._high_ratio
        elif effort == "medium":
            max_budget = self._medium_max
            min_budget = self._medium_min
            ratio = self._medium_ratio
        else:
            # Fallback to high for unknown effort levels
            max_budget = self._high_max
            min_budget = self._high_min
            ratio = self._high_ratio

        # Calculate ratio-based budget (ratio is percentage, e.g., 60 means 60%)
        ratio_budget = max_tokens * ratio // 100

        # Apply priority: max > min > ratio
        budget = min(max_budget, max(min_budget, ratio_budget))

        return budget

    def is_argmax_invariant(self) -> bool:
        """This processor can change argmax result by forcing specific tokens."""
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        reasoning_effort = params.reasoning_effort or DEFAULT_REASONING_EFFORT
        reasoning_budget = self._calculate_reasoning_budget(
            reasoning_effort, params.max_tokens
        )
        return SolarOpenTemplateEnforcer(
            is_reasoning_request=is_reasoning_request(params),
            is_structured_outputs=is_structured_outputs(params),
            reasoning_budget=reasoning_budget,
            tool_call_id_budget=self._tool_call_id_budget,
        )

