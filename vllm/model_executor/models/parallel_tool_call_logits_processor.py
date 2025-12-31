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
TOOL_CALL_END_TOKEN_ID = 32  # <|tool_call:end|>
CALLS_TOKEN_ID = 25  # <|calls|>


class SingleToolCallEnforcer:
    """Request-level logits processor that enforces single tool call.

    When <|tool_call:end|> token is generated, forces the next token
    to be <|calls|> (which is a stop token), preventing parallel tool calls.
    """

    def __init__(
        self,
        tool_call_end_token_id: int,
        calls_token_id: int,
    ):
        self._tool_call_end_token_id = tool_call_end_token_id
        self._calls_token_id = calls_token_id

    def __call__(
        self,
        output_token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # Check if last generated token is <|tool_call:end|>
        if output_token_ids and output_token_ids[-1] == self._tool_call_end_token_id:
            # Force next token to be <|calls|> by masking all other tokens
            mask = torch.full_like(logits, -float("inf"))
            mask[self._calls_token_id] = logits[self._calls_token_id]
            return mask

        return logits


class ParallelToolCallLogitsProcessor(AdapterLogitsProcessor):
    """Logits processor that enforces single tool call when parallel_tool_calls=False.

    When parallel_tool_calls is disabled in SamplingParams, this processor
    ensures that after <|tool_call:end|> is generated, the next token is
    forced to be <|calls|> (a stop token), preventing multiple tool calls.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ):
        super().__init__(vllm_config, device, is_pin_memory)

    def is_argmax_invariant(self) -> bool:
        """This processor can change argmax result by forcing specific tokens."""
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        """Return a request-level logits processor if parallel_tool_calls=False.

        Args:
            params: Request sampling params

        Returns:
            SingleToolCallEnforcer if parallel_tool_calls is False, otherwise None.
        """
        # Only apply when parallel_tool_calls is explicitly disabled
        if params.parallel_tool_calls is False:
            return SingleToolCallEnforcer(
                tool_call_end_token_id=TOOL_CALL_END_TOKEN_ID,
                calls_token_id=CALLS_TOKEN_ID,
            )

        return None

