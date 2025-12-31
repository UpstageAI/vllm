from collections import namedtuple

import pytest

from vllm.entrypoints.openai.tool_parsers.solar_open_tool_parser import ToolCallID

ToolCallIDValueTestCase = namedtuple("ToolCallIDValueTestCase", ["name", "value", "valid"])

id_val_cases = [
    # Valid cases
    ToolCallIDValueTestCase(name="numeric", value="1234567890", valid=True),
    ToolCallIDValueTestCase(name="alphabet lowercase", value="abcdefghij", valid=True),
    ToolCallIDValueTestCase(name="alphanumeric (lowercase)", value="a1b2c3d4e5", valid=True),
    # Invalid cases
    ToolCallIDValueTestCase(name="alphabet uppercase", value="Abcdefghij", valid=False),
    ToolCallIDValueTestCase(name="non-alphanumeric", value="a1b2c3d4e!", valid=False),
    ToolCallIDValueTestCase(name="empty", value="", valid=False),
    ToolCallIDValueTestCase(name="too short", value="a", valid=False),
    ToolCallIDValueTestCase(name="too long", value="12345678901", valid=False),
]


class TestToolCallID:
    def test_random(self):
        ToolCallID.random(validation=True)

    @pytest.mark.parametrize("case", id_val_cases, ids=[c.name for c in id_val_cases])
    def test_init(self, case: ToolCallIDValueTestCase):
        test_func = lambda: ToolCallID(case.value, validation=True)
        if case.valid:
            test_func()
        else:
            pytest.raises(Exception, test_func)
