from pulsar.providers.openai_compat import OpenAICompatibleProvider


def test_openai_compatible_tool_call_parsing():
    data = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression":"6*7"}'},
                }],
            }
        }]
    }
    result = OpenAICompatibleProvider._extract_tool_result(data)
    assert result.content == ""
    assert result.tool_calls[0].name == "calculator"
    assert result.tool_calls[0].arguments == {"expression": "6*7"}
