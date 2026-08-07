import unittest

import tool_compat


def function_tool(name="read_file", strict=True):
    return {
        "type": "function",
        "name": name,
        "description": "short",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "enum": ["test.txt"],
                    "oneOf": [{"const": "test.txt"}],
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": strict,
    }


def namespace_tool(name="mcp__demo", sub_names=("read_file", "list_files")):
    return {
        "type": "namespace",
        "name": name,
        "description": "group",
        "tools": [function_tool(sub_name) for sub_name in sub_names],
    }


class ToolCompatTests(unittest.TestCase):
    def setUp(self):
        self.chat_caps = tool_compat.ProviderToolCapabilities(
            supports_function_tools=True,
            supports_strict_tools=False,
        )
        self.native_caps = tool_compat.ProviderToolCapabilities(
            native_responses_tools=True,
            supports_function_tools=True,
            supports_web_search=True,
            supports_strict_tools=True,
        )

    def test_function_responses_to_chat(self):
        result = tool_compat.translate_tools_for_chat(
            [function_tool()],
            self.chat_caps,
            requested_model="minimaxai/minimax-m3",
            max_tools=128,
            max_schema_bytes=2097152,
        )
        self.assertEqual(result.tools[0]["function"]["name"], "read_file")
        self.assertNotIn("strict", result.tools[0]["function"])

    def test_function_chat_shape_is_accepted(self):
        chat_tool = {"type": "function", "function": function_tool("already") | {"strict": True}}
        result = tool_compat.translate_tools_for_chat(
            [chat_tool],
            self.native_caps,
            requested_model="x",
            max_tools=128,
            max_schema_bytes=2097152,
        )
        self.assertEqual(result.tools[0]["function"]["name"], "already")
        self.assertTrue(result.tools[0]["function"]["strict"])

    def test_namespace_is_flattened_with_reverse_map(self):
        result = tool_compat.translate_tools_for_chat(
            [namespace_tool()],
            self.chat_caps,
            requested_model="minimaxai/minimax-m3",
            max_tools=128,
            max_schema_bytes=2097152,
        )
        names = [tool["function"]["name"] for tool in result.tools]
        self.assertEqual(names, ["mcp__demo__read_file", "mcp__demo__list_files"])
        self.assertEqual(result.reverse_name_map["mcp__demo__read_file"], {"namespace": "mcp__demo", "name": "read_file"})

    def test_multiple_namespaces_collision_is_rejected(self):
        tools = [namespace_tool("mcp__demo", ("read_file",)), namespace_tool("mcp__demo", ("read_file",))]
        with self.assertRaises(tool_compat.ToolCompatibilityError):
            tool_compat.translate_tools_for_chat(
                tools,
                self.chat_caps,
                requested_model="minimaxai/minimax-m3",
                max_tools=128,
                max_schema_bytes=2097152,
            )

    def test_restore_tool_call_name(self):
        reverse = {"mcp__demo__read_file": {"namespace": "mcp__demo", "name": "read_file"}}
        self.assertEqual(
            tool_compat.restore_tool_call_name("mcp__demo__read_file", reverse),
            {"provider_name": "mcp__demo__read_file", "namespace": "mcp__demo", "name": "read_file"},
        )

    def test_web_search_pass_through_native_and_filtered_chat_unless_required(self):
        native = tool_compat.translate_tools_for_native_responses(
            [{"type": "web_search", "name": "web_search"}],
            self.native_caps,
            max_tools=128,
            max_schema_bytes=2097152,
        )
        self.assertEqual(native.tools[0]["type"], "web_search")
        chat = tool_compat.translate_tools_for_chat(
            [function_tool("search_web"), {"type": "web_search", "name": "web_search"}],
            self.chat_caps,
            requested_model="minimaxai/minimax-m3",
            max_tools=128,
            max_schema_bytes=2097152,
        )
        self.assertEqual([tool["function"]["name"] for tool in chat.tools], ["search_web"])
        self.assertTrue(any("dropped hosted web_search" in warning for warning in chat.warnings))
        with self.assertRaises(tool_compat.ToolCompatibilityError):
            tool_compat.translate_tools_for_chat(
                [{"type": "web_search", "name": "web_search"}],
                self.chat_caps,
                requested_model="minimaxai/minimax-m3",
                max_tools=128,
                max_schema_bytes=2097152,
                tool_choice={"type": "web_search"},
            )

    def test_unknown_type_and_limits(self):
        with self.assertRaises(tool_compat.ToolCompatibilityError):
            tool_compat.translate_tools_for_chat(
                [{"type": "unknown"}],
                self.chat_caps,
                requested_model="x",
                max_tools=128,
                max_schema_bytes=2097152,
            )
        with self.assertRaises(tool_compat.ToolCompatibilityError):
            tool_compat.validate_tools([function_tool("a"), function_tool("b")], max_tools=1, max_schema_bytes=2097152)
        with self.assertRaises(tool_compat.ToolCompatibilityError):
            tool_compat.validate_tools([function_tool("a")], max_tools=128, max_schema_bytes=1)

    def test_sanitizer_is_conservative(self):
        schema = function_tool()["parameters"]
        sanitized = tool_compat.sanitize_function_schema(schema, self.chat_caps)
        self.assertIn("enum", sanitized["properties"]["path"])
        self.assertIn("oneOf", sanitized["properties"]["path"])
        self.assertEqual(sanitized["properties"]["path"]["oneOf"][0]["const"], "test.txt")


if __name__ == "__main__":
    unittest.main()
