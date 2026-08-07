from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any


FUNCTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


class ToolCompatibilityError(Exception):
    pass


@dataclass(frozen=True)
class ProviderToolCapabilities:
    native_responses_tools: bool = False
    supports_function_tools: bool = False
    supports_namespace_tools: bool = False
    supports_web_search: bool = False
    supports_strict_tools: bool = False
    supports_parallel_tool_calls: bool = False


@dataclass
class ToolTranslationResult:
    tools: list
    reverse_name_map: dict[str, dict[str, str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def capabilities_from_config(cfg: dict, *, native_responses: bool = False) -> ProviderToolCapabilities:
    return ProviderToolCapabilities(
        native_responses_tools=bool(native_responses or cfg.get("native_responses_provider")),
        supports_function_tools=bool(cfg.get("supports_tools")),
        supports_namespace_tools=bool(cfg.get("supports_namespace_tools")),
        supports_web_search=bool(cfg.get("supports_web_search")),
        supports_strict_tools=bool(cfg.get("supports_strict_tools")),
        supports_parallel_tool_calls=bool(cfg.get("supports_parallel_tool_calls")),
    )


def validate_function_name(name: Any) -> str:
    name_s = str(name or "")
    if not FUNCTION_NAME_RE.match(name_s):
        raise ToolCompatibilityError("Function tool name is invalid")
    return name_s


def inspect_tools(tools: list | None, *, max_tools: int | None = None, max_schema_bytes: int | None = None) -> dict:
    tool_list = tools or []
    sizes = [json_bytes(tool) for tool in tool_list]
    tool_types = [
        str(tool.get("type") or "<missing>") if isinstance(tool, dict) else type(tool).__name__
        for tool in tool_list
    ]
    unknown = sorted({typ for typ in tool_types if typ not in {"function", "namespace", "web_search", "web_search_preview"}})
    return {
        "tool_count": len(tool_list),
        "tool_types": sorted(set(tool_types)),
        "tools_size_bytes": json_bytes(tool_list),
        "max_tool_size_bytes": max(sizes, default=0),
        "namespace_count": sum(1 for typ in tool_types if typ == "namespace"),
        "flattened_function_count": 0,
        "web_search_count": sum(1 for typ in tool_types if typ in {"web_search", "web_search_preview"}),
        "unknown_types": unknown,
        "max_tools": max_tools,
        "max_tool_schema_bytes": max_schema_bytes,
    }


def inspect_tool_structures(tools: list | None) -> list[dict]:
    summary: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            summary.append({"type": type(tool).__name__, "keys": [], "name": None, "subtool_count": None, "json_bytes": json_bytes(tool)})
            continue
        subtool_count = None
        for key in ("tools", "functions", "members"):
            if isinstance(tool.get(key), list):
                subtool_count = len(tool[key])
                break
        summary.append({
            "type": str(tool.get("type") or "<missing>"),
            "keys": sorted(str(key) for key in tool.keys()),
            "name": tool.get("name") or tool.get("tool_name") or tool.get("namespace"),
            "subtool_count": subtool_count,
            "json_bytes": json_bytes(tool),
        })
    return summary


def validate_tools(tools: list | None, *, max_tools: int, max_schema_bytes: int) -> None:
    tool_list = tools or []
    if len(tool_list) > max_tools:
        raise ToolCompatibilityError(f"Too many tools: max {max_tools}")
    if json_bytes(tool_list) > max_schema_bytes:
        raise ToolCompatibilityError("Tool schema exceeds configured byte limit")


def _safe_tool_name_part(value: Any) -> str:
    part = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip()).strip("_")
    if not part or not re.match(r"^[A-Za-z_]", part):
        part = f"tool_{part}" if part else "tool"
    return part


def flattened_namespace_tool_name(namespace: str, name: str) -> str:
    base = f"{_safe_tool_name_part(namespace)}__{_safe_tool_name_part(name)}"
    if len(base) <= 64:
        return base
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:10]
    keep = max(1, 64 - len(digest) - 1)
    return f"{base[:keep]}_{digest}"


def _ensure_unique(name: str, seen: set[str]) -> str:
    if name in seen:
        raise ToolCompatibilityError(f"Tool name collision after conversion: {name}")
    seen.add(name)
    return name


def _responses_function_tool(tool: dict, *, override_name: str | None, capabilities: ProviderToolCapabilities, warnings: list[str]) -> dict:
    source = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = validate_function_name(override_name or source.get("name"))
    parameters = source.get("parameters")
    if not isinstance(parameters, dict):
        raise ToolCompatibilityError("Function tool parameters must be a JSON Schema object")
    fn = {
        "name": name,
        "description": str(source.get("description") or ""),
        "parameters": sanitize_function_schema(parameters, capabilities),
    }
    strict_source = source if "strict" in source else tool
    if "strict" in strict_source:
        if capabilities.supports_strict_tools:
            fn["strict"] = bool(strict_source.get("strict"))
        else:
            warnings.append(f"dropped strict for tool {name}: provider does not support strict tools")
    return {"type": "function", "function": fn}


def _tool_choice_explicitly_targets_web_search(tool_choice: Any) -> bool:
    if not isinstance(tool_choice, dict):
        return False
    choice_type = str(tool_choice.get("type") or "")
    if choice_type in ("web_search", "web_search_preview"):
        return True
    name = tool_choice.get("name")
    if name is None and isinstance(tool_choice.get("function"), dict):
        name = tool_choice["function"].get("name")
    return str(name or "") in ("web_search", "web_search_preview")


def translate_tools_for_chat(
    tools: list | None,
    capabilities: ProviderToolCapabilities,
    *,
    requested_model: str,
    max_tools: int,
    max_schema_bytes: int,
    tool_choice: Any = None,
) -> ToolTranslationResult:
    if not tools:
        return ToolTranslationResult(tools=[])
    validate_tools(tools, max_tools=max_tools, max_schema_bytes=max_schema_bytes)
    if not capabilities.supports_function_tools:
        raise ToolCompatibilityError(f"Model '{requested_model}' does not declare supports_tools=true")

    converted: list[dict] = []
    reverse_name_map: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise ToolCompatibilityError("Responses tools must be objects")
        tool_type = tool.get("type")
        if tool_type == "function":
            name_source = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            _ensure_unique(validate_function_name(name_source.get("name")), seen)
            converted.append(_responses_function_tool(tool, override_name=None, capabilities=capabilities, warnings=warnings))
            continue
        if tool_type == "namespace":
            if capabilities.supports_namespace_tools:
                raise ToolCompatibilityError("Native namespace tools are not valid for Chat Completions translation")
            namespace = validate_function_name(tool.get("name"))
            subtools = tool.get("tools")
            if not isinstance(subtools, list) or not subtools:
                raise ToolCompatibilityError("Namespace tool must contain a non-empty tools list")
            for subtool in subtools:
                if not isinstance(subtool, dict):
                    raise ToolCompatibilityError("Namespace subtools must be objects")
                if subtool.get("type") != "function":
                    raise ToolCompatibilityError("Only function subtools are supported inside namespace tools")
                original_name = validate_function_name(subtool.get("name"))
                provider_name = _ensure_unique(flattened_namespace_tool_name(namespace, original_name), seen)
                reverse_name_map[provider_name] = {"namespace": namespace, "name": original_name}
                converted.append(_responses_function_tool(subtool, override_name=provider_name, capabilities=capabilities, warnings=warnings))
            continue
        if tool_type in ("web_search", "web_search_preview"):
            if capabilities.supports_web_search:
                raise ToolCompatibilityError("Native web_search tools are not valid for Chat Completions translation")
            if _tool_choice_explicitly_targets_web_search(tool_choice):
                raise ToolCompatibilityError(f"Tool choice explicitly requires hosted {tool_type}, which cannot be transported faithfully to Chat Completions for model '{requested_model}'")
            warnings.append(f"dropped hosted {tool_type}: provider does not support native web search tools")
            continue
        raise ToolCompatibilityError(f"Unsupported tool type '{tool_type}'")

    if json_bytes(converted) > max_schema_bytes:
        raise ToolCompatibilityError("Tool schema exceeds configured byte limit after conversion")
    return ToolTranslationResult(tools=converted, reverse_name_map=reverse_name_map, warnings=warnings)


def translate_tools_for_native_responses(
    tools: list | None,
    capabilities: ProviderToolCapabilities,
    *,
    max_tools: int,
    max_schema_bytes: int,
) -> ToolTranslationResult:
    if not tools:
        return ToolTranslationResult(tools=[])
    validate_tools(tools, max_tools=max_tools, max_schema_bytes=max_schema_bytes)
    if not capabilities.native_responses_tools:
        raise ToolCompatibilityError("Provider does not support native Responses tools")
    native: list[dict] = []
    warnings: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ToolCompatibilityError("Responses tools must be objects")
        if tool.get("type") == "function":
            source = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            native_tool = {"type": "function", "name": validate_function_name(source.get("name"))}
            if "description" in source:
                native_tool["description"] = str(source.get("description") or "")
            if "parameters" in source:
                native_tool["parameters"] = sanitize_function_schema(source.get("parameters") or {}, capabilities)
            if "strict" in source:
                if capabilities.supports_strict_tools:
                    native_tool["strict"] = bool(source.get("strict"))
                else:
                    warnings.append(f"dropped strict for tool {native_tool['name']}: provider does not support strict tools")
            native.append(native_tool)
        else:
            native.append(dict(tool))
    if json_bytes(native) > max_schema_bytes:
        raise ToolCompatibilityError("Tool schema exceeds configured byte limit after conversion")
    return ToolTranslationResult(tools=native, warnings=warnings)


def sanitize_function_schema(schema: dict, capabilities: ProviderToolCapabilities) -> dict:
    if not isinstance(schema, dict):
        raise ToolCompatibilityError("Function tool parameters must be a JSON Schema object")
    return dict(schema)


def restore_tool_call_name(provider_name: str, reverse_name_map: dict | None) -> dict[str, str]:
    if reverse_name_map and provider_name in reverse_name_map:
        entry = reverse_name_map[provider_name]
        return {"provider_name": provider_name, "namespace": str(entry["namespace"]), "name": str(entry["name"])}
    return {"provider_name": provider_name, "name": provider_name}
