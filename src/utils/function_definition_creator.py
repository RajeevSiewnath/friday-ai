import inspect
import typing
from typing import get_type_hints, get_origin, get_args, Literal, Union


def _parse_google_docstring_with_returns(docstring: str):
    if not docstring:
        return {}, "", None

    lines = docstring.splitlines()
    param_docs = {}
    description_lines = []
    return_desc = None
    in_args = False
    in_returns = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Args:"):
            in_args = True
            in_returns = False
            continue
        elif stripped.startswith("Returns:"):
            in_args = False
            in_returns = True
            continue

        if in_args:
            if not stripped:
                continue
            if ":" in stripped:
                name, desc = stripped.split(":", 1)
                param_docs[name.strip()] = desc.strip()
        elif in_returns:
            if return_desc is None:
                return_desc = stripped
            else:
                return_desc += " " + stripped
        else:
            description_lines.append(stripped)

    description = " ".join(description_lines).strip()
    return param_docs, description, return_desc


def _python_type_to_schema(py_type):
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_schema(non_none[0])
        return {"anyOf": [_python_type_to_schema(a) for a in non_none]}

    if origin is Literal:
        return {"type": "string", "enum": list(args)}

    if origin in (list, typing.List):
        item_type = args[0] if args else str
        return {"type": "array", "items": _python_type_to_schema(item_type)}

    if origin in (dict, typing.Dict):
        return {"type": "object", "additionalProperties": True}

    mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}
    return {"type": mapping.get(py_type, "string")}


def function_definition_creator(func_or_instance):
    """
    Accepts either:
      - a plain function
      - a Pipeline instance (uses its `run` method)
    """
    from pipelines.pipeline import Pipeline

    if isinstance(func_or_instance, Pipeline):
        # It's a Pipeline instance — use its bound run method
        func = func_or_instance.run
        name = type(func_or_instance).__name__
    elif callable(func_or_instance):
        func = func_or_instance
        name = func.__name__
    else:
        raise TypeError("Argument must be a function or a Pipeline instance")

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    docstring = inspect.getdoc(func) or ""
    param_docs, description, return_desc = _parse_google_docstring_with_returns(
        docstring
    )

    properties = {}
    required = []

    for param in sig.parameters.values():
        param_type = type_hints.get(param.name, str)
        schema = _python_type_to_schema(param_type)
        if param.name in param_docs:
            schema["description"] = param_docs[param.name]
        properties[param.name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(param.name)

    # Handle return type
    return_type = type_hints.get("return", None)
    return_schema = (
        _python_type_to_schema(return_type) if return_type else {"type": "string"}
    )
    if return_desc:
        return_schema["description"] = return_desc

    tool_spec = {
        "type": "function",
        "name": name,
        "description": description or "No description provided.",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
        # "returns": return_schema,
    }

    return tool_spec
