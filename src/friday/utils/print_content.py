def print_content(content):
    if isinstance(content, list):
        return "\n".join([print_content(c) for c in content])
    elif isinstance(content, dict):
        return content["text"]
    elif isinstance(content, str):
        return content
    else:
        return ""