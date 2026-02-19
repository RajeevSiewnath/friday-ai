def get_sections():
    return {
        "sections": [
            "profile",
            "work experience",
            "education",
            "other projects",
            "skills",
        ]
    }


get_sections_tool = {
    "type": "function",
    "name": "get_sections",
    "description": "Returns a list of available CV sections.",
    "parameters": {"type": "object", "properties": {}, "required": []},  # no inputs
    "returns": {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The available CV sections.",
            }
        },
        "required": ["sections"],
    },
}
