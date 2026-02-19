def get_section(section):
    sections_data = {
        "profile": {"section_info": "profile info"},
        "work experience": {"section_info": "work experience info"},
        "education": {"section_info": "education info"},
        "other projects": {"section_info": "other projects info"},
        "skills": {"section_info": "skills info"},
    }
    info = sections_data.get(section, {"section_info": None})
    return {"section": section, **info}


get_section_tool = {
    "type": "function",
    "name": "get_section",
    "description": "Returns detailed information for a specified CV section.",
    "parameters": {
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "enum": [
                    "profile",
                    "work experience",
                    "education",
                    "other projects",
                    "skills",
                ],
                "description": "The section to retrieve information for.",
            }
        },
        "required": ["section"],
    },
    "returns": {
        "type": "object",
        "properties": {
            "section": {"type": "string", "description": "The section requested"},
            "section_info": {
                "type": "string",
                "description": "Detailed info about the section",
            },
        },
        "required": ["section", "section_info"],
    },
}
