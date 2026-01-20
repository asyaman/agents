import re


def extract_json_subtexts(text: str):
    """ " Extracts json objects from text
    Args:
        text: text to be extracted
    """
    json_pattern = r"```json(.*)```"
    # llm are verbose and return additional text prior and aftre the josn output
    # llms sometimes are confused and return multiple json outputs, pick first
    matches = re.findall(json_pattern, text, flags=re.DOTALL)
    # llms returns incomplete outputs when the context lenght is reached
    partial_matches = re.findall("```json(.*)", text, flags=re.DOTALL)
    if matches:
        parsed = matches[0].strip()
    elif partial_matches:
        parsed = partial_matches[0].strip()
    else:
        parsed = text
    return parsed


def normalize_tool_name(tool_name: str) -> str:
    return tool_name.upper().replace(" ", "_")
