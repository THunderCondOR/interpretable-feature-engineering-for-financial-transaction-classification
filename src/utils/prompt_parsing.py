import json
import re

def extract_last_json(s):
    """
    Extract the last JSON object or array from a string and parse it.
    Returns the parsed Python object, or None if no JSON is found.
    """
    # Regex to find JSON arrays or objects
    json_pattern = re.compile(r'(\{.*?\}|\[.*?\])', re.DOTALL)
    matches = json_pattern.findall(s)
    
    if not matches:
        return None
    
    # Take the last match
    last_json_str = matches[-1]
    
    try:
        return json.loads(last_json_str)
    except json.JSONDecodeError:
        # Sometimes regex captures partial JSON, try to fix by trimming
        # Find the last closing bracket
        for i in range(len(last_json_str), 0, -1):
            try:
                return json.loads(last_json_str[:i])
            except:
                continue
        return None

def extract_last_boxed_text(s):
    matches = re.findall(r'\\boxed\{(.*?)\}', s)

    return matches[-1] if matches else None