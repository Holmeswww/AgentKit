import json
def extract_json_objects(input_string):
    try:
        # Initialize variables
        json_objects = []
        stack = []
        start_index = -1
        
        # Helper function to try to decode a JSON object string
        def try_decode_json(json_str):
            try:
                # Attempt to parse the string as JSON
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If an error occurs, return None
                return None

        # Iterate over each character and its index in the input string
        for index, char in enumerate(input_string):
            # Check for opening braces/brackets
            if char in '{[':
                # If this is the first opening brace/bracket, take note of the index
                if not stack:
                    start_index = index
                # Push the character onto the stack
                stack.append(char)
            
            # Check for closing braces/brackets
            elif char in '}]':
                if not stack:
                    continue  # Ignore unmatched closing braces/brackets

                # Check if the closing brace/bracket matches the last opening one
                if (char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '['):
                    stack.pop()  # Pop from stack for a valid closing match
                    # When stack is empty, we have a full JSON string
                    if not stack:
                        # Extract the JSON string and try to decode it
                        json_str = input_string[start_index:index+1]
                        json_obj = try_decode_json(json_str)
                        # If successfully decoded, add to the json_objects list
                        if json_obj is not None:
                            json_objects.append(json_obj)
                else:
                    continue  # Skip unmatched closing brace/bracket
        if len(json_objects)==0:
            return None, "Error: No json objects found"
        return json_objects, None
    except Exception as e:
        return None, "Error: {}\nTraceback: {}".format(e, traceback.format_exc())