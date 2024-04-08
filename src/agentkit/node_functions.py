def error_msg_default(prompt, result, error):
    """Default function to append the error message to the prompt.

    Args:
        prompt (list): List of messages in OpenAI format.
        result (str): Result of the LLM query.
        error (str): Error message for the LLM.
    
    Returns:
        prompt (list): List of messages in OpenAI format.
    """
    prompt.append({"role":"assistant", "content":result})
    prompt.append({"role":"user", "content":error})
    return prompt