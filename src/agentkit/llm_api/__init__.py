token_counter = {
    "token_completion":{},
    "token_prompt":{},
    "api_calls":{}
}

def get_token_counts():
    global token_counter
    return token_counter

def get_query(LLM_name):
    global token_counter
    token_counter["token_completion"][LLM_name] = 0
    token_counter["token_prompt"][LLM_name] = 0
    token_counter["api_calls"][LLM_name] = 0

def query_gpt_chat(model):
    from .GPT import GPT_chat
    global token_counter
    query_model = GPT_chat(model, token_counter)
    return query_model

def query_claude_chat(model):
    from .claude import Claude_chat
    global token_counter
    query_model = Claude_chat(model, token_counter)
    return query_model

def get_query(LLM_name):
    """Get the query model for the specified LLM_name.

    Currently supported LLMs:

    - GPT-4

    - GPT-3.5

    - Claude
    
    Args:
        LLM_name (str): Name of the LLM model
    
    Returns:
        LLM_API_FUNCTION: The query model for the specified LLM_name
    """
    global token_counter
    token_counter["token_completion"][LLM_name] = 0
    token_counter["token_prompt"][LLM_name] = 0
    token_counter["api_calls"][LLM_name] = 0

    if LLM_name.lower().startswith("gpt-4") or LLM_name.lower().startswith("gpt-3.5"):
        return query_gpt_chat(model=LLM_name)
    elif LLM_name.lower().startswith("claude"):
        return query_claude_chat(model=LLM_name)
    else:
        raise NotImplementedError("LLM {} not implemented".format(LLM_name))