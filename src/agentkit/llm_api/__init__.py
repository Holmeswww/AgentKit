token_counter = {
    "token_completion":{},
    "token_prompt":{},
    "api_calls":{}
}

def get_token_counts():
    global token_counter
    return token_counter

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

def query_llama_chat(model,ollama_url, tokenmodel_path):
    from .ollama import Ollama_chat
    global token_counter
    query_model = Ollama_chat(model, ollama_url, tokenmodel_path, token_counter)
    return query_model

def get_query(LLM_name, ollama_url=None, tokenmodel_path=None):
    """Get the query model for the specified LLM_name.

    Currently supported LLMs:

    - GPT-4

    - GPT-3.5

    - Claude

    - Ollama
    
    Args:
        LLM_name (str): Name of the LLM model
        tokenmodel_path (str): Path to the token model
    
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
    elif LLM_name.lower().startswith("ollama"):
        model_name = LLM_name.replace('ollama:','')
        token_counter["token_completion"][model_name] = 0
        token_counter["token_prompt"][model_name] = 0
        token_counter["api_calls"][model_name] = 0
        return query_llama_chat(model=LLM_name,ollama_url=ollama_url,tokenmodel_path=tokenmodel_path)
    else:
        raise NotImplementedError("LLM {} not implemented".format(LLM_name))