import numpy as np
try:
    import tiktoken
except ImportError:
    raise ImportError("Please install tiktoken to use built-in LLM API.")
import difflib


enc_fns = {
    "gpt-4":tiktoken.encoding_for_model("gpt-4"),
    "gpt-4o":tiktoken.encoding_for_model("gpt-4"),
    "gpt-4-turbo":tiktoken.encoding_for_model("gpt-4"),
    "gpt-4-1106-preview":tiktoken.encoding_for_model("gpt-4"),
    "gpt-4-32k-0613":tiktoken.encoding_for_model("gpt-4"),
    "gpt-3.5-turbo":tiktoken.encoding_for_model("gpt-3.5-turbo"),
    "gpt-3.5-turbo-1106":tiktoken.encoding_for_model("gpt-3.5-turbo"),
    "gpt-3.5-turbo-0125":tiktoken.encoding_for_model("gpt-3.5-turbo"),
    "claude-3": None,
    "claude-2.1": None,
    "ollama": None,
}
model_maxes = {
    "gpt-4":8192,
    "gpt-4o":128000,
    "gpt-4-turbo":128000,
    "gpt-4-1106-preview":128000,
    "gpt-4-32k-0613":32768,
    "gpt-3.5-turbo":16385,
    "gpt-3.5-turbo-1106":16385,
    "gpt-3.5-turbo-0125":16385,
    "claude-3":50000,
    "claude-2.1":50000,
    "ollama":8192,
}

def match_model(model_name):

    # match model name to the closest string in enc_fns.keys()
    model_name = model_name.lower()
    if model_name.startswith('ollama'):
        model_name = 'ollama'
    
    matches = []
    matches += [model for model in model_maxes.keys() if model_name.startswith(model)]
    # sort by length of match
    matches = sorted(matches, key=lambda x: len(x), reverse=True)
    if len(matches) == 0:
        matches += difflib.get_close_matches(model_name, enc_fns.keys())
    if len(matches) == 0:
        raise ValueError("Model name {} not found!".format(model_name))
    else:
        model = matches[0]
    print("Matched model_name {} to: {}. If this match is not accurate, then the token counter will not function properly.".format(model_name, model))

    model_max = model_maxes[model]
    enc_fn = enc_fns[model]
    return model_max, enc_fn
