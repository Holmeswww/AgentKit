import os
try:
    import anthropic
except ImportError:
    raise ImportError("Please install anthropic to use built-in LLM API.")
from .utils import match_model
import time
from .base import BaseModel

if os.environ.get("ANTHROPIC_KEY") is None:
    print("Environment variable for Anthropic key not found, using Anthropic API key from ~/.openai/openai.key.")
    if not os.path.exists(os.path.join(os.path.expanduser('~'), ".openai/openai.key")):
        raise FileNotFoundError("Please create a file at ~/.openai/openai.key with your OpenAI API key and organization ID. The third line should be your anthropic API key.")
    else:
        with open(os.path.join(os.path.expanduser('~'), ".openai/openai.key"), 'r') as f:
            org_key = f.readlines()
            ANTHROPIC_KEY = org_key[2].strip()
else:
    print("Using Anthropic API key from environment variable.")
    ANTHROPIC_KEY = os.environ.get("ANTHROPIC_KEY")

client = anthropic.Anthropic(
    api_key=ANTHROPIC_KEY,
)

class Claude_chat(BaseModel):

    def __init__(self, model_name, global_counter=None):
        model_max, _ = match_model(model_name)
        self.model_max = model_max
        self.tokenizer = client.get_tokenizer()
        super().__init__(model_name, global_counter, "chat")

    def encode(self, txt):
        return self.tokenizer.encode(txt)

    def decode(self, txt):
        return self.tokenizer.decode(txt)

    def convert_anthropic(self, msg, shrink_idx):
        system = None
        if msg[0]['role'] == 'system':
            system = msg[0]['content']
            msg = msg[1:]
            shrink_idx -= 1
        
        message = []
        role = "user"
        added_terms = 0
        new_index = shrink_idx
        for i,m in enumerate(msg):
            if role == 'user':
                if m['role'] in ('system', 'user'):
                    message.append({"role":"user", "content":m['content']})
                else:
                    message.append({"role":"user", "content":""})
                    added_terms+=1
                    message.append({"role":"assistant", "content":m['content']})
            elif role == 'assistant':
                if m['role'] in ('system', 'user'):
                    message.append({"role":"assistant", "content":"Understood."})
                    added_terms+=1
                    message.append({"role":"user", "content":m['content']})
                else:
                    message.append({"role":"assistant", "content":m['content']})

            if i == shrink_idx:
                new_index = shrink_idx + added_terms
            
            if message[-1]['role'] == 'user':
                role = 'assistant'
            else:
                role = 'user'
        
        return message, system, new_index

    
    def query_chat(self, messages, shrink_idx, max_gen=512, temp=0.):
        model_max = self.model_max
        messages, system, shrink_idx = self.convert_anthropic(messages, shrink_idx)
        messages = self.shrink_msg(messages, shrink_idx, model_max-max_gen)
        while(True):
            try:
                message = client.messages.create(
                    model=self.name,
                    system=system,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_gen,
                )
                return message.content[0].text, {"prompt":message.usage.input_tokens, "completion":message.usage.output_tokens, "total":message.usage.input_tokens+message.usage.output_tokens}
            except Exception as e:
                if self.debug:
                    raise e
                elif isinstance(e, anthropic.APIConnectionError) or isinstance(e, anthropic.APIStatusError) or isinstance(e, anthropic.InternalServerError):
                    time.sleep(30)
                elif isinstance(e, anthropic.RateLimitError):
                    time.sleep(5*60)
                elif "However, your messages resulted in" in str(e):
                    print("error:", e, str(e))
                    e = str(e)
                    index = e.find("your messages resulted in ")
                    import re
                    val = int(re.findall(r'\d+', e[index + len("your messages resulted in ") : ])[0])
                    index2 = e.find("maximum context length is ")
                    model_max = int(re.findall(r'\d+', e[index2 + len("maximum context length is "):])[0])
                    messages = self.shrink_msg_by(messages, shrink_idx, val-model_max)
                else:
                    time.sleep(5)
                    print(e)