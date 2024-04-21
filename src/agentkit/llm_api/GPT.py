try:
    import openai
except ImportError:
    raise ImportError("Please install openai to use built-in LLM API.")
from openai import OpenAI
import time
import os
from .utils import match_model
from .base import BaseModel

if os.environ.get("OPENAI_KEY") is None:
    print("Environment variable for OpenAI key not found, using OpenAI API key from ~/.openai/openai.key.")
    if not os.path.exists(os.path.join(os.path.expanduser('~'), ".openai/openai.key")):
        raise FileNotFoundError("Please create a file at ~/.openai/openai.key with your OpenAI API key and organization ID. The first line should be your API key and the second line should be your organization ID.")
    else:
        with open(os.path.join(os.path.expanduser('~'), ".openai/openai.key"), 'r') as f:
            org_key = f.readlines()
            OpenAI_KEY = org_key[0].strip()
            OpenAI_ORG = org_key[1].strip()
else:
    print("Using OpenAI API key from environment variable.")
    OpenAI_KEY = os.environ.get("OPENAI_KEY")
    OpenAI_ORG = os.environ.get("OPENAI_ORG")
client = OpenAI(
    api_key=OpenAI_KEY,
    organization=OpenAI_ORG,
)

class GPT_chat(BaseModel):

    def __init__(self, model_name, global_counter=None):
        model_max, enc_fn = match_model(model_name)
        self.enc_fn = enc_fn
        self.model_max = model_max
        super().__init__(model_name, global_counter, "chat")
    
    def encode(self, txt):
        return self.enc_fn.encode(txt)

    def decode(self, txt):
        return self.enc_fn.decode(txt)

    def query_chat(self, messages, shrink_idx, max_gen=512, temp=0.):
        model_max = self.model_max
        messages = self.shrink_msg(messages, shrink_idx, model_max-max_gen)
        while(True):
            try:
                completion = client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_gen,
                )
                return completion.choices[0].message.content, {"prompt":completion.usage.prompt_tokens, "completion":completion.usage.completion_tokens, "total":completion.usage.total_tokens}
            except (openai.RateLimitError, openai.APIStatusError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                time.sleep(30)
            except Exception as e:
                e = str(e)
                if "However, your messages resulted in" in e:
                    print("error:", e)
                    index = e.find("your messages resulted in ")
                    import re
                    val = int(re.findall(r'\d+', e[index + len("your messages resulted in ") : ])[0])
                    index2 = e.find("maximum context length is ")
                    model_max = int(re.findall(r'\d+', e[index2 + len("maximum context length is "):])[0])
                    messages = self.shrink_msg_by(messages, shrink_idx, val-model_max)
                else:
                    time.sleep(5)
                    print(e)