try:
    import llama
except ImportError:
    raise ImportError("Please install llama to use Ollama LLM.")
from typing import List
from llama import Tokenizer
import time
import requests, json
import os
from .utils import match_model
from .base import BaseModel

def initialize_client():
    ollama_modelpath = os.path.join(os.path.expanduser('~'), ".ollama", "ollama_model.info")
    if os.environ.get("OLLAMA_URL"):
        tokenmodel_path = os.environ.get("OLLAMA_TOKENIZER_PATH")
        ollama_url = os.environ.get("OLLAMA_URL")
        return tokenmodel_path, ollama_url
    elif os.path.exists(ollama_modelpath):
        with open(ollama_modelpath, 'r') as f:
            lines = f.read().splitlines()
            if len(lines) >= 2:
                tokenmodel_path = lines[0].strip()
                ollama_url = lines[1].strip()
                return tokenmodel_path, ollama_url
        raise FileNotFoundError(
            """
            Please create a file at ~/.ollama/ollama_model.info with your Ollama tokenizer model path and Ollama URL.
            The first line should be your tokenizer model path and the second line should be your Ollama URL.
            """
        )
    else:
        raise FileNotFoundError(
            """
            Please set the environment variables OLLAMA_URL and OLLAMA_TOKENIZER_PATH or create a file at ~/.ollama/ollama_model.info with your Ollama tokenizer model path and Ollama URL.
            """
        )

class Ollama_chat(BaseModel):

    def __init__(self, model_name, global_counter=None):

        tokenmodel_path, ollama_url = initialize_client()
        model_max, enc_fn = match_model(model_name)
        model_name = model_name.replace('ollama:','')
        self.model_name = model_name
        self.name = model_name
        self.sp_model = Tokenizer(tokenmodel_path)
        self.url = f'{ollama_url}/api/chat'
        self.model_max = model_max
        super().__init__(model_name, global_counter, "chat")
    
    def encode(self, txt, bos: bool = True, eos: bool = False) -> List[int]:
        return self.sp_model.encode(txt,bos=bos,eos=eos)

    def decode(self, txt, bos: bool = True, eos: bool = False) -> str:
        return self.sp_model.decode(txt,bos,eos)

    def query_chat(self, messages, shrink_idx, max_gen=512, temp=0.):
        model_max = self.model_max
        messages = self.shrink_msg(messages, shrink_idx, model_max-max_gen)
        while(True):
            try:
                ollama_body = {"messages": messages, "model": self.name, "stream": False}
                completion = requests.post(self.url,json=ollama_body)
                print(ollama_body)
                response = json.loads(completion.content.decode('utf-8'))
                content = response['message']['content']

                completion_count = self.count_tokens(content)
                prompt_count = self.count_tokens(messages)
                total_tokens = prompt_count + completion_count
                return content, {"prompt":prompt_count, "completion":completion_count, "total":total_tokens}
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