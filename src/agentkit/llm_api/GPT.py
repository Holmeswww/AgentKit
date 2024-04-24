try:
    import openai
except ImportError:
    raise ImportError("Please install openai to use built-in LLM API.")
from openai import OpenAI, AzureOpenAI
import time
import os
from .utils import match_model
from .base import BaseModel

def initialize_client():
    if os.environ.get("OPENAI_KEY"):
        return OpenAI(
            api_key=os.environ["OPENAI_KEY"],
            organization=os.environ["OPENAI_ORG"],
        )
    elif os.environ.get("AZURE_OPENAI_API_KEY"):
        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        ), os.environ.get("AZURE_DEPLOYMENT_NAME")

    openai_key_path = os.path.join(os.path.expanduser('~'), ".openai", "openai.key")
    azure_key_path = os.path.join(os.path.expanduser('~'), ".openai", "azure_openai.key")

    if os.path.exists(openai_key_path):
        with open(openai_key_path, 'r') as f:
            lines = f.read().splitlines()
            if len(lines) >= 2:
                return OpenAI(
                    api_key=lines[0].strip(),
                    organization=lines[1].strip(),
                )
        raise FileNotFoundError(
            """
            Please create a file at ~/.openai/openai.key with your OpenAI API key and organization ID. 
            The first line should be your API key and the second line should be your organization ID.
            """
        )
    
    if os.path.exists(azure_key_path):
        with open(azure_key_path, 'r') as f:
            lines = f.read().splitlines()
            if len(lines) >= 3:
                return AzureOpenAI(
                    api_key=lines[0].strip(),
                    api_version=lines[1].strip(),
                    azure_endpoint=lines[2].strip(),
                ), lines[3].strip()

        raise FileNotFoundError(
            """
            Please create a file at ~/.openai/azure_openai.key with your Azure OpenAI API key, 
            API version, Azure endpoint, and deployment name. 
            The first line should be your API key, the second line should be your API version, 
            the third line should be your Azure endpoint, and the fourth line should be your deployment name.
            """
        )

client_and_deployment = initialize_client()
client = client_and_deployment[0] if isinstance(client_and_deployment, tuple) else client_and_deployment
deployment_name = client_and_deployment[1] if isinstance(client_and_deployment, tuple) else None

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
                if isinstance(client, OpenAI):
                    completion = client.chat.completions.create(
                        model=self.name,
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_gen,
                    )
                elif isinstance(client, AzureOpenAI):
                    completion = client.completions.create(
                        model=deployment_name,
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_gen,
                    )
                return completion.choices[0].message.content, {"prompt":completion.usage.prompt_tokens, "completion":completion.usage.completion_tokens, "total":completion.usage.total_tokens}
            except Exception as e:
                if self.debug:
                    raise e
                elif isinstance(e, openai.RateLimitError) or isinstance(e, openai.APIStatusError) or isinstance(e, openai.APITimeoutError) or isinstance(e, openai.APIConnectionError) or isinstance(e, openai.InternalServerError):
                    time.sleep(30)
                    print(e)
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
                    print("error:", e)
                    print("retrying in 5 seconds")
                    time.sleep(5)
