
import time
import os
from .utils import match_model
from .base import BaseModel
import requests

with open(os.path.join(os.path.expanduser('~'), ".openai/openai.key"), 'r') as f:
    org_key = f.readlines()
    yintat_key = org_key[2].strip()


class GPT_chat(BaseModel):

    def __init__(self, model_name, global_counter=None):
        model_name = model_name.replace("ms-", "")
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
            response_text = ""
            try:
                url = "http://gateway.phyagi.net/api/chat/completions"
                header = {
                    'Authorization': yintat_key,
                    "Content-Type": "application/json",
                }

                payload = {
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_gen,
                    "model": self.name,
                    }
                
                response = requests.post(url, headers=header, json=payload)
                response_text = response.text

                error_patterns = ["Max retries exceeded", "call rate limit", "Rate limit reached", "exceeded token rate limit", "NoCapacity", "Provider returned error", "token rate limit", "RateLimitReached"]
                if any([pattern in response_text for pattern in error_patterns]):
                    print("error:", response_text)
                    time.sleep(5)
                    continue

                response_json = response.json()
                completion_content = response_json["choices"][-1]["message"]['content']
                usage = {
                    "prompt": response_json["usage"]["prompt_tokens"],
                    "completion": response_json["usage"]["completion_tokens"],
                }
                return completion_content, usage
            except Exception as e:
                if self.debug:
                    raise e
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
                    print("error:", e, response_text)
                    print("retrying in 5 seconds")
                    time.sleep(5)



# import os
# import requests
# import base64

# # Configuration
# API_KEY = "YOUR_API_KEY"
# IMAGE_PATH = "YOUR_IMAGE_PATH"
# encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
# headers = {
#     "Content-Type": "application/json",
#     "api-key": API_KEY,
# }

# # Payload for the request
# payload = {
#   "messages": [
#     {
#       "role": "system",
#       "content": [
#         {
#           "type": "text",
#           "text": "You are an AI assistant that helps people find information."
#         }
#       ]
#     }
#   ],
#   "temperature": 0.7,
#   "top_p": 0.95,
#   "max_tokens": 800
# }

# ENDPOINT = "https://infinity-aoai-east-us-2.openai.azure.com/openai/deployments/gpt-4o-240513/chat/completions?api-version=2024-02-15-preview"

# # Send request
# try:
#     response = requests.post(ENDPOINT, headers=headers, json=payload)
#     response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
# except requests.RequestException as e:
#     raise SystemExit(f"Failed to make the request. Error: {e}")

# # Handle the response as needed (e.g., print or process)
# print(response.json())