import numpy as np

class BaseModel:

    max_gen_default = 1024
    model_max = 2048

    def __init__(self, model_name, global_conter=None, model_type = "chat"):
        self.name = model_name
        assert model_type in ["chat", "completion"], "type should be either 'chat' or 'completion'"
        self.type = model_type
        self.global_counter = global_conter
        self.debug = False
    
    def query_chat(self, messages, shrink_idx, model, max_gen=1024, temp=0.):
        raise NotImplementedError
    
    def query_completion(self, messages, shrink_idx, model, max_gen=1024, temp=0.):
        raise NotImplementedError
    
    def encode(self, txt):
        print("Warning: encode is not implemented for this model, returning words.")
        return txt.split(" ")

    def decode(self, txt):
        print("Warning: decode is not implemented for this model, returning concatenation of input.")
        return " ".join(txt)
    
    def count_tokens(self, text):
        if type(text) == list:
            return min(sum([len(self.encode(x['content'])) for x in text]), self.model_max)
        elif type(text) == str:
            return min(len(self.encode(text)), self.model_max)
        else:
            raise ValueError("Unknown type for text: {}".format(type(text)))

    def __call__(self, msg, shrink_idx, max_gen=None, temp=0.):
        if max_gen is None:
            max_gen = self.max_gen_default

        assert type(msg) == list, "msg should be a list in openai format"
        
        if self.type == "chat":
            result, usage = self.query_chat(msg, shrink_idx, max_gen, temp)
        else:
            result, usage = self.query_completion(msg, shrink_idx, max_gen, temp)
        
        if self.global_counter is not None:
            self.global_counter["token_completion"][self.name] += usage["completion"]
            self.global_counter["token_prompt"][self.name] += usage["prompt"]
            self.global_counter["api_calls"][self.name] += 1
        
        return result, usage

    

    def shrink_msg_by(self, msg, shrink_idx, L):
        if L <= 0:
            return msg
        counter = 0
        while L > 0:
            L_old = len(self.encode(self.compile_msg_txt(msg)))
            print("Shrinking message by {} tokens...".format(L))
            if len(self.encode(msg[shrink_idx]['content']))<L:
                shrink_idx = np.argmax([len(self.encode(x['content'])) for x in msg])
            msg[shrink_idx]['content'] = self.decode(self.encode(msg[shrink_idx]['content'])[L:])
            L_new = len(self.encode(self.compile_msg_txt(msg)))
            L = L - (L_old - L_new)
            if L_old - L_new <= 0:
                counter+=1
                if counter>=3:
                    print("Shrinking failed!")
                    import ipdb; ipdb.set_trace()
            else:
                counter = 0
        return msg
    
    def compile_msg_txt(self, msg):
        result = ""
        for x in msg:
            if x['role'].lower() == 'system':
                role = "context"
            elif x['role'].lower() == 'user':
                role = "question"
            else:
                role = "answer"
            result += "## {} ##\n".format(role)
            result += x['content'].strip() + "\n\n"
        result += "## answer ##\n"
        return result.strip()

    def compute_length(self, msg):
        L = 0
        for x in msg:
            L += len(self.encode(x['content']))
        return L

    def shrink_msg(self, msg, shrink_idx, L_max):
        L = self.compute_length(msg)
        return self.shrink_msg_by(msg, shrink_idx, L-L_max)

    def shrink_text(self, msg, shrink_idx, L_max):
        L = len(self.encode(self.compile_msg(msg)))
        return self.shrink_msg_by(msg, shrink_idx, L-L_max)
    
    def shrink_raw(self, text, L_max):
        L = len(self.encode(text))
        if L>L_max:
            return self.encode(text)[L-L_max+1:]
        else:
            return text