from GPT2 import GPT2
import torch
import torch.nn.functional as F
import random
import re
from nltk import tokenize

class GenericGenerator:

    def __init__(self, module=None, gpu=True, temperature=1, top_p=0.9):
        self.gpu = gpu
        self.temperature = temperature
        self.top_p = top_p
        if module != None:
            self.gpt2 = module
        else:
            self.gpt2 = GPT2()
    
    def _next_word(self, prompt):
        # mostly from https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_generation.py
        device = 'cuda' if self.gpu else 'cpu'
        indexed_tokens = self.gpt2.tokenizer.encode(prompt)
        input_ids = torch.tensor([indexed_tokens], device=device)
        input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.gpt2.model(input_ids)
            next_token_logits = outputs[0][0, -1, :] / self.temperature
            # filter key_p
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('Inf')
            
        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).item()
        return self.gpt2.tokenizer.decode([next_token])
        
    def _clean_text(self, text):
        # I didn't write all of this, but I can't remember where it came from...
        # it fixes up common GPT-2 generation quirks
        text = text.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ',').replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't").replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
        #text = self.gpt2.tokenizer.clean_up_tokenization(text)
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub('\n +', '\n', text)
        text = re.sub(r'(\b\S+\b)\s+\b\1\b', r'\1', text)
        return text
    
    def generate(self, prompt, stop, constraints=None, max_tokens=50, max_tries=5, attempt=0):
        '''Generate and return text between `prompt` and `stop`. Good for names and other short strings.'''
        
        if stop == None: stop = []
        elif type(stop) == str: stop = [stop]
        
        generated = ''
        for _ in range(max_tokens):
            # generate the next word
            word = self._next_word(prompt + generated)
            if word == '<|endoftext|>': break
            # look for any of the stop strings in the newest word
            if True in [check in word for check in stop]: break
            generated += word
        
        generated = self._clean_text(generated)
        if constraints == None or constraints(generated):
            return generated
        elif attempt >= max_tries-1:
            return None
        else:
            attempt += 1
            # if the text did not pass the constraints, try it all over again
            return self.generate(prompt, stop, constraints, max_tokens, max_tries, attempt)
        
    def sentence(self, preamble, prompt='', constraints=None, max_tries=5, attempt=0):
        '''Generate or complete a single sentence'''

        generated = ''
        # 100 here is arbitrary, but it represents that max number of tokens a sentence could have
        for _ in range(100):
            # gnereate the next word
            word = self._next_word(preamble + prompt + generated)
            if word == '<|endoftext|>': break
            sentences = tokenize.sent_tokenize(prompt + generated + word)
            # we know the sentence is complete when the sentence tokenizer finds 2 sentences
            if len(sentences) > 1:
                generated = sentences[0]
                break
            if '\n' not in word:
                generated += word
        
        generated = self._clean_text(generated).strip()
        if constraints == None or constraints(generated):
            return generated
        elif attempt >= max_tries-1:
            return ''
        else:
            attempt += 1
            # if the text did not pass the constraints, try it all over again
            return self.sentence(preamble, prompt, constraints, max_tries, attempt)
