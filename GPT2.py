from pytorch_transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

class GPT2:
    '''This just loads up the pretrained GPT-2 models'''

    def __init__(self, gpu=True):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.model.eval()
        if gpu: self.model.to('cuda')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')