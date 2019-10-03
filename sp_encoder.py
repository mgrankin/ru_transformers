"""Byte pair encoding utilities"""
import os
import sentencepiece as spm
import hashlib
from pytorch_transformers.tokenization_utils import PreTrainedTokenizer
import shutil
import regex as re

class SPEncoder(PreTrainedTokenizer):
    def_name = 'encoder.model'
    def __init__(self, filename, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.max_len_single_sentence = 1024 # no default special tokens - you can update this value if you add special tokens
        self.max_len_sentences_pair = 1024 # no default special tokens - you can update this value if you add special tokens

        if os.path.isdir(filename): filename = os.path.join(filename, self.def_name)

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(filename)
        self.hash = hashlib.sha512(open(filename, 'rb').read()).hexdigest()[:10]
        self.filename = filename

    def encode(self, text):
        if text and text[0] != ' ': text = ' ' + text
        text = re.sub(r'(?=[^ ])([\W])([\w])',r'\g<1> \g<2>',text)
        text = text.replace('\n', '<|n|>')
        stext = re.split('(<\|n\|>)', text)
        result = [token 
                for item in stext 
                    for token in self.sp.EncodeAsIds(item)]
        return result

    def decode(self, tokens):
        if not isinstance(tokens,list):
            tokens = tokens.tolist()
        result = self.sp.DecodeIds(tokens).replace('<|n|>', '\n')
        result = re.sub(r'([«"''\n(]) (\w)',r'\g<1>\g<2>', result)
        result = re.sub(r'(\w)- (\w)',r'\g<1>-\g<2>', result)
        return result

    def tokenize(self, text, **kwargs):
        return self.sp.EncodeAsPieces(text)

    def convert_tokens_to_ids(self, tokens):
        return [self.sp.PieceToId(token) for token in tokens]

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        return cls(*inputs, **kwargs)

    def add_special_tokens_single_sentence(self, token_ids):
        return token_ids

    def save_pretrained(self, save_directory):
        shutil.copyfile(self.filename, os.path.join(save_directory, self.def_name))
