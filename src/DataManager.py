import re
from collections import defaultdict
import pandas as pd
import numpy as np



class SimpleTokenizer:
    def __init__(self):
        self.token_to_id = defaultdict(lambda: len(self.token_to_id))
        self.token_to_id['<PAD>'] = 0  # Padding token

    def tokenize(self, text):
        # Simple tokenization by splitting on non-word characters
        tokens = re.findall(r'\w+|\S', text)
        return tokens
    
    def merge_strings(self, input_values):
        target_array = []
        for i in input_values:
            result = " ".join(i)
            target_array.append(i)

        return target_array

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id[token] for token in tokens]

    def tokenize_column(self, series):
        # Apply tokenization to each row in the pandas series
        tokenized = series.apply(self.tokenize)
        return tokenized
    
    def run_convert(self):
        pass


class BetterTokenizer:
    def __init__(self, starting_tokenizer=None):
        
        self.token_to_id = defaultdict(lambda: len(self.token_to_id))
        self.id_to_token = {}
        self.token_to_id['<PAD>'] = 0  # Padding token

        self.check_tokenizer(starting_tokenizer=starting_tokenizer)

    def check_tokenizer(self,starting_tokenizer):
        if starting_tokenizer is None:
            return
        else:
            self.assign_tokenizer(input_dict=starting_tokenizer)

    def assign_tokenizer(self, input_dict):
        self.token_to_id = input_dict['token_to_id']
        self.id_to_token = input_dict['id_to_token']
        self.id_to_token = {int(k): v for k, v in input_dict['id_to_token'].items()}


    def tokenize(self, text):
        # Simple tokenization by splitting on non-word characters
        tokens = re.findall(r'\w+|\S', text)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            token_id = self.token_to_id[token]
            self.id_to_token[token_id] = token  # Store the reverse mapping
            ids.append(token_id)
        return ids

    def decode(self, token_ids):
        # Convert a list of token IDs back to a string
        tokens = [self.id_to_token.get(token_id, '') for token_id in token_ids]
        return ''.join(tokens)  # Modify this line as needed for your URL format

    def tokenize_column(self, series):
        # Apply tokenization to each row in the pandas series
        tokenized = series.apply(self.tokenize)
        # Convert each list of tokens to a list of token IDs
        tokenized_ids = tokenized.apply(self.convert_tokens_to_ids)
        return tokenized_ids
    
    def decode_padded(self, sequence):
        padding_idx = list(sequence).index(0) if 0 in sequence else len(sequence)

        tokens = [self.id_to_token[token_id] for token_id in sequence[:padding_idx]]
        decoded_string = ''.join(tokens)
        return decoded_string

    def run_convert(self):
        pass


class Tokenizer:
    token_to_id = defaultdict(lambda: len(Tokenizer.token_to_id))
    id_to_token = {}
    token_to_id['<PAD>'] = 0  # Padding token

    @classmethod
    def tokenize(cls, text):
        tokens = re.findall(r'\w+|\S', text)
        return tokens

    @classmethod
    def convert_tokens_to_ids(cls, tokens):
        ids = []
        for token in tokens:
            token_id = cls.token_to_id[token]
            cls.id_to_token[token_id] = token
            ids.append(token_id)
        return ids

    @classmethod
    def decode(cls, token_ids):
        tokens = [cls.id_to_token.get(token_id, '') for token_id in token_ids]
        return ' '.join(tokens)

    @classmethod
    def tokenize_column(cls, series):
        tokenized = series.apply(cls.tokenize)
        tokenized_ids = tokenized.apply(cls.convert_tokens_to_ids)
        return tokenized_ids





    




