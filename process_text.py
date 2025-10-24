import torch
import torch.nn.functional as F
import numpy as np
from functools import lru_cache

class TextProcessor:
    def __init__(self, model_setup):
        self.models = model_setup.models
        self.tokenizers = model_setup.tokenizers
        self.model_types = model_setup.model_types
        self.device = model_setup.device
        self.tokens = {}
    
    # Cache tokens

    def save_token_cache(self, filepath='token_cache.pkl'):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.tokens, f)
        print(f"Token cache saved to {filepath}")

    def load_token_cache(self, filepath='token_cache.pkl'):
        import pickle
        try:
            with open(filepath, 'rb') as f:
                self.tokens = pickle.load(f)
        except FileNotFoundError:
            print(f"No cache file found at {filepath}")
    
    @lru_cache(maxsize=1000)
    def tokenize_with_cache(self, text, model_name):
        tokenizer = self.tokenizers[model_name]
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        return tokens, token_ids

    def clear_cache(self):
        self.tokenize_with_cache.cache_clear()
    
    # Find the target phrase (the verb)
    def find_target_phrase_token(self, model_name, sentence, target_phrase):
        tokenizer = self.tokenizers[model_name]

        sentence_tokens, _ = self.tokenize_with_cache(sentence, model_name) # the whole sentence
        target_tokens, _ = self.tokenize_with_cache(target_phrase, model_name) # target (e.g., 읽었다)

        if not target_tokens:
            return None

        for i in range(len(sentence_tokens) - len(target_tokens) + 1):
            if sentence_tokens[i:i+len(target_tokens)] == target_tokens:
                return(i+1, i+1+len(target_tokens), target_tokens) # +1 for [CLS]
        
        # Fuzzy match if no exact match
        # This is because tokenization is context-sensitive for causal LM
        return self.fuzzy_find_target_phrase(model_name, sentence, target_phrase, sentence_tokens, target_tokens)
    
    def fuzzy_find_target_phrase(self, model_name, sentence, target_phrase, sentence_tokens, target_tokens):
        joined_sentence = ''.join(sentence_tokens).replace('##', '').replace('_', '')
        joined_target = ''.join(target_tokens).replace('##', '').replace('_', '')

        if joined_target in joined_sentence:
            char_start = joined_sentence.find(joined_target)
            approx_token_start = len(self.tokenizers[model_name].tokenize(joined_sentence[:char_start])) + 1
            approx_token_end = approx_token_start + len(target_tokens)
            return (approx_token_start, approx_token_end, target_tokens)
        
        return None