import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

class ModelEvaluator:
    def __init__(self, device: str="auto"):
        self.models = {}
        self.tokenizers = {}
        self.model_types = {} # masked (bert) vs. causal
        self.device = self._setup_device(device)
        self.load_models()

    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def load_models(self):
        print(f"Loading models on device: {self.device}")

        model_configs = {
            # Bert-based models (masked LM)
            "bert-kor-base": {
                "path": "kykim/bert-kor-base", 
                "type": "bert", 
                "model_class": AutoModelForMaskedLM
            },
            "koelectra-base":{
                "path": "monologg/koelectra-base-generator",
                "type": "bert",
                "model_class": AutoModelForMaskedLM
            },

            # Causual LM models
            "gemma": {
                "path": "google/gemma-3-270m",
                "type": "causal", 
                "model_class": AutoModelForCausalLM
            }
        }

        for name, config in model_configs.items(): 
            try:
                print(f"Loading {name}...")

                self.tokenizers[name] = AutoTokenizer.from_pretrained(config['path'])

                # Add padding if missing
                if config['type'] == 'causal':
                # Add padding token for causal models
                    if self.tokenizers[name].pad_token is None:
                        self.tokenizers[name].pad_token = self.tokenizers[name].eos_token
            
                model_kwargs = {}
                if config['type'] == 'causal':
                    if self.device.type == 'cuda':
                        model_kwargs['torch_dtype'] = torch.float16
                        model_kwargs['device_map'] = 'auto'
                    else:
                        model_kwargs['torch_dtype'] = torch.float32 # use float 32 on cpu to avoid nan
                     
                self.models[name] = config['model_class'].from_pretrained(
                    config['path'], **model_kwargs
                )

                # Resize embeddings if we added tokens
                if config['type'] == 'bert' and self.tokenizers[name].mask_token_id is not None:
                    self.models[name].resize_token_embeddings(len(self.tokenizers[name]))

                self.models[name].eval()
                self.model_types[name] = config['type']
                if self.device.type == 'cpu' or 'device_map' not in model_kwargs:
                    self.models[name] = self.models[name].to(self.device)

                # # Verify mask token for BERT models
                # if config['type'] == 'bert':
                #     print(f"  Mask token: {self.tokenizers[name].mask_token} (id={self.tokenizers[name].mask_token_id})")
                
                print(f"Loaded {name}.")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                import traceback
                traceback.print_exc()

    @lru_cache(maxsize=1000)
    def tokenize_with_cache(self, text: str, model_name: str) -> Tuple[List[str], List[int]]:
        """Caches the tokenization results for faster processing"""
        tokenizer = self.tokenizers[model_name]
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        return tokens, token_ids
    
    def find_target_phrase_token(self, model_name: str, sentence: str, target_phrase: str) -> Optional[Tuple[int, int, List[str]]]:
        tokenizer = self.tokenizers[model_name]

        # Use cached tokenization
        sentence_tokens, _ = self.tokenize_with_cache(sentence, model_name) # the whole sentence
        target_tokens, _ = self.tokenize_with_cache(target_phrase, model_name) # target phrase (the verb phrase)

        if not target_tokens:
            return None
        
        # Find the match b/t sentence & target phrase
        for i in range(len(sentence_tokens) - len(target_tokens) + 1):
            if sentence_tokens[i:i+len(target_tokens)] == target_tokens:
                return (i+1, i+1+len(target_tokens), target_tokens) # +1 for [CLS]
        
        # Fuzzy matching if no exact match
        # This is becauze tokenization is context-sensitive
        return self.fuzzy_find_target_phrase(model_name, sentence, target_phrase, 
                                            sentence_tokens, target_tokens)
    
    def fuzzy_find_target_phrase(self, model_name: str, sentence: str, target_phrase: str,
                                 sentence_tokens: List[str], target_tokens: List[str]) -> Optional[Tuple[int, int, List[str]]]:
        joined_sentence = ''.join(sentence_tokens).replace('##', '').replace('_', '')
        joined_target = ''.join(target_tokens).replace('##', '').replace('_', '')

        if joined_target in joined_sentence:
            char_start = joined_sentence.find(joined_target)
            approx_token_start = len(self.tokenizers[model_name].tokenize(joined_sentence[:char_start])) + 1
            approx_token_end = approx_token_start + len(target_tokens)
            return (approx_token_start, approx_token_end, target_tokens)
        
        return None

    def char_to_token_positions(self, model_name: str, sentence: str, char_pos: int) -> int:
        """
        Convert a character position in a sentence to its corresponding token index.

        Maps a zero-indexed character position in the original sentence to the index of the token that contains that character, after tokenization with the specified model's tokenizer.

        Args:
        model_name: Name of the tokenizer model to use for tokenization
        sentence: The input sentence/text to tokenize
        char_pos: Zero-indexed character position in the original sentence
    
        Returns:
            The index of the token containing the character at char_pos. Special tokens
            (e.g., [CLS], [SEP], <bos>) are included in the indexing. Returns the last
            token index if char_pos is out of bounds.
      
        Example:
            >>> char_to_token_positions("bert-base", "playing games", 5)
            1  # Returns index of "##ing" token (if "playing" tokenizes as ["play", "##ing"])
        """
        # Return the last token idx if the character position is beyond the sentence length
        if char_pos >= len(sentence): 
            _, token_ids = self.tokenize_with_cache(sentence, model_name)
            return len(token_ids) - 1
        
        # Tokenize the sentence
        tokenizer = self.tokenizers[model_name]
        _, token_ids = self.tokenize_with_cache(sentence, model_name)
        tokens_with_special = tokenizer.convert_ids_to_tokens(token_ids)

        # Skip special tokens, clean them, maintain character ranges, and find the match
        current_pos = 0
        for i, token in enumerate(tokens_with_special):
            if token in ['[CLS]', '<bos>', '[SEP]', '</s>', '<eos>']:
                continue

            clean_token = token.replace('##', '').replace('_', ' ')
            token_end_pos = current_pos + len(clean_token)

            # check whether the current_pos falls in the current token
            if current_pos <= char_pos < token_end_pos: 
                return i
            
            current_pos = token_end_pos

        return len(tokens_with_special) - 1 # returns the last token index if no match is found

    @torch.no_grad()
    def get_sentence_probability_bert(self, model_name: str, sentence: str) -> float:
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        _, token_ids = self.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)

        if tokens.shape[1] <= 2: # Only special tokens
            return 0.0

        total_log_prob = 0.0
        content_tokens = 0

        # Process all positions
        for i in range(1, tokens.shape[1] - 1): # Skip [CLS] and [SEP]
            masked_tokens = tokens.clone()
            original_token = masked_tokens[0, i].item()
            masked_tokens[0, i] = tokenizer.mask_token_id

            outputs = model(masked_tokens)
            log_probs = F.log_softmax(outputs.logits[0, i], dim=-1)
            total_log_prob += log_probs[original_token].item()
            content_tokens += 1
        
        return total_log_prob / content_tokens if content_tokens > 0 else 0.0

    @torch.no_grad()
    def get_sentence_probability_causal(self, model_name: str, sentence: str) -> float:
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        try:
            print(f"DEBUG: Tokenizing '{sentence}' with {model_name}")
            _, token_ids = self.tokenize_with_cache(sentence, model_name)
            print(f"DEBUG: Token IDs: {token_ids}")
            tokens = torch.tensor([token_ids], device=self.device)
            print(f"DEBUG: Token shape: {tokens.shape}, device: {tokens.device}")

            outputs = model(tokens, labels=tokens)
            print(f"DEBUG: Loss: {outputs.loss.item()}")

            return -outputs.loss.item()
        except Exception as e:
            print(f"Error in causal probability for {model_name}: {e}")
            return float('nan')

    def get_target_phrase_probability(self, model_name: str, sentence: str, target_phrase: str) -> float:
        target_info = self.find_target_phrase_token(model_name, sentence, target_phrase)
        if not target_info:
            return 0.0

        start_idx, end_idx, target_tokens = target_info

        if self.model_types[model_name] == "bert":
            return self.get_target_phrase_prob_bert(model_name, sentence, start_idx, end_idx)
        else:
            return self.get_target_phrase_prob_causal(model_name, sentence, start_idx, end_idx)

    @torch.no_grad()
    def get_target_phrase_prob_bert(self, model_name: str, sentence: str, start_idx: int, end_idx: int) -> float:
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        _, token_ids = self.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)

        total_log_prob = 0.0
        num_tokens = 0

        for i in range(start_idx, min(end_idx, tokens.shape[1])):
            masked_tokens = tokens.clone()
            original_token = masked_tokens[0, i].item()
            masked_tokens[0, i] = tokenizer.mask_token_id

            outputs = model(masked_tokens)
            log_probs = F.log_softmax(outputs.logits[0, i], dim=-1)
            total_log_prob += log_probs[original_token].item()
            num_tokens += 1

        return total_log_prob / num_tokens if num_tokens > 0 else 0.0

    @torch.no_grad()
    def get_target_phrase_prob_causal(self, model_name: str, sentence: str,
                                    start_idx: int, end_idx: int) -> float:
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        try:
            print(f"DEBUG: Target phrase prob - start:{start_idx}, end:{end_idx}")
            _, token_ids = self.tokenize_with_cache(sentence, model_name)
            tokens = torch.tensor([token_ids], device=self.device)

            outputs = model(tokens)
            logits = outputs.logits[0]
            print(f"DEBUG: Logits shape: {logits.shape}")

            total_log_prob = 0.0
            num_tokens = 0

            for i in range(start_idx, min(end_idx, tokens.shape[1])):
                if i > 0 and i - 1 < logits.shape[0]:
                    target_token_id = tokens[0, i].item()
                    log_probs = F.log_softmax(logits[i - 1], dim=-1)
                    total_log_prob += log_probs[target_token_id].item()
                    num_tokens += 1
            print(f"DEBUG: Total log prob: {total_log_prob}, num_tokens: {num_tokens}")
            return total_log_prob / num_tokens if num_tokens > 0 else 0.0
        except Exception as e:
            print(f"Error in causal target phrase prob for {model_name}: {e}")
            return float('nan')        
 
    def get_surprisal_after_mask(self, model_name: str, sentence: str,
                                 mask_char_pos: int) -> float:
        mask_token_pos = self.char_to_token_positions(model_name, sentence, mask_char_pos)

        if self.model_types[model_name] == "bert":
            return self.get_surprisal_bert_at_token(model_name, sentence, mask_token_pos)
        else:
            return self.get_surprisal_causal_at_token(model_name, sentence, mask_token_pos)

    @torch.no_grad()
    def get_surprisal_bert_at_token(self, model_name: str, sentence: str, token_position: int) -> float:
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        _, token_ids = self.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)

        if token_position >= tokens.shape[1] or token_position < 0:
            return float('inf')
        
        masked_tokens = tokens.clone()
        original_token = masked_tokens[0, token_position].item()
        masked_tokens[0, token_position] = tokenizer.mask_token_id
        
        outputs = model(masked_tokens)
        log_probs = F.log_softmax(outputs.logits[0, token_position], dim=-1)
        log_prob = log_probs[original_token].item()

        return -log_prob / np.log(2) #convert to bits

    @torch.no_grad()
    def get_surprisal_causal_at_token(self, model_name: str, sentence: str, token_position: int) -> float:
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        try:
            _, token_ids = self. tokenize_with_cache(sentence, model_name)
            tokens = torch.tensor([token_ids], device=self.device)

            if token_position >= tokens.shape[1] or token_position <= 0:
                return float('inf')
        
            outputs = model(tokens)
            logits = outputs.logits[0]

            if token_position - 1 < logits.shape[0]:
                target_token_id = tokens[0, token_position].item()
                log_probs = F.log_softmax(logits[token_position - 1], dim=-1)
                log_prob = log_probs[target_token_id].item()
                return -log_prob / np.log(2)
            return float('inf')
        except Exception as e:
            print(f"Error in causal surprisal for {model_name}: {e}")
            return float('nan')

    def evaluate_batch(self, test_data: List[Dict], batch_size: int = 8) -> pd.DataFrame:
        results = []

        # Group by model to minimize model switching overhead
        for model_name in self.models.keys():
            print(f"\nEvaluating with {model_name}...")

            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size]

                for item in batch:
                    sentence = item["sentence"]
                    target_phrase = item["target_phrase"]
                    condition = item["condition"]
                    mask_char_pos = item.get("mask_char_pos")

                    try:
                        # Calculate
                        if self.model_types[model_name] == "bert":
                            avg_prob = self.get_sentence_probability_bert(model_name, sentence)
                        else:
                            avg_prob = self.get_sentence_probability_causal(model_name, sentence)
                        
                        target_prob = self.get_target_phrase_probability(model_name, sentence, target_phrase)

                        surprisal = None
                        if mask_char_pos is not None:
                            surprisal = self.get_surprisal_after_mask(model_name, sentence, mask_char_pos)

                        results.append({
                            "condition": condition,
                            "model": model_name, 
                            "sentence": sentence,
                            "target_phrase": target_phrase,
                            "avg_sentence_prob": avg_prob, 
                            "target_phrase_prob": target_prob,
                            "surprisal_after_mask": surprisal
                        })

                    except Exception as e:
                        print(f"Error evaluating {sentence} with {model_name}: {e}")
                        results.append({
                            "condition": condition,
                            "model": model_name,
                            "sentence": sentence,
                            "target_phrase": target_phrase, 
                            "avg_sentence_prob": None,
                            "target_phrase_prob": None,
                            "surprisal_after_mask": None
                        })
        return pd.DataFrame(results)


    def clear_cache(self):
        """Clear tokenization cache"""
        self.tokenize_with_cache.cache_clear()

def create_sample_data() -> List[Dict]:
    return [
        {
            "condition": "께서_시_best",
            "sentence": "선생님께서 책을 읽으셨어요.",
            "target_phrase": "읽으셨어요",
            "mask_char_pos": 11
        },
        {
            "condition": "이가_시",
            "sentence": "선생님이 책을 읽으셨어요.",
            "target_phrase": "읽으셨어요",
            "mask_char_pos": 10
        },
        {
            "condition": "께서_no시_worst",
            "sentence": "선생님께서 책을 읽었어요.",
            "target_phrase": "읽었어요",
            "mask_char_pos": 10
        },
        {
            "condition": "이가_no시",
            "sentence": "선생님이 책을 읽었어요.",
            "target_phrase": "읽었어요",
            "mask_char_pos": 9
        }
    ]

def sanity_check_proba(evaluator):
    test_cases = {
        "very_common": "내일 할게.",
        "grammatical": "선생님께서 책을 읽으셨다.",
        "ungrammatical": "선생님께서 책을 읽었다.",
        "random_tokens": "가나다라마바사아자차카타파하."
    }

    print("\n---SANITY CHECKS---")
    for model_name in evaluator.models.keys():
        print(f"\nModel: {model_name}")
        probs = {}
        for label, sentence in test_cases.items():
            if evaluator.model_types[model_name] == "bert":
                prob = evaluator.get_sentence_probability_bert(model_name, sentence)
            else:
                prob = evaluator.get_sentence_probability_causal(model_name, sentence)
            probs[label] = prob
            print(f" {label:20s}: {prob:.4f}")
        # Check expectations
        print(f"\n  Checks:")
        print(f"    ✓ Very common > Random? {probs['very_common'] > probs['random_tokens']}")
        print(f"    ✓ Grammatical > Ungrammatical? {probs['grammatical'] > probs['ungrammatical']}")
        print(f"    ✓ No NaN values? {not any(np.isnan(p) for p in probs.values())}")
        print(f"    ✓ No Inf values? {not any(np.isinf(p) for p in probs.values())}")

def load_csv(filename):
    df = pd.read_csv(filename, encoding='utf-8')
    if 'grammatical' in df.columns:
        df['grammatical'] = df['grammatical'].astype(str).str.lower() == 'true'
    test_data = df.to_dict('records')
    return test_data

def main():
    print("Initializing the model evaluator...")
    evaluator = ModelEvaluator()

    # Check if gemma loaded successfully
    print("\nChecking loaded models:")
    for name, model in evaluator.models.items():
        print(f"  {name}: {type(model)} - Type: {evaluator.model_types[name]}")
    
    if 'gemma' not in evaluator.models:
        print("WARNING: Gemma did not load.")

    sanity_check_proba(evaluator)

    # # Create sample data (for debugging)
    # test_data = create_sample_data()

    test_data = load_csv('data/sentences.csv')
    
    # Run batch evaluation
    print("Running evaluation...")
    results_df = evaluator.evaluate_batch(test_data, batch_size=4)

    # Save results
    output_file = "data/results_evaluation.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Results saved to {output_file}")

    # Display summary
    print("\nSumamry by condition and model:")
    summary = results_df.groupby(['model', 'condition']).agg({
        'avg_sentence_prob': 'mean',
        'target_phrase_prob': 'mean', 
        'surprisal_after_mask': 'mean'
    }).round(4)
    print(summary)

    evaluator.clear_cache()

    return results_df

if __name__ == "__main__":
    results = main()