import torch
import torch.nn.functional as F
import numpy as np
from functools import lru_cache

class TextProcessor:
    def __init__(self, model_setup):
        self.model_setup = model_setup
        self.tokens = {}

    @property
    def models(self):
        return self.model_setup.models
    
    @property
    def tokenizers(self):
        return self.model_setup.tokenizers
    
    @property
    def model_types(self):
        return self.model_setup.model_types
    
    @property
    def device(self):
        return self.model_setup.device
    
    def _get_tokenization_group(self, model_name):
        """
        Determine which tokenization group a model belongs to.
        Uses the tokenizer_group from model_configs.
        
        Groups:
        - Group 1: BERT, GEMMA, KoGPT3 (standard tokenization)
        - Group 2: SOLAR (byte-level tokenization)
        - Group 3: Llama, Polyglot (uninterpretable raw tokens)
        
        Returns: 'group_1', 'group_2', or 'group_3'
        """
        # Get tokenizer group from model_setup's config
        if hasattr(self.model_setup, 'model_configs') and model_name in self.model_setup.model_configs:
            group_num = self.model_setup.model_configs[model_name].get('tokenizer_group', 1)
            return f'group_{group_num}'
        
        # Fallback for models not in config
        print(f"WARNING: Model '{model_name}' not found in model_configs, using fallback detection")
        model_name_lower = model_name.lower()
        
        if 'solar' in model_name_lower:
            return 'group_2'
        elif 'llama' in model_name_lower or 'polyglot' in model_name_lower:
            return 'group_3'
        else:
            return 'group_1'  # Default to standard tokenization
    
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
        """
        Tokenize the sentences (=text). When the same text is tokenized, retrieve the pre-computed tokens from cache. 
        """
        tokenizer = self.tokenizers[model_name]
        tokenization_group = self._get_tokenization_group(model_name)

        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        
        print(f"Tokenizing {text}")
        print(f"Tokens: {tokens}")

        if tokenization_group == 'group_3':
            # Try to get readable representation
            readable_tokens = []
            for token in tokens:
                # Try to decode individual tokens for better readability
                try:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    decoded = tokenizer.decode([token_id], skip_special_tokens=True)
                    readable_tokens.append(decoded if decoded else token)
                except:
                    readable_tokens.append(token)
            
            print(f"\tRaw tokens: {tokens}")
            print(f"\tReadable tokens: {readable_tokens}")
        
        return tokens, token_ids

    def clear_cache(self):
        self.tokenize_with_cache.cache_clear()
    
    # Find the target phrase (the verb)
    def find_target_phrase_token(self, model_name, sentence, target_phrase):
        """
        Find target phrase in sentence using character-level matching.
        
        1. Find the character position of target phrase in sentence.
        2. Tokenize the full sentence.
        3. Map token positions back to character positions.
        4. Find which tokens overlap with the target phrase character span.
        5. VALIDATE that we captured the complete target phrase.
        """
        tokenizer = self.tokenizers[model_name]
        model_type = self.model_types[model_name]
        tokenization_group = self._get_tokenization_group(model_name)

        # Find character position of target phrase
        char_start = sentence.find(target_phrase)
        if char_start == -1:
            print(f"ERROR: Target phrase '{target_phrase}' not found in sentence '{sentence}'")
            return None, None, None
        char_end = char_start + len(target_phrase)

        # Tokenize the full sentence
        sentence_tokens, token_ids = self.tokenize_with_cache(sentence, model_name)
        
        # Get character offsets for each token
        encoding = tokenizer(sentence, add_special_tokens=True, return_offsets_mapping=True)

        if 'offset_mapping' not in encoding:
            print("WARNING: offset_mapping not available, using fallback method")
            return self._find_target_phrase_fallback(model_name, sentence, target_phrase, sentence_tokens, token_ids)

        offset_mapping = encoding['offset_mapping']

        start_token_idx = None
        end_token_idx = None

        # Find tokens that overlap with target phrase
        # Offset [idx, idx) -> end exclusive
        for i, (token_char_start, token_char_end) in enumerate(offset_mapping):
            # Skip special tokens (they have offset (0, 0))
            if token_char_start == token_char_end and i > 0:
                continue
            
            # A token is part of the target phrase if its character range overlaps
            # Overlap occurs when: token_start < target_end AND token_end > target_start
            # We use <= for char_start to include tokens that START at target position
            has_overlap = (token_char_start < char_end and token_char_end > char_start)
            
            # Also explicitly check if this token contains the start of the target
            contains_start = (token_char_start <= char_start < token_char_end)
            
            if has_overlap or contains_start:
                if start_token_idx is None:
                    start_token_idx = i
                end_token_idx = i + 1

        if start_token_idx is None or end_token_idx is None:
            print(f"ERROR: Could not find token indices for target phrase")
            print(f"\tCharacter span: [{char_start}, {char_end})")
            print(f"\tTarget phrase: '{target_phrase}'")
            print(f"\tSentence: '{sentence}'")
            print(f"\tOffset mapping: {offset_mapping}")
            return None, None, None

        # VALIDATION: Check if we captured the complete target phrase
        # Reconstruct text from the identified token span
        captured_char_start = offset_mapping[start_token_idx][0]
        captured_char_end = offset_mapping[end_token_idx - 1][1]
        captured_text = sentence[captured_char_start:captured_char_end]
        
        print(f"\tInitial capture: tokens[{start_token_idx}:{end_token_idx}]")
        print(f"\tCharacter span: [{captured_char_start}, {captured_char_end})")
        print(f"\tCaptured text: '{captured_text}'")
        print(f"\tTarget phrase: '{target_phrase}'")
        
        # If we didn't capture the full target phrase, expand the span
        if target_phrase not in captured_text:
            print(f"WARNING: Initial capture '{captured_text}' doesn't contain full target '{target_phrase}'")
            print(f"\tAttempting to expand token span...")
            
            # Expand backwards if needed
            while start_token_idx > 0 and target_phrase not in sentence[offset_mapping[start_token_idx][0]:captured_char_end]:
                start_token_idx -= 1
                captured_char_start = offset_mapping[start_token_idx][0]
                captured_text = sentence[captured_char_start:captured_char_end]
                if target_phrase in captured_text:
                    break
            
            # Expand forwards if needed
            while end_token_idx < len(offset_mapping) and target_phrase not in sentence[captured_char_start:offset_mapping[end_token_idx - 1][1]]:
                if end_token_idx < len(offset_mapping):
                    end_token_idx += 1
                    captured_char_end = offset_mapping[end_token_idx - 1][1]
                    captured_text = sentence[captured_char_start:captured_char_end]
                    if target_phrase in captured_text:
                        break
                else:
                    break
            
            # Final validation
            captured_text = sentence[offset_mapping[start_token_idx][0]:offset_mapping[end_token_idx - 1][1]]
            if target_phrase not in captured_text:
                print(f"ERROR: Could not capture full target phrase even after expansion")
                print(f"  Target: '{target_phrase}'")
                print(f"  Captured: '{captured_text}'")
                return None, None, None
            else:
                print(f"  SUCCESS: Expanded to capture full target phrase")

        target_tokens = sentence_tokens[start_token_idx:end_token_idx]
        print(f"\tTarget tokens: {target_tokens}")
        
        return start_token_idx, end_token_idx, target_tokens
        
    def _find_target_phrase_fallback(self, model_name, sentence, target_phrase, sentence_tokens, token_ids):
        """
        Fallback method for tokenizers that don't support offset_mapping.
        Uses a sliding window approach to match token sequences.
        
        Improved to ensure we capture the COMPLETE target phrase.
        """
        print("Using fallback method for target phrase detection")
        tokenization_group = self._get_tokenization_group(model_name)
        
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize target phrase separately to get an idea of what to look for
        target_tokens_isolated, _ = self.tokenize_with_cache(target_phrase, model_name)
        
        # Remove special tokens from isolated tokenization
        special_tokens = [tokenizer.bos_token, tokenizer.eos_token, 
                         tokenizer.pad_token, tokenizer.cls_token, 
                         tokenizer.sep_token]
        target_tokens_isolated = [t for t in target_tokens_isolated if t not in special_tokens]
        
        print(f"Target phrase isolated tokens: {target_tokens_isolated}")
        
        # Try to find a substring match in the full sentence tokens
        best_match = None
        best_score = 0
        
        # Search with varying window sizes around the expected length
        min_len = len(target_tokens_isolated)
        max_len = min(len(target_tokens_isolated) + 5, len(sentence_tokens))
        
        for start_idx in range(len(sentence_tokens)):
            for window_len in range(min_len, max_len + 1):
                end_idx = min(start_idx + window_len, len(sentence_tokens))
                if end_idx > len(token_ids):
                    continue
                    
                # Decode this span
                span_tokens = token_ids[start_idx:end_idx]
                try:
                    decoded = tokenizer.decode(span_tokens, skip_special_tokens=True)
                    # Clean up whitespace for comparison
                    decoded_clean = decoded.strip().replace(' ', '').replace('▁', '')
                    target_clean = target_phrase.strip().replace(' ', '').replace('▁', '')
                    
                    # Check if this span contains the COMPLETE target
                    if target_clean == decoded_clean:
                        # Perfect match - highest priority
                        score = 1.0
                        best_score = score
                        best_match = (start_idx, end_idx)
                        break
                    elif target_clean in decoded_clean and len(decoded_clean) <= len(target_clean) * 1.5:
                        # Contains target and not too much extra
                        score = len(target_clean) / len(decoded_clean)
                        if score > best_score:
                            best_score = score
                            best_match = (start_idx, end_idx)
                except:
                    continue
            
            if best_score == 1.0:  # Found perfect match
                break
        
        if best_match is None:
            print(f"ERROR: Could not find target phrase using fallback method")
            return None, None, None
        
        start_token_idx, end_token_idx = best_match
        target_tokens = sentence_tokens[start_token_idx:end_token_idx]
        
        # Validate the match
        span_tokens = token_ids[start_token_idx:end_token_idx]
        decoded = tokenizer.decode(span_tokens, skip_special_tokens=True)
        decoded_clean = decoded.strip().replace(' ', '').replace('▁', '')
        target_clean = target_phrase.strip().replace(' ', '').replace('▁', '')
        
        print(f"\nTokenizing {sentence}")
        print(f"\tTokens: {sentence_tokens}")
        print(f"\tTokenizing {target_phrase}")
        print(f"\tTokens: {target_tokens}")
        print(f"\tToken indices: [{start_token_idx}, {end_token_idx}) [FALLBACK METHOD]")
        print(f"\tDecoded: '{decoded}' (cleaned: '{decoded_clean}')")
        print(f"\tTarget: '{target_phrase}' (cleaned: '{target_clean}')")
        print(f"\tMatch quality: {best_score:.2%}")
        print(f"\tTokenization group: {tokenization_group}")
        
        if target_clean not in decoded_clean:
            print(f"WARNING: Captured text doesn't fully contain target phrase!")
        
        return start_token_idx, end_token_idx, target_tokens
