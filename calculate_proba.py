import torch
import torch.nn.functional as F
import numpy as np

class ProbaCalculator:
    def __init__(self, model_setup, text_processor):
        self.model_setup = model_setup
        self.text_processor = text_processor
        self.device = model_setup.device
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

    # Sentence probability
    @torch.no_grad()
    def get_sentence_probability_bert(self, model_name, sentence):
        """
        1. Mask each token in the sentence
        2. Use BERT to predict what should be in that masked position
        3. Calculate the log probability of the actual token being in that position
            - Gets probability distribution over vocab at position i
            - Extracts log probability of the original token from that distribution
        4. Sum all log probabilities and divide by the number of tokens
        """
        print(f"[{model_name}]: Calculating the sentence probability: {sentence}")
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        token_probs = []
        token_strings = []
        cache_key = f"{model_name}::{sentence}"

        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)

        print(f"\tToken shape:{tokens.shape}")

        if tokens.shape[1] <= 2: # contains only special tokens
            print("\t\tWarning: Sentence contains only special tokens")
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
            token_log_prob = log_probs[original_token].item()
            total_log_prob += token_log_prob
            content_tokens += 1

            token_probs.append(token_log_prob)
            token_strings.append(tokenizer.convert_ids_to_tokens([original_token])[0])

        print(f"\tTokens used for calculating: {token_strings}")
        print(f"\tProbabilities: {token_probs}")
        self.tokens[cache_key] = {'sentence': sentence,
                                  'model': model_name,
                                  'tokens': token_strings,
                                  'token_probs': token_probs}
        
        proba_out = total_log_prob / content_tokens if content_tokens > 0 else 0.0
        print(f"\tAveraged Probability: {proba_out}")
        return proba_out
    
    @torch.no_grad()
    def get_sentence_probability_bert_without_mask(self, model_name, sentence):
        """
        1. Run BERT on the sentence without any masking -> returns probability distribution over entire vocabulary for each position
        2. Convert to probabilities with softmax
        3. For each token position, get probability of the actual token and then average them
        """
        cache_key = f"{model_name}::{sentence}"
        print(f"[{model_name}]: Calculating the sentence probability without mask: {sentence}")

        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
    
        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)
        print(f"\tToken shape:{tokens.shape}")
        
        outputs = model(tokens)
        logits = outputs.logits
        print(f"\tOutput logits shape: {logits.shape}")

        log_probs = F.log_softmax(logits[0], dim=-1)
        
        token_probs = []
        token_strings = []

        total_log_prob = 0.0
        num_tokens = 0
        
        for i in range(1, tokens.shape[1]-1): # skip CLS & SEP
            actual_token_id = tokens[0, i].item()
            token_prob = log_probs[i, actual_token_id].item()
            
            total_log_prob += token_prob 
            num_tokens += 1

            token_probs.append(token_prob)
            token_strings.append(tokenizer.convert_ids_to_tokens([actual_token_id])[0])
        
        proba_out = total_log_prob / num_tokens if num_tokens > 0 else 0.0

        print(f"\tTokens used for calculating: {token_strings}")
        print(f"\tProbabilities: {token_probs}")
        print(f"\tAveraged Probability: {proba_out}")
        self.tokens[cache_key] = {'sentence': sentence,
                                  'model': model_name,
                                  'tokens': token_strings,
                                  'token_probs': token_probs}
        return proba_out
    
    @torch.no_grad()
    def get_sentence_probability_causal(self, model_name, sentence):
        """
        1. Feed the entire sentence to the causal LM
        2. The model calculates cross-entropy loss (- log P (token_i | token 1... i-1)) for each token
        3. The model returns negative log probability (loss)
        4. Negate the cross entropy loss to get average log probability
        """
        cache_key = f"{model_name}::{sentence}"
        print(f"[{model_name}]: Calculating the sentence probability: {sentence}")
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)

        # Auto-compute loss
        outputs = model(tokens, labels=tokens)
        proba_out_auto = -outputs.loss.item() # avg log prob per token

        print(f"\tToken shape (auto-computed): {tokens.shape}")
        print(f"\tAveraged Probability (auto-computed): {proba_out_auto}")

        # Manually computed loss (for debugging purposes)
        outputs = model(tokens) # returns only logits without calculating loss
        logits = outputs.logits[0]

        total_log_prob = 0.0
        content_tokens = 0
        token_probs = []
        token_strings = []
        for i in range(tokens.shape[1] - 1):
            next_token_id = tokens[0, i+1].item()
            log_probs = F.log_softmax(logits[i], dim=-1)
            token_log_prob = log_probs[next_token_id].item() 

            total_log_prob += token_log_prob
            content_tokens += 1

            token_probs.append(token_log_prob)
            token_strings.append(tokenizer.convert_ids_to_tokens([next_token_id])[0])

        print(f"\tTokens used for calculating (manually-computed):{token_strings}")
        print(f"\tProbabilities (manually-computed): {token_probs}")

        proba_out_manual = total_log_prob / content_tokens if content_tokens > 0 else 0.0
        print(f"\tAveraged Probability (manually-computed): {proba_out_manual}")

        self.tokens[cache_key] = {'sentence': sentence,
                                  'model': model_name,
                                  'tokens': token_strings,
                                  'token_probs': token_probs}

        return proba_out_auto

    # Target phrase probability
    @torch.no_grad()
    def get_target_phrase_prob_bert(self, model_name, sentence, start_idx, end_idx):
        """
        1. Mask the target token(s)
        2. BERT calculates the probability of the target token(s) and takes an average
           e.g., P(읽|context) + P(으|context) + P(셨다|context) / 3
        """
        print(f"[{model_name}]: Calculating the target phrase probability: {sentence}")

        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)
        target_token_ids = token_ids[start_idx:end_idx]
        target_token_strings = tokenizer.convert_ids_to_tokens(target_token_ids)

        print(f"\tToken shape: {tokens.shape}")
        print(f"\tTarget phrase: [{start_idx}, {end_idx}) {target_token_strings}")
        
        total_log_prob = 0.0
        num_tokens = 0
        token_probs = []
        token_strings = []
        for i in range(start_idx, min(end_idx, tokens.shape[1])):
            masked_tokens = tokens.clone()
            original_token = masked_tokens[0, i].item()
            masked_tokens[0, i] = tokenizer.mask_token_id

            outputs = model(masked_tokens)
            log_probs = F.log_softmax(outputs.logits[0, i], dim=-1)
            token_log_prob = log_probs[original_token].item()
            total_log_prob += token_log_prob
            num_tokens += 1

            token_probs.append(token_log_prob)
            token_strings.append(tokenizer.convert_ids_to_tokens([original_token])[0])
        
        print(f"\tTokens used for calculating: {token_strings}")
        print(f"\tProbabilities: {token_probs}")
        
        proba_out = total_log_prob / num_tokens if num_tokens > 0 else 0.0
        print(f"\tAveraged Probability: {proba_out}")
        return proba_out
    
    @torch.no_grad()
    def get_target_phrase_prob_bert_without_mask(self, model_name, sentence, start_idx, end_idx):
        """
        1. Run BERT on the sentence without any masking -> returns probability distribution over entire vocabulary for each position
        2. Convert to probabilities with softmax
        3. For each token position, get probability of the actual token and then average them
        """
        print(f"[{model_name}]: Calculating the target phrase probability w/o mask: {sentence}")
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
    
        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)
        target_token_ids = token_ids[start_idx:end_idx]
        target_token_strings = tokenizer.convert_ids_to_tokens(target_token_ids)
        
        outputs = model(tokens)
        logits = outputs.logits
        
        print(f"\tToken shape: {tokens.shape}")
        print(f"\tTarget phrase: [{start_idx}, {end_idx}) {target_token_strings}")
        print(f"\tOutput logits shape: {logits.shape}")

        log_probs = F.log_softmax(logits[0], dim=-1)
        
        total_log_prob = 0.0
        num_tokens = 0
        token_probs = []
        token_strings = []
        
        for i in range(start_idx, min(end_idx, tokens.shape[1])):
            actual_token_id = tokens[0, i].item()
            token_prob = log_probs[i, actual_token_id].item()
            total_log_prob += token_prob 
            num_tokens += 1

            token_probs.append(token_prob)
            token_strings.append(tokenizer.convert_ids_to_tokens([actual_token_id])[0])
        
        print(f"\tTokens used for calculating: {token_strings}")
        print(f"\tProbabilities: {token_probs}")

        proba_out = total_log_prob / num_tokens if num_tokens > 0 else 0.0
        print(f"\tAveraged Probability: {proba_out}")
        return proba_out

    @torch.no_grad()
    def get_target_phrase_prob_causal(self, model_name, sentence, start_idx, end_idx):
        """
        1. For each token in the target phrase (e.g., tokens 5-7), get the model's prediction at the previous position (e.g.,. tokens 4-6)
        2. Extract log probability of the actual token
           Example: 
            For token 5 (읽): P (읽 | 선생님께서 책을)
                token 6 (으): P (으 | 선생님께서 책을 읽)
                token 7 (셨다): P (셨다 | 선생님께서 책을 읽으)
            And then take average of these 3 probabilities
        """
        print(f"[{model_name}]: Calculating the target phrase probability: {sentence}")
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)

        # Sanity check index bounds
        if start_idx >= len(token_ids) or end_idx > len(token_ids) or start_idx < 0:
            print(f"\tERROR: Invalid indices [{start_idx}, {end_idx}) for {len(token_ids)} tokens")
            return None
        
        target_token_ids = token_ids[start_idx:end_idx]
        target_token_strings = tokenizer.convert_ids_to_tokens(target_token_ids)
        
        outputs = model(tokens)
        logits = outputs.logits[0]

        print(f"\tToken shape: {tokens.shape}")
        print(f"\tTarget phrase: [{start_idx}, {end_idx}) {target_token_strings}")
        print(f"\tLogits shape: {logits.shape}")
        
        total_log_prob = 0.0
        num_tokens = 0
        token_probs = []
        token_strings = []
        
        for i in range(start_idx, min(end_idx, tokens.shape[1])):
            if i > 0 and i - 1 < logits.shape[0]:
                target_token_id = tokens[0, i].item()
                log_probs = F.log_softmax(logits[i - 1], dim=-1)
                token_prob = log_probs[target_token_id].item()
                total_log_prob += token_prob
                num_tokens += 1

                token_probs.append(token_prob)
                token_strings.append(tokenizer.convert_ids_to_tokens([target_token_id])[0])
        
        print(f"\tTokens used for calculating: {token_strings}")
        print(f"\tProbabilities: {token_probs}")
        
        proba_out = total_log_prob / num_tokens if num_tokens > 0 else 0.0
        print(f"\tAverage Probability: {proba_out}")
        return proba_out
    
    # Surprisal
    @torch.no_grad()
    def get_surprisal_bert_at_target(self, model_name, sentence, start_idx, end_idx):
        """
        1. For each token in the target phrase range, get log probability of that token after masking
        2. Calculate surprisal = -log_prob and average the probabilities
        """
        print(f"[{model_name}]: Calculating the target phrase surprisal: {sentence}")

        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)
        target_token_ids = token_ids[start_idx:end_idx]
        target_token_strings = tokenizer.convert_ids_to_tokens(target_token_ids)
        
        print(f"\tToken shape: {tokens.shape}")
        print(f"\tTarget phrase: [{start_idx}, {end_idx}) {target_token_strings}")

        total_surprisal = 0.0
        num_tokens = 0
        token_strings = []
        token_surprisals = []

        # Calculate surprisal for each token in the target phrase range
        for token_position in range(start_idx, min(end_idx, tokens.shape[1])):
            if token_position >= tokens.shape[1] or token_position < 0:
                continue
            
            masked_tokens = tokens.clone()
            original_token = masked_tokens[0, token_position].item()
            masked_tokens[0, token_position] = tokenizer.mask_token_id
            
            outputs = model(masked_tokens)
            log_probs = F.log_softmax(outputs.logits[0, token_position], dim=-1)
            log_prob = log_probs[original_token].item()
            
            # Surprisal in bits
            surprisal = -log_prob / np.log(2)
            total_surprisal += surprisal
            num_tokens += 1

            token_surprisals.append(surprisal)
            token_strings.append(tokenizer.convert_ids_to_tokens([original_token])[0])
        
        print(f"\tTokens used for calculating: {token_strings}")
        print(f"\tSurprisals: {token_surprisals}")

        surprisal_out = total_surprisal / num_tokens if num_tokens > 0 else 0.0
        print(f"\tAveraged Surprisal: {surprisal_out}")
        return surprisal_out
    
    @torch.no_grad()
    def get_surprisal_causal_at_target(self, model_name, sentence, start_idx, end_idx):
        """
        1. For each token in the target phrase range, get log probability from the previous position's prediction
        2. Calculate surprisal = -log_prob and average it
        """
        print(f"[{model_name}]: Calculating the target phrase surprisal: {sentence}")
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)
        target_token_ids = token_ids[start_idx:end_idx]
        target_token_strings = tokenizer.convert_ids_to_tokens(target_token_ids)
    
        outputs = model(tokens)
        logits = outputs.logits[0]

        print(f"\tToken shape: {tokens.shape}")
        print(f"\tTarget phrase: [{start_idx}, {end_idx}) {target_token_strings}")
        print(f"\tOutput logits shape: {logits.shape}")
        
        # Convert logits to log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        total_surprisal = 0.0
        token_strings = []
        token_surprisals = []
        for i in range(start_idx, end_idx):
            if i > 0:  # Can't get surprisal for first token
                # Get the log probability of actual token at position i
                # from the prediction at position i-1
                actual_token_id = token_ids[i]
                log_prob = log_probs[i-1, actual_token_id]
                
                # Surprisal = -log(P(token))
                surprisal = -log_prob.item() / np.log(2) # in bits
                total_surprisal += surprisal

                token_surprisals.append(surprisal)
                token_strings.append(tokenizer.convert_ids_to_tokens([actual_token_id])[0])
        
        # Return average surprisal for the phrase
        num_tokens = end_idx - start_idx
        if start_idx == 0:
            num_tokens -= 1  # First token has no surprisal
        
        print(f"\tTokens used for calculating: {token_strings}")
        print(f"\tSurprisals: {token_surprisals}")

        surprisal_out = total_surprisal / num_tokens if num_tokens > 0 else 0.0
        print(f"\tAveraged Surprisal: {surprisal_out}")
        return surprisal_out