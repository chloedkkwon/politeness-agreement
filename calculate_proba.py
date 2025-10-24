import torch
import torch.nn.functional as F
import numpy as np

class ProbaCalculator:
    def __init__(self, model_setup, text_processor):
        self.models = model_setup.models
        self.tokenizers = model_setup.tokenizers
        self.model_types = model_setup.model_types
        self.device = model_setup.device
        self.text_processor = text_processor
        self.tokens = {}
    
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
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        token_probs = []
        token_strings = []
        cache_key = f"{model_name}::{sentence}"

        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)

        if tokens.shape[1] <= 2: # contains only special tokens
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

            token_probs.append(total_log_prob)
            token_strings.append(tokenizer.convert_ids_to_tokens([original_token])[0])

        self.tokens[cache_key] = {'sentence': sentence,
                                  'model': model,
                                  'tokens': token_strings,
                                  'token_probs': token_probs}
        
        return total_log_prob / content_tokens if content_tokens > 0 else 0.0
    
    @torch.no_grad()
    def get_sentence_probability_bert_without_mask(self, model_name, sentence):
        """
        1. Run BERT on the sentence without any masking -> returns probability distribution over entire vocabulary for each position
        2. Convert to probabilities with softmax
        3. For each token position, get probability of the actual token and then average them
        """
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
    
        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)
        
        outputs = model(tokens)
        logits = outputs.logits
        
        probs = F.softmax(logits[0], dim=1)
        
        total_log_prob = 0.0
        num_tokens = 0
        
        for i in range(tokens.shape[1]):
            actual_token_id = tokens[0, i].item()
            token_prob = probs[i, actual_token_id].item()
            total_log_prob += np.log(token_prob) if token_prob > 0 else -float('inf')
            num_tokens += 1
        
        return total_log_prob / num_tokens if num_tokens > 0 else 0.0
    
    @torch.no_grad()
    def get_sentence_probability_causal(self, model_name, sentence):
        """
        1. Feed the entire sentence to the causal LM
        2. The model calculates cross-entropy loss (- log P (token_i | token 1... i-1)) for each token
        3. The model returns negative log probability (loss)
        4. Negate the cross entropy loss to get average log probability
        """
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)
        outputs = model(tokens, labels=tokens)

        return -outputs.loss.item() # avg log prob per token

    # Target phrase probability
    @torch.no_grad()
    def get_target_phrase_prob_bert(self, model_name, sentence, start_idx, end_idx):
        """
        1. Mask the target token(s)
        2. BERT calculates the probability of the target token(s) and takes an average
           e.g., P(읽|context) + P(으|context) + P(셨다|context) / 3
        """
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
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
    def get_target_phrase_prob_bert_without_mask(self, model_name, sentence, start_idx, end_idx):
        """
        1. Run BERT on the sentence without any masking -> returns probability distribution over entire vocabulary for each position
        2. Convert to probabilities with softmax
        3. For each token position, get probability of the actual token and then average them
        """
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
    
        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)
        
        outputs = model(tokens)
        logits = outputs.logits
        
        probs = F.softmax(logits[0], dim=1)
        
        total_log_prob = 0.0
        num_tokens = 0
        
        for i in range(start_idx, min(end_idx, tokens.shape[1])):
            actual_token_id = tokens[0, i].item()
            token_prob = probs[i, actual_token_id].item()
            total_log_prob += np.log(token_prob) if token_prob > 0 else -float('inf')
            num_tokens += 1
        
        return total_log_prob / num_tokens if num_tokens > 0 else 0.0

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
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
    
        print(f"\tDEBUG: Target phrase prob - start:{start_idx}, end:{end_idx}")
        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)
        
        outputs = model(tokens)
        logits = outputs.logits[0]
        print(f"\tDEBUG: Logits shape: {logits.shape}")
        
        total_log_prob = 0.0
        num_tokens = 0
        
        for i in range(start_idx, min(end_idx, tokens.shape[1])):
            if i > 0 and i - 1 < logits.shape[0]:
                target_token_id = tokens[0, i].item()
                log_probs = F.log_softmax(logits[i - 1], dim=-1)
                total_log_prob += log_probs[target_token_id].item()
                num_tokens += 1
        print(f"\tDEBUG: Total log prob: {total_log_prob}, num_tokens: {num_tokens}")

        return total_log_prob / num_tokens if num_tokens > 0 else 0.0
    
    # Surprisal
    @torch.no_grad()
    def get_surprisal_bert_at_target(self, model_name, sentence, start_idx, end_idx):
        """
        1. For each token in the target phrase range, get log probability of that token after masking
        2. Calculate surprisal = -log_prob and average the probabilities
        """
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)
        
        total_surprisal = 0.0
        num_tokens = 0
        
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
        
        return total_surprisal / num_tokens if num_tokens > 0 else 0.0
    
    @torch.no_grad()
    def get_surprisal_causal_at_target(self, model_name, sentence, start_idx, end_idx):
        """
        1. For each token in the target phrase range, get log probability from the previous position's prediction
        2. Calculate surprisal = -log_prob and average it
        """
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        _, token_ids = self.text_processor.tokenize_with_cache(sentence, model_name)
        tokens = torch.tensor([token_ids], device=self.device)
    
        outputs = model(tokens)
        logits = outputs.logits[0]
        
        # Convert logits to log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        total_surprisal = 0.0
        
        for i in range(start_idx, end_idx):
            if i > 0:  # Can't get surprisal for first token
                # Get the log probability of actual token at position i
                # from the prediction at position i-1
                actual_token_id = token_ids[i]
                log_prob = log_probs[i-1, actual_token_id]
                
                # Surprisal = -log(P(token))
                surprisal = -log_prob.item()
                total_surprisal += surprisal
        
        # Return average surprisal for the phrase
        num_tokens = end_idx - start_idx
        if start_idx == 0:
            num_tokens -= 1  # First token has no surprisal
        
        if num_tokens > 0:
            return total_surprisal / num_tokens
        else:
            return 0.0  # No valid tokens to compute surprisal
    
    