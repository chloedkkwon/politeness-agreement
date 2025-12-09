import pandas as pd
import numpy as np
from setup_model import ModelSetup
from process_text import TextProcessor
from calculate_proba import ProbaCalculator
import torch
import gc # for cache clearing

class ModelEvaluator:
    def __init__(self, device = "auto"):
        self.model_setup = ModelSetup(device)
        self.text_processor = TextProcessor(self.model_setup)
        self.probability_calculator = ProbaCalculator(self.model_setup, self.text_processor)
    
    @property
    def models(self):
        return self.model_setup.get_available_models()
    
    @property
    def model_types(self):
        return self.model_setup.model_types
    
    def evaluate_batch(self, test_data, batch_size = 8):
        results = []
        
        # Group by model to minimize model switching overhead
        for model_name in self.models:
            print(f"\nEvaluating with {model_name}...")

            model, tokenizer = self.model_setup.load_model(model_name)
            
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size]
                
                for item in batch:
                    item_number = item["item_number"]
                    sentence = item["sentence"]
                    target_phrase = item["target_phrase"]
                    condition = item["condition"]
                    distance = item["distance"]
                    subject = item["subject"]
                    verb_phrase = item["verb_phrase"]
                    grammatical = item["grammatical"]

                    target_info = self.text_processor.find_target_phrase_token(model_name, sentence, target_phrase)
                    start_idx, end_idx, target_tokens = target_info
                    
                    try:
                        # Calculate
                        if self.model_types[model_name] == "bert":
                            avg_prob_with_mask = self.probability_calculator.get_sentence_probability_bert(model_name, sentence)
                            avg_prob_without_mask = self.probability_calculator.get_sentence_probability_bert_without_mask(model_name, sentence)
                            target_prob_with_mask = self.probability_calculator.get_target_phrase_prob_bert(model_name, sentence, start_idx, end_idx)
                            target_prob_without_mask = self.probability_calculator.get_target_phrase_prob_bert_without_mask(model_name, sentence, start_idx, end_idx)
                            surprisal_at_target = self.probability_calculator.get_surprisal_bert_at_target(model_name, sentence, start_idx, end_idx)
                        
                        else:
                            avg_prob_with_mask = None
                            avg_prob_without_mask = self.probability_calculator.get_sentence_probability_causal(model_name, sentence)
                            target_prob_with_mask = None
                            target_prob_without_mask = self.probability_calculator.get_target_phrase_prob_causal(model_name, sentence, start_idx, end_idx)
                            surprisal_at_target = self.probability_calculator.get_surprisal_causal_at_target(model_name, sentence, start_idx, end_idx)
                        
                        results.append({
                            "item_number": item_number,
                            "condition": condition,
                            "distance": distance,
                            "model": model_name, 
                            "sentence": sentence,
                            "target_phrase": target_phrase,
                            "avg_sentence_prob_with_mask": avg_prob_with_mask, 
                            "avg_sentence_prob_without_mask": avg_prob_without_mask,
                            "target_phrase_prob_with_mask": target_prob_with_mask,
                            "target_phrase_prob_without_mask": target_prob_without_mask,
                            "surprisal_at_target": surprisal_at_target,
                            "subject": subject, 
                            "verb_phrase": verb_phrase,
                            "grammatical": grammatical
                        })
                    
                    except Exception as e:
                        print(f"Error evaluating {sentence} with {model_name}: {e}")
                        results.append({
                            "item_number": item_number,
                            "condition": condition,
                            "distance": distance,
                            "model": model_name,
                            "sentence": sentence,
                            "target_phrase": target_phrase, 
                            "avg_sentence_prob_with_mask": None,
                            "avg_sentence_prob_without_mask": None,
                            "target_phrase_prob_with_mask": None,
                            "target_phrase_prob_without_mask": None,
                            "surprisal_at_target": None,
                            "subject": subject, 
                            "verb_phrase": verb_phrase,
                            "grammatical": grammatical
                        })
            print(f"Unloading {model_name}...")
            self.model_setup.unload_model(model_name)

            del model
            del tokenizer

            gc.collect()
            torch.cuda.empty_cache()

            print(f"Memory after unloading:")
            self.model_setup.get_memory_usage()

        return pd.DataFrame(results)
    
    def save_token_cache(self, filepath='token_cache.pkl'):
        self.text_processor.save_token_cache(filepath)
    
    def load_token_cache(self, filepath='token_cache.pkl'):
        self.text_processor.load_token_cache(filepath)
    
    def clear_cache(self):
        self.text_processor.clear_cache()

def load_csv(filename):
    df = pd.read_csv(filename, encoding='utf-8')
    if 'grammatical' in df.columns:
        df['grammatical'] = df['grammatical'].astype(str).str.lower() == 'true'
    test_data = df.to_dict('records')
    return test_data


def main():
    print("Initializing the model evaluator...")
    evaluator = ModelEvaluator()
    test_data = load_csv('data/sentences.csv')
    
    # Run batch evaluation
    print("Running evaluation...")
    results_df = evaluator.evaluate_batch(test_data, batch_size=8)
    
    # Save results
    output_file = "data/results_evaluation.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Results saved to {output_file}")
    
    # Display summary
    print("\nSummary by condition and model:")
    summary = results_df.groupby(['model', 'condition']).agg({
        'avg_sentence_prob_without_mask': 'mean',
        'target_phrase_prob_without_mask': 'mean', 
        'surprisal_at_target': 'mean'
    }).round(4)
    print(summary)
    
    evaluator.save_token_cache('data/token_cache.pkl')
    
    evaluator.clear_cache()
    
    return results_df


if __name__ == "__main__":
    results = main()
