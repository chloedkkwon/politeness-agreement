import pandas as pd
import numpy as np
from setup_model import ModelSetup
from process_text import TextProcessor
from calculate_proba import ProbaCalculator


class ModelEvaluator:
    def __init__(self, device = "auto"):
        self.model_setup = ModelSetup(device)
        self.text_processor = TextProcessor(self.model_setup)
        self.probability_calculator = ProbaCalculator(self.model_setup, self.text_processor)
    
    @property
    def models(self):
        return self.model_setup.models
    
    @property
    def model_types(self):
        return self.model_setup.model_types
    
    def evaluate_batch(self, test_data, batch_size = 8):
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
                            "condition": condition,
                            "model": model_name, 
                            "sentence": sentence,
                            "target_phrase": target_phrase,
                            "avg_sentence_prob_with_mask": avg_prob_with_mask, 
                            "avg_sentence_prob_without_mask": avg_prob_without_mask,
                            "target_phrase_prob_with_mask": target_prob_with_mask,
                            "target_phrase_prob_without_mask": target_prob_without_mask,
                            "surprisal_at_target": surprisal_at_target
                        })
                    
                    except Exception as e:
                        print(f"Error evaluating {sentence} with {model_name}: {e}")
                        results.append({
                            "condition": condition,
                            "model": model_name,
                            "sentence": sentence,
                            "target_phrase": target_phrase, 
                            "avg_sentence_prob_with_mask": None,
                            "avg_sentence_prob_without_mask": None,
                            "target_phrase_prob_with_mask": None,
                            "target_phrase_prob_without_mask": None,
                            "surprisal_at_target": None
                        })
        return pd.DataFrame(results)
    
    def save_token_cache(self, filepath='token_cache.pkl'):
        self.text_processor.save_token_cache(filepath)
    
    def load_token_cache(self, filepath='token_cache.pkl'):
        self.text_processor.load_token_cache(filepath)
    
    def clear_cache(self):
        self.text_processor.clear_cache()

def sanity_check_proba(evaluator):
    """Run sanity checks on probability calculations"""
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
                prob = evaluator.probability_calculator.get_sentence_probability_bert(model_name, sentence)
            else:
                prob = evaluator.probability_calculator.get_sentence_probability_causal(model_name, sentence)
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
    
    print("\nChecking loaded models:")
    for name, model in evaluator.models.items():
        print(f"  {name}: {type(model)} - Type: {evaluator.model_types[name]}")
    
    sanity_check_proba(evaluator)
    
    # # Create sample data (for debugging)
    # test_data = create_sample_data()
    
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
