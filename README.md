# Large Language Model Evaluation: Korean Honorific Agreement

This project offers a comprehensive framework for evaluating how language models handle Korean honorific agreement (존댓말) by computing probabilities, surprisal values, and analyzing grammaticality judgments across multiple transformer-based models.

## Overview

This project investigates whether language models can capture the grammatical constraints of Korean honorific agreement, where subject honorifics (e.g., `-께서`) must agree with verb honorifics (e.g., `-시-`). The framework tests multiple Korean and multilingual language models on sentences with varying honorific conditions and syntactic distances.

### What are Korean Honorifics?

Korean honorifics (존댓말, jondaenmal) are a grammatical system that expresses social hierarchy and respect through morphological markers. Korean has a rich honorific system that marks respect in multiple ways:

**Subject Honorifics:**
- Subject particles change to show respect for the subject:
  - Plain: `-이/가` (e.g., 선생님**이** "teacher-NOM")
  - Honorific: `-께서` (e.g., 선생님**께서** "teacher-HON.NOM")

**Verb Honorifics:**
- Verbs take the honorific infix `-시-` (or `-으시-`) to show respect for the subject:
  - Plain: 읽**었다** "read-PAST"
  - Honorific: 읽**으셨다** "read-HON-PAST"

**Honorific Agreement:**

Korean requires **grammatical agreement** between subject and verb honorifics. When a subject is marked with the honorific particle `-께서`, the verb must also carry the honorific marker `-시-`:

✅ **Grammatical (both plain):**
- 선생님**이** 책을 읽**었다**
- "The teacher read a book" (neutral)

✅ **Grammatical (verb honorific only, acceptable with plain subjects):**
- 선생님**이** 책을 읽**으셨다**
- "The teacher read a book" (polite)

❌ **Ungrammatical (subject honorific without verb honorific):**
- \*선생님**께서** 책을 읽**었다**
- "The teacher-HON read a book" (missing verb honorific)

✅ **Grammatical (both honorific):**
- 선생님**께서** 책을 읽**으셨다**
- "The teacher-HON read-HON a book" (fully respectful)

This agreement pattern creates a grammatical dependency between the subject and verb that language models must learn to respect. The key insight is that honorific marking on the subject (-께서) obligatorily requires honorific marking on the verb (-시-), making the noun_only condition ungrammatical.

**Why This Matters for Language Models:**

Testing honorific agreement in Korean provides insights into whether language models learn:
1. **Long-distance dependencies**: The subject and verb can be separated by multiple words
2. **Morphosyntactic agreement**: Models must track morphological features across syntactic positions
3. **Grammaticality distinctions**: Models should assign lower probabilities to ungrammatical constructions
4. **Language-specific constraints**: Korean honorifics are not present in most training data languages

By measuring sentence probabilities and surprisal values, we can quantify whether models have internalized these grammatical constraints or are merely capturing surface-level patterns.

## Supported Models

### BERT-based Models (Masked LM)
- **bert-kor-base**: `kykim/bert-kor-base` (~110M parameters)
- **koelectra-base**: `monologg/koelectra-base-generator` (~110M parameters)

### Causal Language Models
- **gemma**: `google/gemma-3-1b-it` (~1B parameters)
- **solar**: `upstage/SOLAR-10.7B-v1.0` (~10.7B parameters)
- **llama**: `meta-llama/Llama-3.2-1B` (~1B parameters)
- **polyglot**: `EleutherAI/polyglot-ko-1.3b` (~1.3B parameters)
- **kogpt3**: `skt/ko-gpt-trinity-1.2B-v0.5` (~1.2B parameters)

## Project Structure

```
├── setup_model.py          # Model loading and initialization
├── process_text.py         # Text tokenization and target phrase detection
├── calculate_proba.py      # Probability and surprisal calculations
├── generate_sentences.py   # Test data generation
├── evaluate.py            # Main evaluation pipeline
└── data/
    ├── sentences.csv       # Generated test sentences
    ├── results_evaluation.csv   # Evaluation results
    └── token_cache.pkl     # Cached tokenizations
```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### HuggingFace Authentication

Some models require HuggingFace authentication. Set your token as an environment variable:

```bash
export HF_TOKEN="your_huggingface_token"
```

Or pass it directly when initializing `ModelSetup`:

```python
model_setup = ModelSetup(hf_token="your_token")
```

## Usage

### 1. Generate Test Sentences

```python
python generate_sentences.py
```

This creates `data/sentences.csv` with test sentences in 4 honorific conditions × 2 distance conditions:

**Honorific Conditions:**
- `no_honorific`: Plain subject + plain verb (grammatical)
- `verb_only`: Plain subject + honorific verb (grammatical)
- `noun_only`: Honorific subject + plain verb (ungrammatical)
- `all_honorific`: Honorific subject + honorific verb (grammatical)

**Distance Conditions:**
- `close`: Adverb at sentence beginning (subject adjacent to verb)
- `far`: Adverb between subject and verb (increased distance)

### 2. Run Evaluation

```python
python evaluate.py
```

This will:
1. Load each model sequentially
2. Calculate probabilities and surprisal for all test sentences
3. Save results to `data/results_evaluation.csv`
4. Print summary statistics by model and condition

### 3. Custom Evaluation

```python
from evaluate import ModelEvaluator, load_csv

# Initialize evaluator
evaluator = ModelEvaluator(device="auto")

# Load test data
test_data = load_csv('data/sentences.csv')

# Run evaluation
results = evaluator.evaluate_batch(test_data, batch_size=8)

# Save results
results.to_csv('my_results.csv', index=False)
```

## Main Components

### ModelSetup (`setup_model.py`)

Manages model loading, unloading, and configuration.

```python
from setup_model import ModelSetup

setup = ModelSetup(device="auto", hf_token="your_token")
model, tokenizer = setup.load_model("bert-kor-base")
setup.unload_model("bert-kor-base")
```

### TextProcessor (`process_text.py`)

Handles tokenization and target phrase detection with support for different tokenization schemes.

```python
from process_text import TextProcessor

processor = TextProcessor(model_setup)
start_idx, end_idx, tokens = processor.find_target_phrase_token(
    model_name="bert-kor-base",
    sentence="선생님께서 책을 읽으셨다.",
    target_phrase="읽으셨다"
)
```

**Key Features:**
- Character-level target phrase matching
- Support for 3 tokenization groups:
  - Group 1: Standard tokenization (BERT, GEMMA, KoGPT3)
  - Group 2: Byte-level tokenization (SOLAR)
  - Group 3: Raw tokens (Llama, Polyglot)
- Token caching for efficiency

### ProbaCalculator (`calculate_proba.py`)

Computes probabilities and surprisal values.

```python
from calculate_proba import ProbaCalculator

calculator = ProbaCalculator(model_setup, text_processor)

# For BERT models
sentence_prob = calculator.get_sentence_probability_bert(model_name, sentence)
target_prob = calculator.get_target_phrase_prob_bert(model_name, sentence, start_idx, end_idx)
surprisal = calculator.get_surprisal_bert_at_target(model_name, sentence, start_idx, end_idx)

# For causal models
sentence_prob = calculator.get_sentence_probability_causal(model_name, sentence)
target_prob = calculator.get_target_phrase_prob_causal(model_name, sentence, start_idx, end_idx)
surprisal = calculator.get_surprisal_causal_at_target(model_name, sentence, start_idx, end_idx)
```

**Computed Metrics:**

For BERT models:
- `avg_sentence_prob_with_mask`: Average log probability with masking
- `avg_sentence_prob_without_mask`: Average log probability without masking
- `target_phrase_prob_with_mask`: Target phrase log probability with masking
- `target_phrase_prob_without_mask`: Target phrase log probability without masking
- `surprisal_at_target`: Average surprisal (in bits) at target phrase

For causal models:
- `avg_sentence_prob_without_mask`: Average log probability
- `target_phrase_prob_without_mask`: Target phrase log probability
- `surprisal_at_target`: Average surprisal (in bits) at target phrase

### ModelEvaluator (`evaluate.py`)

High-level interface for batch evaluation.

```python
from evaluate import ModelEvaluator

evaluator = ModelEvaluator(device="auto")
results_df = evaluator.evaluate_batch(test_data, batch_size=8)
```

## Output Format

The evaluation produces a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `item_number` | Test item identifier |
| `condition` | Honorific condition (no_honorific, verb_only, noun_only, all_honorific) |
| `distance` | Syntactic distance (close, far) |
| `model` | Model name |
| `sentence` | Full test sentence |
| `target_phrase` | Target verb phrase |
| `subject` | Subject noun phrase |
| `verb_phrase` | Verb phrase |
| `grammatical` | Ground truth grammaticality (True/False) |
| `avg_sentence_prob_with_mask` | Average sentence log probability (BERT, with masking) |
| `avg_sentence_prob_without_mask` | Average sentence log probability (all models) |
| `target_phrase_prob_with_mask` | Target phrase log probability (BERT, with masking) |
| `target_phrase_prob_without_mask` | Target phrase log probability (all models) |
| `surprisal_at_target` | Average surprisal at target phrase (bits) |

## Example Test Sentences

Here are example test sentences showing the four honorific conditions:

```
1. no_honorific (grammatical):
   어제 저녁에 혼자서 선생님이 책을 읽었다.
   Yesterday evening alone teacher-NOM book-OBJ read-PAST
   "The teacher read a book alone yesterday evening." (neutral)

2. verb_only (grammatical):
   어제 저녁에 혼자서 선생님이 책을 읽으셨다.
   Yesterday evening alone teacher-NOM book-OBJ read-HON-PAST
   "The teacher read a book alone yesterday evening." (polite)

3. noun_only (ungrammatical):
   어제 저녁에 혼자서 선생님께서 책을 읽었다.
   Yesterday evening alone teacher-HON.NOM book-OBJ read-PAST
   "The teacher-HON read a book alone yesterday evening." (missing verb honorific)

4. all_honorific (grammatical):
   어제 저녁에 혼자서 선생님께서 책을 읽으셨다.
   Yesterday evening alone teacher-HON.NOM book-OBJ read-HON-PAST
   "The teacher-HON read-HON a book alone yesterday evening." (fully respectful)
```

**Distance Manipulation:**

- **Close condition**: Adverb at beginning → Subject adjacent to verb
  - `[Adverb] [Subject] [Object] [Verb]`
  - `어제 저녁에 혼자서 선생님께서 책을 읽으셨다`

- **Far condition**: Adverb between subject and verb → Increased distance
  - `[Subject] [Adverb] [Object] [Verb]`
  - `선생님께서 어제 저녁에 혼자서 책을 읽으셨다`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{korean_honorific_eval,
  author = {Kwon, Chloe D., Cho, Youngdong},
  title = {Large Language Model Evaluation: Korean Honorific Agreement},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/chloedkkwon/politeness-agreement}},
  url = {https://github.com/chloedkkwon/politeness-agreement}
}
```

## Acknowledgments

This project uses models from:
- Kiyoung Kim (bert-kor-base): https://github.com/kiyoungkim1/LMkor
- Jangwon Park (KoELECTRA): https://github.com/monologg/KoELECTRA
- Google (GEMMA): https://huggingface.co/google/gemma-3-1b-it
- Upstage (SOLAR): https://huggingface.co/upstage/SOLAR-10.7B-v1.0
- Meta (Llama): https://huggingface.co/meta-llama/Llama-3.2-1B
- EleutherAI (Polyglot): https://huggingface.co/EleutherAI/polyglot-ko-1.3b
- SK Telecom (KoGPT3): https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5
