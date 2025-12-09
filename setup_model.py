import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from huggingface_hub import login
import os

class ModelSetup:
    def __init__(self, device="auto", hf_token=None): 
        self.models = {}
        self.tokenizers = {}
        self.model_types = {}
        self.tokens = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setup_huggingface_auth(hf_token)
        # self.load_models()
        self.model_configs = {
            # Bert-based models (masked LM)
            "bert-kor-base": {
                "path": "kykim/bert-kor-base", # ~ 110M
                "type": "bert", 
                "model_class": AutoModelForMaskedLM,
                "tokenizer_group": 1
                },
            "koelectra-base":{
                "path": "monologg/koelectra-base-generator", # ~ 110M
                "type": "bert",
                "model_class": AutoModelForMaskedLM,
                "tokenizer_group": 1
                },
             # Causual LM models (all ~1-1.5B range)
            "gemma": {
                "path": "google/gemma-3-1b-it",
                "type": "causal", 
                "model_class": AutoModelForCausalLM,
                "tokenizer_group": 1
                },
            "solar": {
                "path": "upstage/SOLAR-10.7B-v1.0", # much larger than other models (no other model available)
                "type": "causal",
                "model_class": AutoModelForCausalLM,
                "tokenizer_group": 2
                },
            "llama": {
                "path": "meta-llama/Llama-3.2-1B",
                "type": "causal",
                "model_class": AutoModelForCausalLM,
                "tokenizer_group": 3
                }, 
            "polyglot": {
                "path": "EleutherAI/polyglot-ko-1.3b",
                "type": "causal",
                "model_class": AutoModelForCausalLM,
                "tokenizer_group": 3
                },
            "kogpt3": {
                "path": "skt/ko-gpt-trinity-1.2B-v0.5",
                "type": "causal",
                "model_class": AutoModelForCausalLM,
                "tokenizer_group": 1
                }
            }

        print(f"ModelSetup initialized. Device: {self.device}")
        print(f"Available models: {list(self.model_configs.keys())}")

    def setup_huggingface_auth(self, hf_token=None):
        try:
            if hf_token:
                login(token=hf_token)
                print("Logged in to HuggingFace with provided token")
            elif os.getenv("HF_TOKEN"):
                login(token=os.getenv("HF_TOKEN"))
                print("Logged in to HuggingFace with environment variable")
            else:
                print("No HF token provided.")
        except Exception as e:
            print(f"Note: HuggingFace authentication not set up: {e}")

    def load_model(self, model_name):
        print(f"Loading the model on device: {self.device}")

        if model_name in self.models:
            print(f"{model_name} is already loaded.")
            return self.models[model_name], self.tokenizers[model_name]    
        
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found.")
        
        config = self.model_configs[model_name]

        try:
            print(f"Loading {model_name}...")

            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(config['path'])

            if config['type'] == 'causal':
                # Add padding
                if self.tokenizers[model_name].pad_token is None:
                    self.tokenizers[model_name].pad_token = self.tokenizers[model_name].eos_token
            
            model_kwargs = {}
            if config['type'] == 'causal':
                if self.device.type == 'cuda':
                    model_kwargs['dtype'] = torch.float16
                    model_kwargs['device_map'] = 'auto'
                else:
                    model_kwargs['dtype'] = torch.float32 # use float 32 on cpu to avoid nan
                    
            self.models[model_name] = config['model_class'].from_pretrained(
                config['path'], **model_kwargs
            )

            # Resize embeddings if we added tokens
            if config['type'] == 'bert' and self.tokenizers[model_name].mask_token_id is not None:
                self.models[model_name].resize_token_embeddings(len(self.tokenizers[model_name]))

            self.models[model_name].eval()
            self.model_types[model_name] = config['type']
            if self.device.type == 'cpu' or 'device_map' not in model_kwargs:
                self.models[model_name] = self.models[model_name].to(self.device)

            # # For debugging - Verify mask token for BERT models
            # if config['type'] == 'bert':
            #     print(f"  Mask token: {self.tokenizers[name].mask_token} (id={self.tokenizers[name].mask_token_id})")
            print(f"Loaded {model_name}.")
            return self.models[model_name], self.tokenizers[model_name]
                
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            import traceback
            traceback.print_exc()

    def unload_model(self, model_name):
        print(f"Unloading {model_name}...")

        if model_name in self.models:
            del self.models[model_name]
        if model_name in self.tokenizers:
            del self.tokenizers[model_name]
        if model_name in self.model_types:
            del self.model_types[model_name]
            
        torch.cuda.empty_cache()
        print(f"Unloaded {model_name}.")

    def unload_all_models(self):
        model_names = list(self.models.keys())
        for name in model_names:
            self.unload_model(name)
    
    def get_loaded_models(self):
        return list(self.models.keys())
    
    def get_available_models(self):
        return list(self.model_configs.keys())

    def get_memory_usage(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved.")
        else:
            print("CUDA not available")
