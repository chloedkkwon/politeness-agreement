import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

class ModelSetup:
    def __init__(self, device="auto"): 
        self.models = {}
        self.tokenizers = {}
        self.model_types = {}
        self.tokens = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models()
    
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
                "path": "google/gemma-3-1b-it",
                "type": "causal", 
                "model_class": AutoModelForCausalLM
            },

            "solar": {
                "path": "upstage/solar-1.0-mini-chat", 
                "type": "causal",
                "model_class": AutoModelForCausalLM
            },

            "llama": {
                "path": "meta-llama/Llama-3.2-1B",
                "type": "causal",
                "model_class": AutoModelForCausalLM
            }, 

            "polyglot": {
                "path": "EleutherAI/polyglot-ko-1.3b",
                "type": "causal",
                "model_class": AutoModelForCausalLM
            }
        }

        for name, config in model_configs.items():
            try:
                print(f"Loading {name}...")

                self.tokenizers[name] = AutoTokenizer.from_pretrained(config['path'])

                if config['type'] == 'causal':
                    # Add padding
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

                # # For debugging - Verify mask token for BERT models
                # if config['type'] == 'bert':
                #     print(f"  Mask token: {self.tokenizers[name].mask_token} (id={self.tokenizers[name].mask_token_id})")
                print(f"Loaded {name}.")
                
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                import traceback
                traceback.print_exc()