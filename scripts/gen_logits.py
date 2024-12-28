# Copyright 2024 THU BPM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.prompt_manager import PromptManager
from utils.generation_utils import get_first_token_logits, get_logits_dict, update_root_dict_with_nested


def run(model_name, model_path, prompt_type, combinations_rules, prompt_dir, logits_dir, device):
    # Initialize PromptManager
    prompt_manager = PromptManager(prompt_dir, prompt_type)
    prompts = prompt_manager.load_prompts()
    combinations = prompt_manager.generate_combinations(combinations_rules)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, device=device)
    vocab_size = len(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    logits_paths = [
        os.path.join(logits_dir, f"{prompt_type}-p1-logits-{model_name}.pickle"),
        os.path.join(logits_dir, f"{prompt_type}-p2-logits-{model_name}.pickle"),
    ]

    for idx, logits_file_path in enumerate(logits_paths):
        with torch.no_grad():
            results = {}
            print(f"Getting logits for: {prompts[idx]}")
            D_original = get_first_token_logits(model, tokenizer, prompts[idx], vocab_size, device)
            results["logits"] = D_original

            for item in combinations:
                print(f"Getting logits for: {item}")
                nested_dict = get_logits_dict(prompts[idx], item, model, tokenizer, vocab_size, device)
                update_root_dict_with_nested(results, nested_dict)

        os.makedirs(os.path.dirname(logits_file_path), exist_ok=True)
        with open(logits_file_path, "wb") as file:
            pickle.dump(results, file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="opt27b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, default="ngram")
    parser.add_argument("--prompt_dir", type=str, default="../data/prompts")
    parser.add_argument("--logits_dir", type=str, default="../data/logits")
    
    args = parser.parse_args()

    # Default combination rules
    # combinations for all possible contexts
    if args.prompt_type == "ngram":
        combinations_rules = [
            [f" {chr(i)}" for i in range(65, 91)],  # Letters
            [" zero", " one", " two", " three", " four", " five", " six", " seven", " eight", " nine"],  # Numbers
            [" cat", " dog", " tiger", " lion"],  # Animals
        ]
    elif args.prompt_type == "fixkey":
        combinations_rules = [
            [f" {chr(i)}" for i in range(65, 91)],  # Letters
            [" zero", " one", " two", " three", " four", " five", " six", " seven", " eight", " nine"],  # Numbers
            [" cat", " dog", " tiger", " lion"],  # Animals
            [" |"],  # Separator
        ]
    elif args.prompt_type == "5gram":
        combinations_rules = [
            [f" {chr(i)}" for i in range(65, 91)],  # Letters
            [" zero", " one", " two", " three", " four", " five", " six", " seven", " eight", " nine"],  # Numbers
            [" cat", " dog", " tiger", " lion"],  # Animals
            [" apple", " banana", " orange"],  # Fruits
            [" car", " bus", " truck"],  # Cars
        ]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run(
        model_name=args.model_name,
        model_path=args.model_path,
        prompt_type=args.prompt_type,
        combinations_rules=combinations_rules,
        prompt_dir=args.prompt_dir,
        logits_dir=args.logits_dir,
        device=device,
    )