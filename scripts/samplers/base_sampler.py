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

import torch
from torch.nn import functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import os
import ujson as json
from transformers import AutoTokenizer
import numpy as np
import os
import random
from itertools import product
from utils.prompt_manager import PromptManager


class BaseSampler(ABC):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        prob_type: str,
        batch_size: int,
        samples: int,
        temperature: float,
        top_k: int,
        top_p: float,
        sample_iter: int,
        sampler_type: str,
        config_path: str = "config/sampler_config.json",
        result_dir: str = "../../data/results",
        prompt_dir: str = "../../data/prompts",
        logits_dir: str = "../../data/logits"
    ):
        """Initialize base sampler with common parameters"""
        self.model_name = model_name
        self.prob_type = prob_type
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = batch_size
        self.samples = samples
        self.temperature = temperature
        self.top_k = top_k 
        self.top_p = top_p
        self.sample_iter = sample_iter
        self.sampler_type = sampler_type
        
        # Setup paths
        if self.prob_type == "prob2_5gram" or self.prob_type == "prob2":
            self.result_dir = os.path.join(result_dir, f"prob2")
        elif self.prob_type == "prob1":
            self.result_dir = os.path.join(result_dir, f"prob1")
        else:
            raise ValueError(f"Invalid prob type: {self.prob_type}")
        self.prompt_dir = prompt_dir
        self.logits_dir = logits_dir
        
        # Load sampler config
        self.config = self._load_config(config_path)
        
        # Load tokenizer and set vocab size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.vocab_size = len(self.tokenizer)
        
        # Load prompts
        if self.prob_type == "prob2_5gram":
            self.prompt_type = "5gram"
        # elif self.sampler_type in ["kth", "its"]:
        #     self.prompt_type = "fixkey"
        else:
            self.prompt_type = "ngram"
        
        self.prompt_manager = PromptManager(prompt_dir=self.prompt_dir, prompt_type=self.prompt_type)
        self.prompts = self.prompt_manager.load_prompts()
        
        # Load logits
        self.logits_dict = self._load_logits()
        
        # Load fill parts
        if self.prob_type == "prob1":
            self.fill_parts = self._get_fill_parts()

    def _load_config(self, config_path: str) -> Dict:
        """Load sampler configuration from JSON file"""
        with open(config_path) as f:
            config = json.load(f)
        # Get config for specific sampler type
        if self.sampler_type not in config:
            raise ValueError(f"No configuration found for sampler type: {self.sampler_type}")
        return config[self.sampler_type]

    def _load_logits(self) -> List[Dict]:
        """Load logits from pickle files"""
        import pickle
        logits = []
        for i in range(2):
            if self.prob_type == "prob2_5gram":
                with open(f"{self.logits_dir}/5gram-p{i+1}-logits-{self.model_name}.pickle", "rb") as f:
                    print(f"Loading logits from {self.logits_dir}/5gram-p{i+1}-logits-{self.model_name}.pickle")
                    logits.append(pickle.load(f))
            else:
                with open(f"{self.logits_dir}/ngram-p{i+1}-logits-{self.model_name}.pickle", "rb") as f:
                    print(f"Loading logits from {self.logits_dir}/ngram-p{i+1}-logits-{self.model_name}.pickle")
                    logits.append(pickle.load(f))
        return logits

    @torch.no_grad()
    def _sampling(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Base sampling method that applies temperature, top-k and top-p sampling"""
        assert temperature > 0, "Temperature must be positive"
        
        _logits = logits / temperature

        # Apply top-k sampling
        if top_k is not None and top_k > 0:
            top_k = min(top_k, _logits.size(-1))
            indices_to_remove = _logits < torch.topk(_logits, top_k)[0][..., -1, None]
            _logits[indices_to_remove] = float("-inf")

        # Apply top-p sampling
        if top_p is not None and 0 < top_p < 1:
            sorted_logits, sorted_indices = torch.sort(_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            
            if sorted_indices_to_remove[..., 1:].any():
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            _logits[indices_to_remove] = float("-inf")

        # Get probability distribution
        probs = F.softmax(_logits, dim=-1)
        return probs

    def sample(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Base sampling method"""
        return torch.multinomial(self._sampling(logits, self.temperature, self.top_k, self.top_p), num_samples=1)


    def _get_fill_parts(self) -> List[str]:
        """Get fill parts for prob1 sampling"""
        letters = [f" {chr(i)}" for i in range(65, 91)]
        numbers_en = [" zero", " one", " two", " three", " four", 
                        " five", " six", " seven", " eight", " nine"]
        animal_choice = [" cat", " dog", " tiger", " lion"]
        combinations = ["".join(comb) for comb in product(letters, numbers_en, animal_choice)]
        
        # Fixed seed random sampling
        random.seed(64)
        return random.sample(combinations, self.config["fill_length"])
    
    def _get_logits(self, ctx: str, logits: Dict) -> torch.Tensor:
        """Get logits for prob1 sampling"""
        cur_logits = logits
        pre_str = "Example12:"
        pre_tokens = self.tokenizer.encode(pre_str, add_special_tokens=False)
        pre_ctx_tokens = self.tokenizer.encode(pre_str + ctx, add_special_tokens=False)
        ctx_token = pre_ctx_tokens[len(pre_tokens):]
        
        for id in ctx_token:
            cur_logits = cur_logits[id]
            
        assert len(cur_logits.keys()) == 1, f"cur_logits keys: {cur_logits.keys()}, decoded cur_logits: {[self.tokenizer.decode(token) for token in cur_logits.keys() if isinstance(token, int)]}"
        return cur_logits["logits"].clone().to(self.device)
    
    @torch.no_grad()
    def sample_batch(
        self,
        prompt_idx: int,
        **kwargs
    ) -> Tuple[np.ndarray, List]:
        """Common batch sampling implementation"""
        if self.prob_type == 'prob1':
            batch_size = self.samples // len(self.fill_parts)
            if self.sampler_type in ["kth", "its"]:
                kwargs["xi_indices"] = torch.randint(self.config["keylen"], (batch_size,), device=self.device)
            if self.sampler_type == "waterbag":
                # Randomly sample batch_size keys from self.key_list
                indices = torch.randint(0, len(self.key_list), (batch_size,))
                self.selected_keys = self.key_list[indices].to(self.device)
                self.selected_prfs = self.prfs[indices].to(self.device)
                
                # random generate a [0,1] binary list of batch size
                self.indicator_list = torch.randint(0, 2, (batch_size,)).to(self.device)
                self.selected_indicator_list = self.indicator_list[indices].to(self.device)
            
            input_ids = self.tokenizer.encode(
                self.prompts[prompt_idx] + self.fill_part,
                return_tensors="pt"
            ).to(self.device)
            input_ids = input_ids.repeat(batch_size, 1)
            logits = self._get_logits(self.fill_part, self.logits_dict[prompt_idx])
            tokens = self.sample(
                logits=logits.repeat(batch_size, 1),
                input_ids=input_ids,
                **kwargs
            ).squeeze(1)
            
            if self.sampler_type == "waterbag":
                del self.selected_keys, self.indicator_list, self.selected_prfs
            return tokens.cpu().numpy()
        
        ## prob2 & prob2_5gram sampling
        # Setup batch
        cur_logits_batch = [self.logits_dict[prompt_idx]] * self.batch_size
        input_ids = self.tokenizer(
            self.prompts[prompt_idx], 
            return_tensors="pt"
        )["input_ids"].to(self.device)
        input_ids = input_ids.repeat(self.batch_size, 1)
        
        # Special handling for kth/its samplers
        if self.sampler_type in ["kth", "its"]:
            shifts = torch.randint(self.config["keylen"], (self.batch_size,), device=self.device)
            cnts = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
            # if self.sampler_type == "its":
            #     origin_pi_batch = self.pi.repeat(self.batch_size, 1)

        # Track active samples
        active = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        token_ids = torch.full((self.batch_size,), -1, dtype=torch.long, device=self.device)
        context_ids = [None for _ in range(self.batch_size)]
        
        if self.sampler_type == "waterbag":
            # Randomly sample batch_size keys from self.key_list
            indices = torch.randint(0, len(self.key_list), (self.batch_size,))
            self.sampled_keys = self.key_list[indices].to(self.device)
            self.sampled_prfs = self.prfs[indices].to(self.device)
            # random generate a [0,1] binary list of batch size
            self.indicator_list = torch.randint(0, 2, (self.batch_size,)).to(self.device)

        while active.any():
            active_indices = torch.nonzero(active).squeeze(1)
            logits_batch = torch.stack([
                (cur_logits_batch[i]["logits"]).squeeze(0).to(self.device)
                for i in active_indices
            ])
            
            if self.sampler_type in ["kth", "its"]:
                xi_indices = (shifts[active_indices] + cnts[active_indices]) % self.config["keylen"]
                kwargs["xi_indices"] = xi_indices
                
            if self.sampler_type == "waterbag":
                self.selected_keys = self.sampled_keys[active_indices]
                self.selected_prfs = self.sampled_prfs[active_indices]
                self.selected_indicator_list = self.indicator_list[active_indices]

            # Call child class sampling method
            tokens = self.sample(
                logits=logits_batch,
                input_ids=input_ids[active_indices],
                **kwargs
            ).squeeze(1)

            # Update contexts and check completion
            token_idx = 0
            for i in range(self.batch_size):
                if not active[i]:
                    continue

                token_id = tokens[token_idx].item()
                token_idx += 1

                # Update input ids
                input_ids[i] = torch.cat([
                    input_ids[i][1:],
                    torch.tensor([token_id], device=self.device)
                ])

                if token_id in cur_logits_batch[i]:
                    cur_logits_batch[i] = cur_logits_batch[i][token_id]
                    
                    if context_ids[i] is None:
                        context_ids[i] = [token_id]
                    else:
                        context_ids[i].append(token_id)
                else:
                    if len(cur_logits_batch[i]) == 1 and "logits" in cur_logits_batch[i]:
                        token_ids[i] = token_id
                        active[i] = False
                    else:
                        token_ids[i] = -1
                        active[i] = False
        if self.sampler_type == "waterbag":
            del self.sampled_keys, self.indicator_list, self.sampled_prfs, self.selected_keys, self.selected_prfs, self.selected_indicator_list

        if self.sampler_type not in ["kth", "its"]:
            return token_ids.cpu().numpy(), context_ids
        else:
            return token_ids.cpu().numpy(), context_ids, shifts.cpu().numpy()

    @torch.no_grad()
    def run(self):
        """Run sampling for all prompts and save results"""
        for prompt_idx in range(2):
            print(f"Processing prompt {prompt_idx}...")
            
            # Setup result tracking
            mapping = {}
            
            if self.prob_type == 'prob1':
                # Prob1 sampling
                batch_size = self.samples // len(self.fill_parts)
                for fill_part in self.fill_parts:
                    print(f"Processing fill part: {fill_part}")
                    
                    self.fill_part = fill_part         
                    if self.sampler_type == "waterbag":
                        # Sample batch_size keys from self.key_list
                        indices = torch.randint(0, len(self.key_list), (batch_size,))
                        sampled_keys = self.key_list[indices].to(self.device)
                        # random generate a [0,1] binary list of batch size
                        indicator_list = torch.randint(0, 2, (batch_size,)).to(self.device)
                        tokens = self.sample_batch(
                            prompt_idx=prompt_idx,
                            sampled_keys=sampled_keys,
                            indicator_list=indicator_list,
                            **self.config
                        )
                        
                    else:
                        tokens = self.sample_batch(
                            prompt_idx=prompt_idx,
                            **self.config
                        )
                            
                    if fill_part not in mapping:
                        mapping[fill_part] = {}
                        mapping[fill_part]["S_wm"] = [0] * self.vocab_size
                        
                    for token in tokens:
                        mapping[fill_part]["S_wm"][token] += 1

            else: # prob2 sampling
                for iter_idx in range(self.samples // self.batch_size):
                    print(f"Iter: {iter_idx + 1}/{self.samples // self.batch_size}")
                    
                    if self.sampler_type not in ["kth", "its"]:
                        wm_tokens, wm_contexts = self.sample_batch(
                            prompt_idx=prompt_idx,
                            **self.config
                        )
                    else:
                        wm_tokens, wm_contexts, shifts = self.sample_batch(
                            prompt_idx=prompt_idx,
                            iter_idx=iter_idx,
                            **self.config
                        )

                    # Process valid samples
                    wm_valid_indices = np.where(wm_tokens != -1)[0]
                    wm_valid_contexts = [wm_contexts[i] for i in wm_valid_indices]
                    valid_wm_tokens = wm_tokens[wm_valid_indices]
                    if self.sampler_type in ["kth", "its"]:
                        valid_shifts = shifts[wm_valid_indices]
                    for i, ctx in enumerate(wm_valid_contexts):
                        context_str = f' {self.tokenizer.decode(ctx).rsplit("|")[0].strip()}'
                        token = valid_wm_tokens[i]

                        if context_str not in mapping:
                            mapping[context_str] = {}
                            mapping[context_str]["S_wm"] = [0] * self.vocab_size
                            if self.sampler_type in ["kth", "its"]:
                                mapping[context_str]["shifts"] = [0] * self.config["keylen"]
                        mapping[context_str]["S_wm"][token] += 1
                        if self.sampler_type in ["kth", "its"]:
                            mapping[context_str]["shifts"][valid_shifts[i]] += 1

            # Save results
            results = {
               str(k): v for k, v in mapping.items()
            }
            
            self._save_results(prompt_idx, results)
            
            # Clear GPU memory
            torch.cuda.empty_cache()

    def _get_result_filename(self, prompt_idx: int) -> str:
        """Get filename for saving results based on prob type"""
        return (
            f"{self.result_dir}/unwatermarked-{self.prob_type}-p{prompt_idx+1}-"
            f"{self.model_name}-temp-{self.temperature}-"
            f"topk-{self.top_k}-topp-{self.top_p}-"
            f"{self.samples}-iter-{self.sample_iter}.json"
        )

    def _save_results(self, prompt_idx: int, results: Dict):
        """Save sampling results to file"""
        filename = self._get_result_filename(prompt_idx)
        with open(filename, "w") as f:
            json.dump(results, f, separators=(",", ":")) 