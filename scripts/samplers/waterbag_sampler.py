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

from .base_sampler import BaseSampler
import torch
from torch.nn import functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from .mersenne import MersenneRNG

class WaterbagSampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load Waterbag specific config
        self.hash_key = self.config["hash_key"]
        self.prefix_length = self.config["prefix_length"]
        self.gamma = self.config["gamma"]
        self.delta = self.config["delta"]
        self.scheme = self.config["scheme"]
        self.keylen = self.config["keylen"]
        self.f_scheme_map = {"time": self._f_time, "additive": self._f_additive, "skip": self._f_skip, "min": self._f_min}
        
        # Initialize RNG and key list
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.hash_key)
        
        # Generate key list using MersenneRNG
        mersenne_rng = MersenneRNG(seed=42)  # Fixed seed for reproducibility
        if self.keylen != 0:
            self.key_list = torch.tensor([int(mersenne_rng.randint() * 10e8) for _ in range(self.keylen)])
        else:
            self.key_list = torch.tensor([int(mersenne_rng.randint() * 10e8)])
            
        # check if the key in key_list is int
        assert all(isinstance(key.item(), int) for key in self.key_list), "Key in key_list is not int"
        # Create prfs based on keys
        self.prfs = torch.stack(
            [torch.randperm(self.vocab_size, device=self.device, generator=self.rng.manual_seed(key.item())) for key in self.key_list]
        )
        
        # Get the mapping from keys to prfs
        self.key_to_prf = {key: prf for key, prf in zip(self.key_list, self.prfs)}

    def _f(self, input_ids: torch.LongTensor, prefix_length: int, prf: torch.Tensor, vocab_size: int) -> int:
        """Get the previous token based on the scheme"""
        return self.f_scheme_map[self.scheme](input_ids, prefix_length, prf, vocab_size)

    # Hash Strategy
    def _f_additive(self, input_ids: torch.LongTensor, prefix_length: int, prf: torch.Tensor, vocab_size) -> int:
        """Get the previous token additive."""
        batch_size, sequence_length = input_ids.shape
        
        assert sequence_length >= prefix_length, "Sequence length must be at least as long as prefix_length"
        additive_result = torch.ones(batch_size, device=input_ids.device)
        
        for i in range(0, prefix_length):
            additive_result += input_ids[:, -1 - i].float()
            
        return prf[additive_result.long() % vocab_size]

    def _f_skip(self, input_ids: torch.LongTensor, prefix_length: int, prf: torch.Tensor, vocab_size: int) -> int:
        """Get the previous token skip."""
        batch_size, seq_length = input_ids.shape

        # Ensure that we can access the token `prefix_length` before the last one
        assert seq_length >= prefix_length, "Sequence length must be at least as long as prefix_length"

        # Extract the token that is `prefix_length` before the last token for each batch
        skip_tokens = input_ids[:, -prefix_length]  # Shape: (batch_size,)

        # Use these tokens to index into the `prf` tensor
        skip_values = prf[skip_tokens]  # Shape: (batch_size,)

        return skip_values

    def _f_min(self, input_ids: torch.LongTensor, prefix_length: int, prf: torch.Tensor, vocab_size: int) -> int:
        """Get the previous token min."""
        batch_size, seq_length = input_ids.shape
        
        assert seq_length >= prefix_length, "Sequence length must be at least as long as prefix_length"
        
        last_tokens = input_ids[:, -prefix_length:]  # (batch_size, prefix_length)
        prf_values = prf[last_tokens]  # Shape: (batch_size, prefix_length)
        # Compute the minimum along the second dimension (i.e., across the prefix tokens for each batch)
        min_values = prf_values.min(dim=1).values  # Shape: (batch_size,)
        
        return min_values

    def _f_time(
        self,
        input_ids: torch.Tensor,
        prefix_length: int,
        prf: torch.Tensor,
        vocab_size: int
    ) -> torch.Tensor:
        """Time-based hash function"""
        batch_size, sequence_length = input_ids.shape
        time_result = torch.ones(batch_size, device=input_ids.device)

        for i in range(prefix_length):
            time_result *= input_ids[:, -1 - i].float()
        
        indices = (time_result.long() % vocab_size).unsqueeze(1)
        result = torch.gather(prf, 1, indices).squeeze(1)
        return result

    def _get_greenlist_ids(
        self,
        input_ids: torch.Tensor,
        gamma: float,
        prf: torch.Tensor,
        vocab_size: int,
        prefix_length: int,
        keys: torch.Tensor,
        indicators: torch.Tensor
    ) -> torch.Tensor:
        """Get greenlist ids with left/right split based on indicators"""
        time_results = self._f(
            input_ids, 
            prefix_length=prefix_length, 
            prf=prf, 
            vocab_size=vocab_size
        )
        seeds = ((keys * time_results) % vocab_size).to(self.device)

        greenlist_size = int(vocab_size * gamma)
        rng_cuda = torch.Generator(device=self.device)

        vocab_permutations = torch.stack(
            [
                torch.randperm(
                    vocab_size,
                    device=self.device,
                    generator=rng_cuda.manual_seed(seed.item())
                )
                for seed in seeds
            ],
            dim=0,
        )
        # Split based on indicators: 1 -> first part, 0 -> second part
        effective_vocab_size = vocab_size - (vocab_size % 2)
        
        greenlist_ids = torch.where(
            indicators.unsqueeze(1) == 1,
            vocab_permutations[:, :greenlist_size],
            vocab_permutations[:, greenlist_size:effective_vocab_size]
        )
        
        return greenlist_ids

    def _calc_greenlist_mask(
        self,
        scores: torch.Tensor,
        greenlist_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the greenlist mask"""
        batch_size, vocab_size = scores.shape
        green_tokens_mask = torch.zeros(
            batch_size, vocab_size, device=scores.device, dtype=torch.bool
        )
        green_tokens_mask.scatter_(1, greenlist_token_ids, True)
        return green_tokens_mask

    def _bias_greenlist_logits(
        self,
        scores: torch.Tensor,
        greenlist_mask: torch.Tensor,
        greenlist_bias: float
    ) -> torch.Tensor:
        """Bias the greenlist logits"""
        _scores = scores.clone()
        _scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return _scores

    
    def sample(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Waterbag sampling implementation"""
        if input_ids.shape[-1] < self.prefix_length:
            return logits

        batch_size = logits.shape[0]

        # Get greenlist ids
        batched_greenlist_ids = self._get_greenlist_ids(
            input_ids,
            gamma=self.gamma,
            prf=self.selected_prfs,
            vocab_size=self.vocab_size,
            prefix_length=self.prefix_length,
            keys=self.selected_keys,
            indicators=self.selected_indicator_list
        )

        # Calculate greenlist mask
        green_tokens_mask = self._calc_greenlist_mask(
            logits,
            greenlist_token_ids=batched_greenlist_ids
        )

        # Bias logits
        logits = self._bias_greenlist_logits(
            logits,
            greenlist_mask=green_tokens_mask,
            greenlist_bias=self.delta
        )

        # Get probability distribution and sample
        probs = self._sampling(
            logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p
        )
        sampled_indices = torch.multinomial(probs, num_samples=1)

        return sampled_indices

    
    def _get_result_filename(self, prompt_idx: int) -> str:
        """Get filename for saving results"""
        return (
            f"{self.result_dir}/waterbag-{self.prob_type}-p{prompt_idx+1}-"
            f"{self.model_name}-{self.scheme}-temp-{self.temperature}-"
            f"keylen-{self.keylen}-topk-{self.top_k}-topp-{self.top_p}-"
            f"gamma-{self.gamma}-delta-{self.delta}-"
            f"prefixlen-{self.prefix_length}-"
            f"{self.samples}-"
            f"iter-{self.sample_iter}.json"
        )
