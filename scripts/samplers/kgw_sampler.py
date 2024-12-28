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
import numpy as np
from typing import Dict, Optional, Callable
from overrides import override

class KGWsampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load KGW specific config
        self.hash_key = self.config["hash_key"]
        self.prefix_length = self.config["prefix_length"]
        self.scheme = self.config["scheme"]
        self.gamma = self.config["gamma"]
        self.delta = self.config["delta"]
        self.f_scheme_map = {"time": self._f_time, "additive": self._f_additive, "skip": self._f_skip, "min": self._f_min}
        
        # Initialize RNG and PRF
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.hash_key)
        self.prf = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)

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
        
        return prf[time_result.long() % vocab_size]

    def _get_greenlist_ids(
        self,
        input_ids: torch.Tensor,
        gamma: float,
        prf: torch.Tensor,
        vocab_size: int,
        prefix_length: int
    ) -> torch.Tensor:
        """Get greenlist ids for the previous context"""
        hash_results = self._f(
            input_ids, 
            prefix_length=prefix_length, 
            prf=prf, 
            vocab_size=vocab_size
        )
        seeds = ((self.hash_key * hash_results) % vocab_size).to(self.device)

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

        greenlist_ids = vocab_permutations[:, :greenlist_size]
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
        """KGW sampling implementation"""
        if input_ids.shape[-1] < self.prefix_length:
            return logits

        # Get greenlist ids
        batched_greenlist_ids = self._get_greenlist_ids(
            input_ids,
            gamma=self.gamma,
            prf=self.prf,
            vocab_size=self.vocab_size,
            prefix_length=self.prefix_length
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
            f"{self.result_dir}/kgw-{self.prob_type}-p{prompt_idx+1}-"
            f"{self.model_name}-{self.scheme}-temp-{self.temperature}-"
            f"topk-{self.top_k}-topp-{self.top_p}-"
            f"gamma-{self.gamma}-delta-{self.delta}-"
            f"prefixlen-{self.prefix_length}-"
            f"{self.samples}-iter-{self.sample_iter}.json"
        ) 
        