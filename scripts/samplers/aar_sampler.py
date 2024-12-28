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
from typing import Dict, Optional
class AarSampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load AAR specific config
        self.hash_key = self.config["hash_key"]
        self.prefix_length = self.config["prefix_length"]
        
        # Initialize RNG
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.hash_key)

    def _get_random_u(
        self,
        input_ids: torch.Tensor,
        prefix_length: int,
        vocab_size: int
    ) -> torch.Tensor:
        """Get random u for AAR sampling"""
        batch_size, sequence_length = input_ids.shape
        time_result = torch.ones(batch_size, device=input_ids.device)

        # Calculate time result
        for i in range(prefix_length):
            time_result *= input_ids[:, -1 - i].long()
        prev_tokens = time_result % vocab_size
        
        # Generate random values for each batch
        random_u_batch = torch.stack(
            [
                torch.rand(
                    vocab_size,
                    device=self.device,
                    generator=self.rng.manual_seed(int(self.hash_key * seed.item())),
                )
                for seed in prev_tokens
            ],
            dim=0,
        )

        return random_u_batch

    
    def sample(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """AAR sampling implementation"""
        if input_ids.shape[-1] < self.prefix_length:
            raise ValueError("prefix_length must be less than or equal to the input sequence length")

        # Get random values
        batched_random_u = self._get_random_u(
            input_ids=input_ids,
            prefix_length=self.prefix_length,
            vocab_size=self.vocab_size
        )

        # Get probability distribution
        probs = self._sampling(
            logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p
        )
        
        # Apply AAR sampling
        eps = torch.finfo(logits.dtype).eps
        probs = torch.where(probs <= 0, torch.tensor(eps, device=probs.device), probs)
        exp_probs = batched_random_u ** (1 / probs)
        
        # Sample token indices
        sampled_indices = torch.multinomial(exp_probs, num_samples=1)
        
        return sampled_indices

    
    def _get_result_filename(self, prompt_idx: int) -> str:
        """Get filename for saving results"""
        return (
            f"{self.result_dir}/aar-{self.prob_type}-p{prompt_idx+1}-"
            f"{self.model_name}-temp-{self.temperature}-"
            f"topk-{self.top_k}-topp-{self.top_p}-"
            f"prefixlen-{self.prefix_length}-"
            f"{self.samples}-iter-{self.sample_iter}.json"
        )
        
