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
from .mersenne import MersenneRNG

class KTHsampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load KTH specific config
        self.key = self.config["key"]
        self.keylen = self.config["keylen"]
        self.prefix_length = self.config["prefix_length"]
        
        # Initialize RNG and generate xi
        rng = MersenneRNG(self.key)
        self.xi = torch.tensor(
            [rng.rand() for _ in range(self.keylen * self.vocab_size)]
        ).view(self.keylen, self.vocab_size).to(self.device)

    def sample(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """KTH sampling implementation"""
        if input_ids.shape[-1] < self.prefix_length:
            return logits

        # Get random shifts for each batch
        batch_size = input_ids.shape[0]
        # shifts = torch.randint(self.keylen, (batch_size,), device=self.device)
        
        # # Get xi values for current position
        # xi_batch = torch.stack(
        #     [self.xi[shifts[i], :].to(self.device) for i in range(batch_size)]
        # )

        # Get probability distribution
        probs = self._sampling(
            logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p
        )
        
        xi_indices = kwargs["xi_indices"]
        xi_batch = self.xi[xi_indices].to(self.device)

        # Apply KTH reweighting
        eps = torch.finfo(probs.dtype).eps
        probs = torch.where(probs <= 0, torch.tensor(eps, device=probs.device), probs)

        exp_probs = xi_batch ** (1 / probs)

        # Sample token indices
        sampled_indices = torch.multinomial(exp_probs, num_samples=1)
        
        return sampled_indices

    def _get_result_filename(self, prompt_idx: int) -> str:
        """Get filename for saving results"""
        return (
            f"{self.result_dir}/kth-{self.prob_type}-p{prompt_idx+1}-"
            f"{self.model_name}-temp-{self.temperature}-"
            f"keylen-{self.keylen}-topk-{self.top_k}-topp-{self.top_p}-"
            f"{self.samples}-iter-{self.sample_iter}.json"
        ) 
        