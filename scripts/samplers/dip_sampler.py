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
import hashlib
from typing import Dict, Optional, Tuple
import random

class DIPSampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load DIP specific config
        random.seed(self.config["key"])
        self.hash_key = random.getrandbits(1024).to_bytes(128, "big")
        self.alpha = self.config["alpha"]
        self.prefix_length = self.config["prefix_length"]

    def _get_rng_seed(self, context_code: bytes) -> int:
        """Get random seed from context code and private key"""
        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.hash_key)
        
        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
        return seed

    def _extract_context_code(self, context: torch.Tensor, prefix_length: int) -> torch.Tensor:
        """Extract context code from context tensor"""
        if prefix_length == 0:
            return context
        return context[:, -prefix_length:]

    def _get_seeds_for_cipher(self, input_ids: torch.Tensor, prefix_length: int) -> list:
        """Get seeds for cipher using vectorized operations"""
        context_codes = self._extract_context_code(input_ids, prefix_length)
        batch_size = context_codes.size(0)
        seeds = []
        
        for i in range(batch_size):
            context_code = context_codes[i].detach().cpu().numpy().tobytes()
            seed = self._get_rng_seed(context_code)
            seeds.append(seed)
            
        return seeds

    def _from_random(self, rng: list[torch.Generator], vocab_size: int) -> torch.LongTensor:
        """Generate permutation from random number generator"""
        batch_size = len(rng)
        shuffle = torch.stack([
            torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
            for i in range(batch_size)
        ])
        return shuffle

    def _reweight_logits(
        self,
        shuffle: torch.LongTensor,
        p_logits: torch.FloatTensor,
        alpha: float
    ) -> torch.FloatTensor:
        """Reweight logits using shuffle and alpha"""
        unshuffle = torch.argsort(shuffle, dim=-1)
        
        s_p_logits = torch.gather(p_logits, -1, shuffle)
        s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)
        
        s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
        s_cumsum = torch.exp(s_log_cumsum)
        s_p = F.softmax(s_p_logits, dim=-1)

        # Calculate boundary 1
        boundary_1 = torch.argmax((s_cumsum > alpha).to(torch.int), dim=-1, keepdim=True)
        p_boundary_1 = torch.gather(s_p, -1, boundary_1)
        portion_in_right_1 = (torch.gather(s_cumsum, -1, boundary_1) - alpha) / p_boundary_1
        portion_in_right_1 = torch.clamp(portion_in_right_1, 0, 1)
        s_all_portion_in_right_1 = (s_cumsum > alpha).type_as(p_logits)
        s_all_portion_in_right_1.scatter_(-1, boundary_1, portion_in_right_1)

        # Calculate boundary 2
        boundary_2 = torch.argmax((s_cumsum > (1 - alpha)).to(torch.int), dim=-1, keepdim=True)
        p_boundary_2 = torch.gather(s_p, -1, boundary_2)
        portion_in_right_2 = (torch.gather(s_cumsum, -1, boundary_2) - (1 - alpha)) / p_boundary_2
        portion_in_right_2 = torch.clamp(portion_in_right_2, 0, 1)
        s_all_portion_in_right_2 = (s_cumsum > (1 - alpha)).type_as(p_logits)
        s_all_portion_in_right_2.scatter_(-1, boundary_2, portion_in_right_2)

        s_all_portion_in_right = s_all_portion_in_right_2 / 2 + s_all_portion_in_right_1 / 2
        s_shift_logits = torch.log(s_all_portion_in_right)
        shift_logits = torch.gather(s_shift_logits, -1, unshuffle)

        return p_logits + shift_logits

    def _apply_watermark(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        alpha: float,
        prefix_length: int
    ) -> torch.FloatTensor:
        """Apply watermark to scores"""
        seeds = self._get_seeds_for_cipher(input_ids, prefix_length)
        rng = [torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds]
        shuffle = self._from_random(rng, scores.size(1))
        reweighted_scores = self._reweight_logits(shuffle, scores, alpha)
        return reweighted_scores

       
    def sample(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """DIP sampling implementation"""
        if input_ids.shape[-1] < self.prefix_length:
            return logits

        # Bias logits
        reweighted_scores = self._apply_watermark(
            input_ids,
            logits,
            alpha=self.alpha,
            prefix_length=self.prefix_length
        )

        # Get probability distribution and sample
        probs = self._sampling(
            reweighted_scores,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p
        )
        sampled_indices = torch.multinomial(probs, num_samples=1)

        return sampled_indices

    
    def _get_result_filename(self, prompt_idx: int) -> str:
        """Get filename for saving results"""
        return (
            f"{self.result_dir}/dip-{self.prob_type}-p{prompt_idx+1}-"
            f"{self.model_name}-temp-{self.temperature}-"
            f"topk-{self.top_k}-topp-{self.top_p}-"
            f"alpha-{self.alpha}-"
            f"prefixlen-{self.prefix_length}-"
            f"{self.samples}-iter-{self.sample_iter}.json"
        )
        