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

from .aar_sampler import AarSampler
from .dip_sampler import DIPSampler
from .kgw_sampler import KGWsampler
from .kth_sampler import KTHsampler
from .its_sampler import ITSSampler
from .base_sampler import BaseSampler
from .waterbag_sampler import WaterbagSampler

SAMPLER_REGISTRY = {
    "aar": AarSampler,
    "dip": DIPSampler,
    "kgw": KGWsampler,
    "kth": KTHsampler,
    "its": ITSSampler,
    "unwatermarked": BaseSampler,
    "waterbag": WaterbagSampler
}

def get_sampler(sampler_type: str):
    """Get sampler class by name"""
    sampler_type = sampler_type.lower()
    if sampler_type not in SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown sampler type: {sampler_type}. "
            f"Available samplers: {list(SAMPLER_REGISTRY.keys())}"
        )
    return SAMPLER_REGISTRY[sampler_type] 