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
from itertools import product

class PromptManager:
    def __init__(self, prompt_dir: str, prompt_type: str):
        """
        Manage prompts and dynamic combinations.

        Args:
        - prompt_dir (str): Directory containing prompt files.
        - combinations_rules (list[list[str]]): List of lists defining combination elements.
        """
        self.prompt_dir = prompt_dir
        self.prompt_type = prompt_type

    def load_prompts(self) -> list[str]:
        """
        Load prompts based on the probability type (e.g., ngram, fixkey, 5gram).

        Args:
        - prob_type (str): Type of prompt to load.

        Returns:
        - list[str]: List of loaded prompts.
        """
        prompts = []
        for i in range(2):
            with open(f"{self.prompt_dir}/{self.prompt_type}-p{i+1}.txt", "r") as f:
                prompts.append("".join(f.readlines()))
        return prompts

    def generate_combinations(self, combinations_rules: list[list[str]]) -> list[str]:
        """
        Generate combinations based on the provided rules.

        Args:
        - combinations_rules (list[list[str]]): List of lists defining combination elements.

        Returns:
        - list[str]: List of generated combinations.
        """
        return ["".join(comb) for comb in product(*combinations_rules)]