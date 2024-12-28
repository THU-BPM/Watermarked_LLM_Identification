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

import openai
from itertools import groupby
from collections import Counter
import itertools
import numpy as np
from openai import OpenAI
import os
import ujson as json
import argparse
from requests.exceptions import RequestException, Timeout
import time
import re
from tqdm import tqdm

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

os.environ["OPENAI_API_KEY"] = "your_api_key"
os.environ["GEMINI_API_KEY"] = "your_api_key"


def generate_text(system_prompt, user_prompt, model_name, temperature=1.0, max_tokens=8, n=128):
    attempt = 0
    retries = 5
    retry_delay = 5
    while attempt < retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            return [choice.message.content for choice in response.choices]
        
        except (Timeout, RequestException) as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Request failed.")
                return None

def generate_text_gemini(system_prompt, user_prompt, model_name, temperature=1.0, max_tokens=12, n=128):
    # Create the model object
    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
    
    texts = []
    # Generate responses and process them
    for _ in tqdm(range(n)):
        # handle generation refused
        
        response = model.generate_content(user_prompt,
                                        safety_settings={HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                            })
        text = response.text
        texts.append(text)
    
    return texts

def run_experiment(prompt_type, model_name, sample_size=32):
    with open(f"../../data/prompts/api-p{prompt_type}.txt", "r") as f:
        user_prompt = f.read()
        
    system_prompt="""You should only generate the prefix and the number, no other words. The prefix and the number should be selected as randomly as possible."""
    
    generated_texts = []
    for _ in tqdm(range(sample_size)):
        if args.api_key == "OpenAI":
            generated_text = generate_text(system_prompt, user_prompt, model_name, n=128)
        else:
            generated_text = generate_text_gemini(system_prompt, user_prompt, model_name, n=128)
        generated_texts.extend(generated_text)
    prefix_numbers = {}
    lost_texts = []
    
    # Match the format of the context
    pattern_response = re.compile(r"[A-Z]-[0-9]-[a-z]\d")
    
    for text in generated_texts:
        try:
            match = pattern_response.match(text)
            if match:
                match_str = match.group(0)

                prefix = match_str[:-1]
                number = int(match_str[-1])
                
                if prefix not in prefix_numbers:
                    prefix_numbers[prefix] = Counter()

                # Store the count of the number
                prefix_numbers[prefix][number] += 1

        except ValueError:
            lost_texts.append(text)
            continue
    print(len(prefix_numbers.keys()))
    print("Lost texts:", len(lost_texts))

    return prefix_numbers


def calculate_distribution_difference(dist1, dist2):

    dist1 = {k: v for k, v in dist1.items()}
    dist2 = {k: v for k, v in dist2.items()}

    common_keys = set(dist1.keys()) | set(dist2.keys())
    rank_diff = {}

    sorted_dist1 = sorted(dist1.items(), key=lambda item: item[1], reverse=True)
    sorted_dist2 = sorted(dist2.items(), key=lambda item: item[1], reverse=True)

    # Calculate the rank of each number
    rank_dist1 = {k: rank for rank, (k, v) in enumerate(sorted_dist1, 1)}
    rank_dist2 = {k: rank for rank, (k, v) in enumerate(sorted_dist2, 1)}

    for key in common_keys:
        if key not in rank_dist1.keys():
            rank_dist1[key] = -1
        elif key not in rank_dist2.keys():
            rank_dist2[key] = -1
        rank_diff[key] = (
            1
            if rank_dist1[key] > rank_dist2[key]
            else -1 if rank_dist1[key] < rank_dist2[key] else 0
        )

    return rank_diff


def main(model_name, output_path, sample_size=32, threshold=5, common_keys_threshold=2):
    prompt_types = ["1", "2"]
    distributions = {
        prompt_type: run_experiment(prompt_type, model_name, sample_size) for prompt_type in prompt_types
    }

    # Calculate differences and similarities
    differences = []
    similarity_details = []
    
    common_prefix = set(distributions["1"].keys()).intersection(
        set(distributions["2"].keys())
    )

    for prefix1, prefix2 in itertools.combinations(common_prefix, 2):
        # Filter
        t11_sum = sum(distributions["1"][prefix1].values())
        t12_sum = sum(distributions["1"][prefix2].values())
        t21_sum = sum(distributions["2"][prefix1].values())
        t22_sum = sum(distributions["2"][prefix2].values())

        filter_threshold = threshold
        if (
            t11_sum < filter_threshold
            or t21_sum < filter_threshold
            or t12_sum < filter_threshold
            or t22_sum < filter_threshold
        ):
            continue

        dist1_diff = calculate_distribution_difference(
            distributions["1"][prefix1], distributions["1"][prefix2]
        )
        dist2_diff = calculate_distribution_difference(
            distributions["2"][prefix1], distributions["2"][prefix2]
        )

        # Find common keys
        common_keys = set(dist1_diff.keys()) & set(dist2_diff.keys())

        if len(common_keys) <= common_keys_threshold:
            continue
            
        dist1_diff_list = [dist1_diff[key] for key in common_keys]
        dist2_diff_list = [dist2_diff[key] for key in common_keys]

        dot_product = sum(d1 * d2 for d1, d2 in zip(dist1_diff_list, dist2_diff_list))

        dist1_magnitude = sum(d1 * d1 for d1 in dist1_diff_list) ** 0.5
        dist2_magnitude = sum(d2 * d2 for d2 in dist2_diff_list) ** 0.5
        if dist1_magnitude == 0 or dist2_magnitude == 0:
            continue
        similarity = dot_product / (dist1_magnitude * dist2_magnitude)
        differences.append(similarity)
        similarity_details.append({
            "prompt_type1": "1",
            "prompt_type2": "2",
            "prefix1": prefix1,
            "prefix2": prefix2,
            "similarity": similarity
        })

    average_similarity = np.mean(differences)

    print(f"Average similarity across all prefix pairs: {average_similarity:.4f}")
    print(f"Number of prefix pairs: {len(differences)}")

    results = {
        "model_name": model_name,
        "average_similarity": average_similarity,
        "similarities": differences,
        "similarity_details": similarity_details,
        "distributions": distributions
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with different LLM models")
    parser.add_argument("--api_key", type=str, default="OpenAI", help="Gemini or OpenAI")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model name to use")
    parser.add_argument("--sample_size", type=int, default=160, help="Number of samples to generate")
    parser.add_argument("--threshold", type=int, default=5, help="Threshold for the distribution")
    parser.add_argument("--common_keys_threshold", type=int, default=2, help="Threshold for the number of common keys")
    parser.add_argument("--iter", type=int, default=0, help="Number of iterations to run")
    
    args = parser.parse_args()
    
    if args.api_key == "OpenAI":
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    elif args.api_key == "Gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        # Configure the API key for this thread
        genai.configure(api_key=api_key.strip())
    else:
        raise ValueError("Invalid API Name")

    output_path = f"results/{args.model}_chat_prob2_experiment_{args.sample_size}-{args.iter}.json"

    main(args.model, output_path, args.sample_size, args.threshold, args.common_keys_threshold)
    
    
