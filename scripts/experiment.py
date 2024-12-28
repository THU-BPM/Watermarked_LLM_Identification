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

import ujson as json
import numpy as np
import itertools
import math
import csv
import multiprocessing as mps
from functools import partial
import math
import os
import mpmath as mp
# from utils import (
#     process_pair, 
#     cosine_similarity, 
#     update_shared_memory,
#     shared_t1_wm,
#     shared_t2_wm
# )
from tqdm import tqdm
from scipy.stats import rankdata
mp.dps = 1000

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 and norm_b == 0:
        return 1
    elif norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

def avg_cossim_tuple(t12_diff):
    similarities = [cosine_similarity(v1, v2) for v1, v2 in t12_diff.values()]
    return np.mean(similarities)

t1_wm = None
t2_wm = None
# Define the function to process a single pair
def process_pair(value):
    
    key1 = value[0]
    key2 = value[1]

    top_n = 50
    t1_words = {}
    t2_words = {}

    # Sort the pair and get the top_n
    sorted_index_t1_key1 = np.argsort(t1_wm[key1])[::-1]
    t1_words[key1] = sorted_index_t1_key1[:top_n]
    sorted_index_t2_key1 = np.argsort(t2_wm[key1])[::-1]
    t2_words[key1] = sorted_index_t2_key1[:top_n]

    sorted_index_t1_key2 = np.argsort(t1_wm[key2])[::-1]
    t1_words[key2] = sorted_index_t1_key2[:top_n]

    sorted_index_t2_key2 = np.argsort(t2_wm[key2])[::-1]
    t2_words[key2] = sorted_index_t2_key2[:top_n]

    # Get the intersection
    # (t11 ∪ t12) ∩ (t21 ∪ t22)
    word_set = (
        set(t1_words[key1]).union(set(t1_words[key2]))
    ).intersection(set(t2_words[key1]).union(set(t2_words[key2])))

    if word_set == set():
        zero_cnt += 1
        return (None, None)  # The intersection is empty, return None

    # Normalize the probabilities
    t11_prob = t1_wm[key1] / np.sum(t1_wm[key1])
    t12_prob = t1_wm[key2] / np.sum(t1_wm[key2])
    t21_prob = t2_wm[key1] / np.sum(t2_wm[key1])
    t22_prob = t2_wm[key2] / np.sum(t2_wm[key2])

    # use rankdata to get the relative rankings
    t11_rank = rankdata(t11_prob, method="min")
    t12_rank = rankdata(t12_prob, method="min")
    t21_rank = rankdata(t21_prob, method="min")
    t22_rank = rankdata(t22_prob, method="min")

    t11_rank_word_set = [t11_rank[i] for i in word_set]
    t12_rank_word_set = [t12_rank[i] for i in word_set]
    t21_rank_word_set = [t21_rank[i] for i in word_set]
    t22_rank_word_set = [t22_rank[i] for i in word_set]

    t11_t12_rank_diff = [
        (
            1
            if t11_rank_word_set[i] > t12_rank_word_set[i]
            else -1 if t11_rank_word_set[i] < t12_rank_word_set[i] else 0
        )
        for i in range(len(t11_rank_word_set))
    ]
    t21_t22_rank_diff = [
        (
            1
            if t21_rank_word_set[i] > t22_rank_word_set[i]
            else -1 if t21_rank_word_set[i] < t22_rank_word_set[i] else 0
        )
        for i in range(len(t21_rank_word_set))
    ]

    # Return the calculation results
    return (key1, key2), (t11_t12_rank_diff, t21_t22_rank_diff)

def run(
    model,
    num_samples,
    filter_threshold,
    combinations,
    experiment_type,
    **kwargs
):
    global t1_wm, t2_wm
    # Set up paths based on experiment type
    if experiment_type == "kth-prob1":
        prefix_path = "../data/results/csv/KTH-prob1-results"
        file_prefix = "kth"
        prob = "prob1"
        data_path = "../data/results/prob1"
    elif experiment_type == "kth-prob2":
        prefix_path = "../data/results/csv/KTH-prob2-results" 
        file_prefix = "kth"
        prob = "prob2"
        data_path = "../data/results/prob2"
    elif experiment_type == "kth-prob2_5gram":
        prefix_path = "../data/results/csv/KTH-prob2_5gram-results"
        file_prefix = "kth"
        prob = "prob2_5gram"
        data_path = "../data/results/prob2"
    elif experiment_type == "aar-prob1":
        prefix_path = "../data/results/csv/Aar-prob1-results"
        file_prefix = "aar"
        prob = "prob1"
        data_path = "../data/results/prob1"  
    elif experiment_type == "aar-prob2":
        prefix_path = "../data/results/csv/Aar-prob2-results"
        file_prefix = "aar"
        prob = "prob2"
        data_path = "../data/results/prob2"
    elif experiment_type == "dip-prob1":
        prefix_path = "../data/results/csv/DIP-prob1-results"
        file_prefix = "dip"
        prob = "prob1"
        data_path = "../data/results/prob1"
    elif experiment_type == "dip-prob2":
        prefix_path = "../data/results/csv/DIP-prob2-results"
        file_prefix = "dip"
        prob = "prob2"
        data_path = "../data/results/prob2"
    elif experiment_type == "kgw-prob1":
        prefix_path = "../data/results/csv/KGW-prob1-results"
        file_prefix = "kgw"
        prob = "prob1"
        data_path = "../data/results/prob1"
    elif experiment_type == "kgw-prob2":
        prefix_path = "../data/results/csv/KGW-prob2-results"
        file_prefix = "kgw"
        prob = "prob2"
        data_path = "../data/results/prob2"
    elif experiment_type == "its-prob1":
        prefix_path = "../data/results/csv/ITS-prob1-results"
        file_prefix = "its"
        prob = "prob1"
        data_path = "../data/results/prob1"
    elif experiment_type == "its-prob2":
        prefix_path = "../data/results/csv/ITS-prob2-results"
        file_prefix = "its"
        prob = "prob2"
        data_path = "../data/results/prob2"
    elif experiment_type == "unwatermarked-prob1":
        prefix_path = "../data/results/csv/unwatermarked-prob1-results"
        file_prefix = "unwatermarked"
        prob = "prob1"
        data_path = "../data/results/prob1"
    elif experiment_type == "unwatermarked-prob2":
        prefix_path = "../data/results/csv/unwatermarked-prob2-results"
        file_prefix = "unwatermarked"
        prob = "prob2"
        data_path = "../data/results/prob2"
    elif experiment_type == "unwatermarked-prob2_5gram":
        prefix_path = "../data/results/csv/unwatermarked-prob2_5gram-results"
        file_prefix = "unwatermarked"
        prob = "prob2_5gram"
        data_path = "../data/results/prob2"
    elif experiment_type == "unbiased-prob1":# unbiased for DIP(alpha=0.5)
        prefix_path = "../data/results/csv/unbiased-prob1-results"
        file_prefix = "dip"
        prob = "prob1"
        data_path = "../data/results/prob1"
    elif experiment_type == "unbiased-prob2": # unbiased for DIP(alpha=0.5)
        prefix_path = "../data/results/csv/unbiased-prob2-results"
        file_prefix = "dip"
        prob = "prob2"
        data_path = "../data/results/prob2"
    elif experiment_type == "waterbag-prob1":
        prefix_path = "../data/results/csv/waterbag-prob1-results"
        file_prefix = "waterbag"
        prob = "prob1"
        data_path = "../data/results/prob1"
    elif experiment_type == "waterbag-prob2":
        prefix_path = "../data/results/csv/waterbag-prob2-results"
        file_prefix = "waterbag"
        prob = "prob2"
        data_path = "../data/results/prob2"
    elif experiment_type == "waterbag-prob2_5gram":
        prefix_path = "../data/results/csv/waterbag-prob2_5gram-results"
        file_prefix = "waterbag"
        prob = "prob2_5gram"
        data_path = "../data/results/prob2"
    # Add other experiment types...

    for combo in combinations:
        temp, topp, topk = combo["temperature"], combo["topp"], combo["topk"]
        cossim_results = []

        for iter in range(3):
            print(f"Iteration: {iter}")

            # Construct file paths based on experiment type
            if experiment_type in ["kth-prob1", "kth-prob2", "kth-prob2_5gram"]:
                p1_file = f"{data_path}/{experiment_type}-p1-{model}-temp-{temp}-keylen-{kwargs.get('keylen')}-topk-{topk}-topp-{topp}-{num_samples}-iter-{iter}.json"
                p2_file = f"{data_path}/{experiment_type}-p2-{model}-temp-{temp}-keylen-{kwargs.get('keylen')}-topk-{topk}-topp-{topp}-{num_samples}-iter-{iter}.json"
            elif experiment_type in ["aar-prob1", "aar-prob2"]:
                p1_file = f"{data_path}/{experiment_type}-p1-{model}-temp-{temp}-topk-{topk}-topp-{topp}-prefixlen-{kwargs.get('prefix_length')}-{num_samples}-iter-{iter}.json"
                p2_file = f"{data_path}/{experiment_type}-p2-{model}-temp-{temp}-topk-{topk}-topp-{topp}-prefixlen-{kwargs.get('prefix_length')}-{num_samples}-iter-{iter}.json"
            elif experiment_type in ["dip-prob1", "dip-prob2"]:
                p1_file = f"{data_path}/{experiment_type}-p1-{model}-temp-{temp}-topk-{topk}-topp-{topp}-alpha-{kwargs.get('alpha')}-prefixlen-{kwargs.get('prefix_length')}-{num_samples}-iter-{iter}.json"
                p2_file = f"{data_path}/{experiment_type}-p2-{model}-temp-{temp}-topk-{topk}-topp-{topp}-alpha-{kwargs.get('alpha')}-prefixlen-{kwargs.get('prefix_length')}-{num_samples}-iter-{iter}.json"
            elif experiment_type in ["kgw-prob1", "kgw-prob2", "kgw-prob2_5gram"]:
                p1_file = f"{data_path}/{experiment_type}-p1-{model}-{kwargs.get('scheme')}-temp-{temp}-topk-{topk}-topp-{topp}-gamma-{kwargs.get('gamma')}-delta-{kwargs.get('delta')}-prefixlen-{kwargs.get('prefix_length')}-{num_samples}-iter-{iter}.json"
                p2_file = f"{data_path}/{experiment_type}-p2-{model}-{kwargs.get('scheme')}-temp-{temp}-topk-{topk}-topp-{topp}-gamma-{kwargs.get('gamma')}-delta-{kwargs.get('delta')}-prefixlen-{kwargs.get('prefix_length')}-{num_samples}-iter-{iter}.json"
            elif experiment_type in ["its-prob1", "its-prob2"]:
                p1_file = f"{data_path}/{experiment_type}-p1-{model}-temp-{temp}-keylen-{kwargs.get('keylen')}-topk-{topk}-topp-{topp}-{num_samples}-iter-{iter}.json"
                p2_file = f"{data_path}/{experiment_type}-p2-{model}-temp-{temp}-keylen-{kwargs.get('keylen')}-topk-{topk}-topp-{topp}-{num_samples}-iter-{iter}.json"
            elif experiment_type in ["unwatermarked-prob1", "unwatermarked-prob2", "unwatermarked-prob2_5gram"]:
                p1_file = f"{data_path}/{experiment_type}-p1-{model}-temp-{temp}-topk-{topk}-topp-{topp}-{num_samples}-iter-{iter}.json"
                p2_file = f"{data_path}/{experiment_type}-p2-{model}-temp-{temp}-topk-{topk}-topp-{topp}-{num_samples}-iter-{iter}.json"
            elif experiment_type in ["unbiased-prob1", "unbiased-prob2"]:
                p1_file = f"{data_path}/{experiment_type.replace('unbiased', 'dip')}-p1-{model}-temp-{temp}-topk-{topk}-topp-{topp}-alpha-0.5-{num_samples}-iter-{iter}.json"
                p2_file = f"{data_path}/{experiment_type.replace('unbiased', 'dip')}-p2-{model}-temp-{temp}-topk-{topk}-topp-{topp}-alpha-0.5-{num_samples}-iter-{iter}.json"
            elif experiment_type in ["waterbag-prob1", "waterbag-prob2", "waterbag-prob2_5gram"]:
                p1_file = f"{data_path}/{experiment_type}-p1-{model}-{kwargs.get('scheme')}-temp-{temp}-keylen-{kwargs.get('keylen')}-topk-{topk}-topp-{topp}-gamma-{kwargs.get('gamma')}-delta-{kwargs.get('delta')}-prefixlen-{kwargs.get('prefix_length')}-{num_samples}-iter-{iter}.json"
                p2_file = f"{data_path}/{experiment_type}-p2-{model}-{kwargs.get('scheme')}-temp-{temp}-keylen-{kwargs.get('keylen')}-topk-{topk}-topp-{topp}-gamma-{kwargs.get('gamma')}-delta-{kwargs.get('delta')}-prefixlen-{kwargs.get('prefix_length')}-{num_samples}-iter-{iter}.json"

            print("Loading samples...")
            # import ipdb; ipdb.set_trace()
            with open(p1_file, "r") as f:
                data_p1 = json.load(f)
            with open(p2_file, "r") as f:
                data_p2 = json.load(f)

            print("Processing samples...")
            # Process watermark data based on experiment type
            
            t1_wm = {
                key: np.array(data_p1[key]["S_wm"])
                for key in data_p1.keys()
            }
            t2_wm = {
                key: np.array(data_p2[key]["S_wm"])
                for key in data_p2.keys()
            }

            # Filter
            t1_wm = {
                key: t1_wm[key]
                for key in t1_wm.keys()
                if np.sum(t1_wm[key]) >= filter_threshold
            }
            t2_wm = {
                key: t2_wm[key]
                for key in t2_wm.keys()
                if np.sum(t2_wm[key]) >= filter_threshold
            }

            wm_common_keys = set(t1_wm.keys()).intersection(set(t2_wm.keys()))
            t1_wm = {key: t1_wm[key] for key in wm_common_keys}
            t2_wm = {key: t2_wm[key] for key in wm_common_keys}
            
            wm_common_pairs = list(itertools.combinations(wm_common_keys, 2))
            print(f"Length of common pairs: {len(wm_common_pairs)}")
            t12_wm_diff = {}

            print("Calculating differences...")
            with mps.Pool(processes=42) as pool:
                results = list(tqdm(pool.map(process_pair, wm_common_pairs), total=len(wm_common_pairs)))

            # Process results
            for result in results:
                if result is not None:
                    key_pair, diff_data = result
                    t12_wm_diff[key_pair] = diff_data

            # Calculate metrics
            avg_cossim_wm = avg_cossim_tuple(t12_wm_diff)
            cossim_results.append(avg_cossim_wm)
            print(f"Average cosine similarity: {avg_cossim_wm}")

        # Calculate statistics
        cossim_results = np.array(cossim_results)
        cossim_mean = np.mean(cossim_results)
        cossim_std = np.std(cossim_results, ddof=1)
        z_score = (cossim_mean - 0.1) / cossim_std
        z_mpf = mp.mpf(z_score)
        p_value = mp.erfc(z_mpf / mp.sqrt(2)) / 2
        p_value_str = mp.nstr(p_value, 50, strip_zeros=False)

        # Save results
        output_path = construct_output_path(
            prefix_path=prefix_path,
            experiment_type=experiment_type,
            model=model,
            temp=temp,
            num_samples=num_samples,
            **kwargs
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        save_results(
            output_path=output_path,
            model=model,
            temp=temp,
            topk=topk,
            topp=topp,
            pair_count=len(wm_common_pairs),
            cossim_results=cossim_results,
            cossim_mean=cossim_mean,
            cossim_std=cossim_std,
            z_score=z_score,
            p_value_str=p_value_str
        )

def construct_output_path(prefix_path, experiment_type, model, temp, num_samples, **kwargs):
    """Construct output path based on experiment type and parameters"""
    if experiment_type in ["its-prob1", "its-prob2", "kth-prob1", "kth-prob2", "kth-prob2_5gram"]:
        return f"{prefix_path}/{experiment_type}-{model}-{num_samples}-keylen-{kwargs['keylen']}-temp-{temp}-topk-{kwargs['topk']}-topp-{kwargs['topp']}-{kwargs.get('threshold', '')}-{kwargs.get('diff', 'rank')}.csv"
    elif experiment_type in ["aar-prob1", "aar-prob2"]:
        return f"{prefix_path}/{experiment_type}-{model}-{num_samples}-prefixlen-{kwargs['prefix_length']}-temp-{temp}-topk-{kwargs['topk']}-topp-{kwargs['topp']}-{kwargs.get('threshold', '')}-{kwargs.get('diff', 'rank')}.csv"
    elif experiment_type in ["dip-prob1", "dip-prob2"]:
        return f"{prefix_path}/{experiment_type}-{model}-{num_samples}-alpha-{kwargs['alpha']}-prefixlen-{kwargs['prefix_length']}-temp-{temp}-topk-{kwargs['topk']}-topp-{kwargs['topp']}-{kwargs.get('threshold', '')}-{kwargs.get('diff', 'rank')}.csv"
    elif experiment_type in ["kgw-prob1", "kgw-prob2", "kgw-prob2_5gram"]:
        return f"{prefix_path}/{experiment_type}-{model}-{num_samples}-{kwargs.get('scheme', '')}-prefixlen-{kwargs['prefix_length']}-gamma-{kwargs['gamma']}-delta-{kwargs['delta']}-temp-{temp}-topk-{kwargs['topk']}-topp-{kwargs['topp']}-{kwargs.get('threshold', '')}-{kwargs.get('diff', 'rank')}.csv"
    elif experiment_type in ["waterbag-prob1", "waterbag-prob2", "waterbag-prob2_5gram"]:
        return f"{prefix_path}/{experiment_type}-{model}-{num_samples}-{kwargs.get('scheme', '')}-keylen-{kwargs['keylen']}-prefixlen-{kwargs['prefix_length']}-gamma-{kwargs['gamma']}-delta-{kwargs['delta']}-temp-{temp}-topk-{kwargs['topk']}-topp-{kwargs['topp']}-{kwargs.get('threshold', '')}-{kwargs.get('diff', 'rank')}.csv"
    elif experiment_type in ["unwatermarked-prob1", "unwatermarked-prob2"]:
        return f"{prefix_path}/{experiment_type}-{model}-{num_samples}-temp-{temp}-topk-{kwargs['topk']}-topp-{kwargs['topp']}-{kwargs.get('threshold', '')}-{kwargs.get('diff', 'rank')}.csv"
    elif experiment_type in ["unbiased-prob1", "unbiased-prob2"]:
        return f"{prefix_path}/{experiment_type}-{model}-{num_samples}-prefixlen-{kwargs['prefix_length']}-temp-{temp}-topk-{kwargs['topk']}-topp-{kwargs['topp']}-{kwargs.get('threshold', '')}-{kwargs.get('diff', 'rank')}.csv"
    
def save_results(output_path, **kwargs):
    """Save results to CSV file"""
    csv_header = [
        "model_name",
        "temperature",
        "topk",
        "topp", 
        "pair_count",
        "cossim_wm",
        "avg_cossim_wm",
        "std_cossim_wm", 
        "z_score_wm",
        "p_value_wm",
    ]
    
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        
        for idx in range(3):
            writer.writerow([
                kwargs["model"],
                kwargs["temp"],
                kwargs["topk"],
                kwargs["topp"],
                kwargs["pair_count"],
                kwargs["cossim_results"][idx],
                kwargs["cossim_mean"],
                kwargs["cossim_std"],
                kwargs["z_score"],
                kwargs["p_value_str"],
            ])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--samples", type=int, help="Number of samples")
    parser.add_argument("--threshold", type=float, help="Filter threshold")
    
    parser.add_argument("--combinations", type=str, help="Parameter combinations")
    parser.add_argument("--experiment_type", type=str, help="Experiment type")
    parser.add_argument("--diff", default="rank", type=str, help="Difference type")
    parser.add_argument("--prefix_length", default=4, type=int, help="Prefix length")
    # kgw config
    parser.add_argument("--topk", default=0, type=int, help="Top k")
    parser.add_argument("--topp", default=1.0, type=float, help="Top p")
    parser.add_argument("--gamma", default=0.5, type=float, help="Gamma")
    parser.add_argument("--delta", default=2.0, type=float, help="Delta")
    parser.add_argument("--scheme", default="time", type=str, help="Scheme")
    
    # ITS & KTH config
    parser.add_argument("--keylen", type=int, default=420, help="Key length")
    
    # DIP config
    parser.add_argument("--alpha", default=0.45, type=float, help="Alpha") # 0.5 for unbiased
    # parser.add_argument("--fill_length", type=int, help="Fill length")
    # Add other arguments as needed

    args = parser.parse_args()
    
    if args.combinations == "temp":
        combinations = [
            {"temperature": t, "topp": 1.0, "topk": 0}
            for t in [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        ]
    elif args.combinations == "experiment":
        combinations = [
            {"temperature": 1.0, "topp": 1.0, "topk": 0},
        ]

    run(
        model=args.model_name,
        num_samples=args.samples,
        filter_threshold=args.threshold,
        combinations=combinations,
        experiment_type=args.experiment_type,
        keylen=args.keylen,
        diff=args.diff,
        prefix_length=args.prefix_length,
        topk=args.topk,
        topp=args.topp,
        gamma=args.gamma,
        delta=args.delta,
        scheme=args.scheme,
        alpha=args.alpha,
        # fill_length=args.fill_length
    ) 
