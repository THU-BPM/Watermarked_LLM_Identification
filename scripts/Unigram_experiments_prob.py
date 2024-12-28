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

import json
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
from scipy.spatial.distance import cosine
import hashlib
import argparse
import os
import math
instruction="generate a random number sequence between 0 to 9"
prompt_list=[f"{instruction}:2,3,9,8,0,4,7,5,6,1,5,8,7,1,2,4,6,0,9,3,1,3,7,9,2,0,4,6,8,5,0,7,2,3,6,5,1,9,8,4,3,8,0,7,4,5,6,1,9,2,",
            f"{instruction}:3,8,0,7,4,5,6,1,9,2,1,3,7,9,2,0,4,6,8,5,0,7,2,3,6,5,1,9,8,4,2,3,9,8,0,4,7,5,6,1,5,8,7,1,2,4,6,0,9,3,",
            f"{instruction}:5,8,7,1,2,4,6,0,9,3,2,3,9,8,0,4,7,5,6,1,3,8,0,7,4,5,6,1,9,2,1,3,7,9,2,0,4,6,8,5,0,7,2,3,6,5,1,9,8,4,",
            f"{instruction}:0,7,2,3,6,5,1,9,8,4,3,8,0,7,4,5,6,1,9,2,5,8,7,1,2,4,6,0,9,3,2,3,9,8,0,4,7,5,6,1,1,3,7,9,2,0,4,6,8,5,",
            f"{instruction}:1,3,7,9,2,0,4,6,8,5,0,7,2,3,6,5,1,9,8,4,3,8,0,7,4,5,6,1,9,2,5,8,7,1,2,4,6,0,9,3,2,3,9,8,0,4,7,5,6,1,",
            f"{instruction}:6,0,9,3,2,3,9,8,0,4,7,5,6,1,5,8,7,1,2,4,3,8,0,7,4,5,6,1,9,2,1,3,7,9,2,0,4,6,8,5,0,7,2,3,6,5,1,8,9,7,",
            f"{instruction}:4,6,0,9,3,2,3,9,8,0,4,7,5,6,1,5,8,7,1,2,8,0,7,4,5,6,1,9,2,1,3,7,9,2,0,4,6,8,5,0,7,2,3,6,5,1,9,8,4,0,",
            f"{instruction}:9,8,0,4,7,5,6,1,5,8,7,1,2,4,6,0,9,3,1,3,7,9,2,0,4,6,8,5,0,7,2,3,6,5,1,9,8,4,3,8,0,7,4,5,6,1,9,2,2,6,",
            f"{instruction}:7,4,5,6,1,9,2,1,3,7,9,2,0,4,6,8,5,0,7,2,3,6,5,1,9,8,4,2,3,9,8,0,4,7,5,6,1,5,8,7,1,2,4,6,0,9,3,0,7,8,",
            f"{instruction}:8,0,7,4,5,6,1,9,2,5,8,7,1,2,4,6,0,9,3,2,3,9,8,0,4,7,5,6,1,3,8,0,7,4,5,6,1,9,2,1,3,7,9,2,0,4,6,8,5,9,",]


with open("config.json", "r") as f:
    args = json.load(f)
    print(args)
device = torch.device(args['device'])
model_path=args['model_path']
model_name=args['model_name']
threhold=args['threshold']
seed=args['hash_key']
experiments_num=args['experiments_num']
is_logits=args['is_logits']
is_proxy=args['is_proxy']
proxy_model_name=args['proxy_model']
proxy_model_path=args['proxy_model_path']
print("model_name",model_name)
print("is_logits",is_logits)
print("is_proxy",is_proxy)
unknown_tokenizer = AutoTokenizer.from_pretrained(model_path,device_map="auto")
unknown_model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto")
# the average probs distribution of the proxy model, as a prior assumption
prior_list=[[0.18938708937458346, 0.030517441315072096, 0.019496393154328585, 0.044161874063880155, 0.2891853492559508, 0.017907514345067547, 0.1542457589160913, 0.07018404295822794, 0.147266006071896, 0.03764853054490208], [0.10498953383402776, 0.020192089582626, 0.023582359444029442, 0.013867892043141775, 0.3075501796199932, 0.02601525274086861, 0.15121513670742703, 0.12817285463968214, 0.1872367125104615, 0.03717798887774259], [0.1158797776308897, 0.033157543304195396, 0.015170074406029322, 0.01838942629673829, 0.2602103399206404, 0.062116306901325045, 0.18407403383307075, 0.11789837887073847, 0.13962161283798288, 0.05348250599838961], [0.14350846561822858, 0.07411924964892419, 0.014783779799850008, 0.014505930341229654, 0.2540713470853365, 0.007546022021308364, 0.19647720323032142, 0.10456381436532548, 0.15464017964127777, 0.035784008248197996], [0.11606948505752349, 0.014542347003595489, 0.049065898022646576, 0.014776053110083447, 0.258052547780447, 0.011437092220449143, 0.169546315911121, 0.07208526914100143, 0.1644335267816976, 0.12999146497143482], [0.14092755354468464, 0.11580391708466543, 0.045839154659189994, 0.017903113849874055, 0.2671311492764922, 0.009780662000690386, 0.15385026519585307, 0.06449563386313958, 0.1472593254674706, 0.03700922505794003], [0.100660603446859, 0.02100077057529006, 0.015235614211298355, 0.022925001361368966, 0.2882575191143009, 0.025152878523031105, 0.1820078329494223, 0.13424319579035635, 0.15326450222299987, 0.05725208180507305], [0.1400446984478074, 0.030761844155142597, 0.010997761249576318, 0.028525785943890547, 0.27967196870920175, 0.06846598943430664, 0.17577106977102178, 0.09262272891367469, 0.13573409136278272, 0.037404062012595496], [0.11858080423859618, 0.045065081960492054, 0.02595284299315552, 0.02287346885799762, 0.3006092037469671, 0.06415617675806848, 0.1621093611960871, 0.055329405322593936, 0.16472904979263744, 0.040594605133404474], [0.1530485175472553, 0.02753348802234248, 0.05012966986405899, 0.02354926957895103, 0.26798471422958137, 0.010533470994155155, 0.15203187105844052, 0.09662900918103412, 0.18330384567918376, 0.03525614384499723]]
if is_proxy:
        proxy_tokenizer = AutoTokenizer.from_pretrained(proxy_model_path,device_map="auto")
        proxy_model = AutoModelForCausalLM.from_pretrained(proxy_model_path,device_map="auto")
def hash_fn(x: int) -> int:
        """hash function to generate random seed, solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')
# calculate the rank indices of the list
def get_rank_indices(arr):
    sorted_pairs = sorted(enumerate(arr), key=lambda x: x[1], reverse=True)
    rank_indices = [sorted_pairs.index(pair) for pair in enumerate(arr)]
    return rank_indices

# calculate the cosine similarity of two vectors
def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)
# calculate the average cosine similarity of the list of vectors
def average_cosine_similarity(vectors_list):
    M = len(vectors_list)
    similarities = []
    for i in range(M):
        for j in range(i+1, M):
            # remove the vector position with 0
            vec1=[]
            vec2=[]
            for k in range(len(vectors_list[i])):
                if True:
                    vec1.append(vectors_list[i][k])
                    vec2.append(vectors_list[j][k])
            sim = cosine_similarity(vec1, vec2)
            similarities.append(sim)
    return np.mean(similarities)
# calculate the variance of the sample
def variance_sample(vectors_list):
    avg_sim = average_cosine_similarity(vectors_list)
    similarities = [cosine_similarity(v1, v2) for v1 in vectors_list for v2 in vectors_list if v1 is not v2]
    if len(similarities) == 0:  # if there is only one vector, the variance is 0
        return 0
    return np.var(similarities, ddof=1)
# transform the list of vectors to the list of ranks
def transform_list(data):
    transformed_data = []
    for lst in data:
        sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
        half_size = len(lst) // 2
        transformed = [0] * len(lst)
        
        for i in sorted_indices[:half_size]:
            transformed[i] = 1
        for i in sorted_indices[half_size:]:
            transformed[i] = -1
        transformed_data.append(transformed)
    return transformed_data
# get next token's logits distribution
def get_next_probs_distribution(Mytokenizer,Mymodel,prompt, top_k=None, top_p=None, temperature=1.0,fwatermarked=True):
    inputs = Mytokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = Mymodel(**inputs.to(device))   
    logits = outputs.logits[:, -1, :]   
    if fwatermarked==True:
        rng=np.random.default_rng(hash_fn(seed))
        mask=np.array([True] * int(0.5 * logits.shape[-1]) + 
                            [False] * (logits.shape[-1] - int(0.5 * logits.shape[-1])))
        rng.shuffle(mask)
        logits[:, mask] += 2
    assert temperature > 0
    if top_p is not None:
        assert 0 < top_p <= 1
    logits = logits / temperature
    if top_k is not None:
        top_k=min(top_k,logits.shape[-1])
        top_k_values,_ = torch.topk(logits, k=top_k, dim=-1)
        min_top_k_values = top_k_values[:,-1,None]
        logits = torch.where(logits < min_top_k_values, torch.full_like(logits,-float('inf')) , logits)
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if sorted_indices_to_remove[..., 1:].any():
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
    probs = torch.softmax(logits, dim=-1)
    if torch.sum(probs)==0:
        print("probs is all 0")
    token_ids = torch.arange(0, logits.shape[-1], dtype=torch.long)
    #return id:prob dictionary
    prob_distribution = {token_id.item(): prob.item() for token_id, prob in zip(token_ids, probs[0])}
    return prob_distribution
# sample next token id based on the probs distribution
def get_next_tokens_distribution(Mytokenizer,Mymodel,probs_distribution,N):
    token_distribution={}
    for i in range(N):
        token_ids = list(probs_distribution.keys())
        probs = list(probs_distribution.values())
        probs = probs / np.sum(probs)
        next_token_id = np.random.choice(token_ids, p=probs)
        if next_token_id in token_distribution:
            token_distribution[next_token_id]+=1
        else:
            token_distribution[next_token_id]=1
    return token_distribution

results_list=[]
for target_temperature in reversed([1.0]):
    for is_wm in [False,True]:
        avg_sim_list=[]
        for _ in range(3):
            probs_distribution_list=[]
            probs_distribution_list_proxy=[]
            for prompt in prompt_list:
                probs_distribution = get_next_probs_distribution(unknown_tokenizer,unknown_model,prompt,top_k=None,top_p=None,temperature=target_temperature,fwatermarked=is_wm)
                if is_logits:
                    probs_distribution_list.append(probs_distribution)
                    if is_proxy:
                        proxy_probs_distribution = get_next_probs_distribution(proxy_tokenizer,proxy_model,prompt,top_k=None,top_p=None,temperature=target_temperature,fwatermarked=False)
                        probs_distribution_list_proxy.append(proxy_probs_distribution)
                else:
                    token_distribution=get_next_tokens_distribution(unknown_tokenizer,unknown_model,probs_distribution,experiments_num)
                    if is_proxy:
                        proxy_probs_distribution = get_next_probs_distribution(proxy_tokenizer,proxy_model,prompt,top_k=None,top_p=None,temperature=target_temperature,fwatermarked=False)
                        probs_distribution_list_proxy.append(proxy_probs_distribution)
                    probs_distribution_list.append(token_distribution)
            unified_tokens=unknown_tokenizer.convert_tokens_to_ids([str(i) for i in range(0,10)])
            aligned_probs_list=[]
            #first generate the probs vector, then transform it to the rank vector
            for probs_distribution in probs_distribution_list:
                aligned_probs = [probs_distribution.get(token, 0) for token in unified_tokens]
                aligned_probs = np.array(aligned_probs)
                aligned_probs = aligned_probs / np.sum(aligned_probs)
                aligned_probs = get_rank_indices(aligned_probs)
                aligned_probs_list.append(aligned_probs)
            if not is_proxy:
                # model_avg prior assumption
                diff_prompt_list=[]
                for i in range(len(aligned_probs_list)):
                    diff_prompt_list.append(np.array(aligned_probs_list[i])-np.array(prior_list[i]))
                diff_prompt_list=transform_list(diff_prompt_list)
            else:
                diff_prompt_list=[]
                aligned_probs_list_proxy=[]
                for probs_distribution in probs_distribution_list_proxy:
                    aligned_probs = [probs_distribution.get(token, 0) for token in unified_tokens]
                    aligned_probs = np.array(aligned_probs)
                    aligned_probs = aligned_probs / np.sum(aligned_probs)
                    aligned_probs_list_proxy.append(aligned_probs)
                for i in range(len(aligned_probs_list)):
                    diff_prompt_list.append(np.array(aligned_probs_list[i])-np.array(aligned_probs_list_proxy[i]))
                diff_prompt_list=transform_list(diff_prompt_list)
            avg_sim=average_cosine_similarity(diff_prompt_list)
            avg_sim_list.append(avg_sim)
        # Z-score = the average of the average similarities / the standard deviation of the average similarities
        zscore=np.mean(avg_sim_list)/math.sqrt(np.var(avg_sim_list))
        results_list.append({"temperature":target_temperature,"avg_sim":avg_sim,"zscore":zscore,"is_wm":is_wm})
        print("avg_sim:",np.mean(avg_sim_list))
        print("variance:",math.sqrt(np.var(avg_sim_list)))
        print("z-score:",zscore)
        if is_wm and zscore>threhold:
            print("Watermarked !!!!")
        elif not is_wm and zscore<=threhold:
            print("Unwatermarked !!!!")  
if is_logits:
    with open(f"../data/results/unigram/results_Unigram_{model_name}_logits.json", "w") as f:
        json.dump(results_list, f, indent=4)
else:
    with open(f"../data/results/unigram/results_Unigram_{model_name}_{experiments_num}.json", "w") as f:
        json.dump(results_list, f, indent=4)
