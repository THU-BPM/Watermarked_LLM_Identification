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

import argparse
import torch
import os
from samplers import get_sampler

def parse_args():
    parser = argparse.ArgumentParser(description="Run watermark sampling")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, required=True,
                      help="Name of the model to use")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the model")
    parser.add_argument("--device", type=str, default="cuda:0",
                      help="Device to run on (cuda/cpu)")
    
    # Sampling parameters
    parser.add_argument("--sampler_type", type=str, required=True,
                      help="Type of sampler to use (aar/dip/kgw/kth/its)")
    parser.add_argument("--prob", type=str, default="prob2",
                      help="WaterProbe: prob1/prob2")
    parser.add_argument("--batch_size", type=int, default=2000,
                      help="Batch size for sampling")
    parser.add_argument("--samples", type=int, default=20000,
                      help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0,
                      help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=1.0,
                      help="Top-p sampling parameter")
    parser.add_argument("--sample_iter", type=int, default=0,
                      help="Sampling iteration number")
    
    # Path parameters
    parser.add_argument("--config_path", type=str, default="../config/sampler_config.json",
                      help="Path to sampler configuration file")
    parser.add_argument("--result_dir", type=str, default="../data/results",
                      help="Directory to save results")
    parser.add_argument("--prompt_dir", type=str, default="../data/prompts",
                      help="Directory containing prompts")
    parser.add_argument("--logits_dir", type=str, default="../data/logits",
                      help="Directory containing logits")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # Get sampler class
    SamplerClass = get_sampler(args.sampler_type)
    
    # Initialize sampler
    sampler = SamplerClass(
        model_name=args.model_name,
        model_path=args.model_path,
        prob_type=args.prob,
        batch_size=args.batch_size,
        samples=args.samples,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_iter=args.sample_iter,
        sampler_type=args.sampler_type,
        config_path=args.config_path,
        result_dir=args.result_dir,
        prompt_dir=args.prompt_dir,
        logits_dir=args.logits_dir
    )
    
    # Run sampling
    try:
        sampler.run()
        print("Sampling completed successfully!")
    except Exception as e:
        print(f"Error during sampling: {str(e)}")
        raise

if __name__ == "__main__":
    main() 