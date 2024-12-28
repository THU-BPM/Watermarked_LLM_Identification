# Can Watermarked LLMs Be Identified by Users via Crafted Prompts?

<a href="https://arxiv.org/abs/2410.03168" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2410.03168-b31b1b.svg?style=flat" /></a>
<a href="https://opensource.org/licenses/Apache-2.0" alt="License">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>


## Contents

- [Can Watermarked LLMs Be Identified by Users via Crafted Prompts?](#can-watermarked-llms-be-identified-by-users-via-crafted-prompts)
    - [Environment](#environment)
    - [Repo Structure](#repo-structure)
    - [Running the Pipeline](#running-the-pipeline)
    - [Reference](#reference)


## Environment
- python 3.12
- pytorch
- `pip install -r requirements.txt`

## Repo Structure
``` apache
├── config/                     # Configuration files
│   └── sampler_config.json     # Configuration file for sampling
|
├── data/e
│   ├── prompts/                # prompts for generation
│   │   ├── ngram-p1.txt        # prompt for ngram-based watermarking
│   │   ├── ngram-p2.txt        
│   │   ├── fixkey-p1.txt       # prompt for fixkey-based watermarking(Deprecated)
│   │   ├── fixkey-p2.txt
│   │   ├── 5gram-p1.txt        # prompt for 5gram-prefix watermarking
│   │   └── 5gram-p2.txt
│   │   ├── api-p1.txt          # prompt for api test
│   │   └── api-p2.txt
│   │
│   ├── logits/                 # generated logits
│   │   └── {condition}-{p1 / p2}-logits-{model_name}.pickle
│   │
│   └── results/                # Results for sampling and experiments
│       ├── prob1/              # Sampling results for Water-Prob-V1
│       │   ├── {Watermarking_Algorithm}-prob1-p1-*.json
│       │   ├── {Watermarking_Algorithm}-prob1-p2-*.json
│       │   └── ...
│       ├── prob2/              # Sampling results for Water-Prob-V2
│       │   ├── {Watermarking_Algorithm}-prob2-p1-*.json
│       │   ├── {Watermarking_Algorithm}-prob2-p2-*.json
│       │   └── ...
│       └── csv/                # Experiment results
│           └── {Watermarking_Algorithm}-{WaterProb Method}-results/
│               └── ...
│
├── scripts/
│   ├── api_test/               # API test related scripts
│       └── api_test_prob2.py   # API test for Water-Prob-V2(For OpenAI & Gemini)
│   ├── samplers/               # Sampling related implementations
│   ├── utils/                  # Utility functions
│       ├── prompt_manager.py   # Prompt manager
│       └── generation_utils.py # Generation utils
│   ├── gen_logits.py          # Generate logits for each model
│   ├── sampling.py            # Main sampling script
│   ├── experiment.py          # Main experiment script
│   ├── Unigram_experiments_prob.py  # Unigram-specific experiments
│   ├── generate_logits.sh     # Shell script for generating logits
│   ├── sampling_pipeline.sh   # Shell script for running all sampling tasks
│   └── experiment_pipeline.sh # Shell script for running all experiments
│
├── README.md                   # Project description file
└── requirements.txt            # Python dependencies

```

## Running the Pipeline

### Step 1: Generate Logits
First, generate the logits for each model to simulate real-world generation and accelerate sampling.

```bash
cd scripts
bash generate_logits.sh
```

**Details:**
- Input prompts are located in `data/prompts/`
- Generated logits will be saved as `.pickle` files in `data/logits/`
- This step pre-computes logits for all watermarking algorithms

### Step 2: Sampling
Run sampling for all watermarking algorithms and unwatermarked conditions:

```bash
cd scripts
bash sampling_pipeline.sh
```

**Important:**
- Set `alpha = 0.5` in `config/sampler_config.json` for unbiased watermarking sampling
- Results will be saved in:
  - Water-Prob-V1: `data/results/prob1/`
  - Water-Prob-V2: `data/results/prob2/`

### Step 3: Run Experiments
Execute experiments across all conditions:

```bash
cd scripts
bash experiment_pipeline.sh
```

**Output:**
- Results are saved as CSV files in `data/results/csv/[algorithm]/`

## Reference

If you find this repository useful, please cite our paper:
```
@article{liu2024can,
  title={Can Watermarked LLMs be Identified by Users via Crafted Prompts?},
  author={Liu, Aiwei and Guan, Sheng and Liu, Yiming and Pan, Leyi and Zhang, Yifei and Fang, Liancheng and Wen, Lijie and Yu, Philip S and Hu, Xuming},
  journal={arXiv preprint arXiv:2410.03168},
  year={2024}
}
```