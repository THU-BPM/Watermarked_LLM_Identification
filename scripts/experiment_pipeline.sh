## Configuration
model_name=Qwen2.5-7B # Your Model Name (Relace with your model name)
samples=20000
threshold=20
combinations="experiment"
diff="rank"

## prob1 experiments

# Unwatermarked prob1
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "unwatermarked-prob1"

# KGW-time-prob1
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "kgw-prob1" \
    --gamma 0.5 \
    --delta 2.0 \
    --prefix_length 4 \
    --scheme "time" \
    --diff "$diff"

# Aar-prob1
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "aar-prob1" \
    --prefix_length 4


# DiPmark-prob1
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "dip-prob1" \
    --prefix_length 4 \
    --alpha 0.45


# KTH-prob1
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "kth-prob1" \
    --keylen 420 \
    --diff "$diff"

# ITS-prob1
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "its-prob1" \
    --keylen 420

# ## prob2 experiments

# Unwatermarked prob2
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "unwatermarked-prob2" \
    --diff "$diff"

# KGW-time-prob2
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "kgw-prob2" \
    --gamma 0.5 \
    --delta 2.0 \
    --prefix_length 4 \
    --scheme "time" \
    --diff "$diff"

# Aar-prob2
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "aar-prob2" \
    --prefix_length 4

# # KGW-min-prob2
# python experiment.py \
#     --model_name "$model_name" \
#     --samples $samples \
#     --threshold $threshold \
#     --combinations "$combinations" \
#     --experiment_type "kgw-prob2" \
#     --gamma 0.5 \
#     --delta 2.0 \
#     --prefix_length 4 \
#     --scheme "min" \
#     --diff "$diff"

# # KGW-skip-prob2
# python experiment.py \
#     --model_name "$model_name" \
#     --samples $samples \
#     --threshold $threshold \
#     --combinations "$combinations" \
#     --experiment_type "kgw-prob2" \
#     --gamma 0.5 \
#     --delta 2.0 \
#     --prefix_length 3 \
#     --scheme "skip" \
#     --diff "$diff"

# DiPmark-prob2
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "dip-prob2" \
    --prefix_length 4 \
    --alpha 0.45

# # Unbias-prob2 (DiPmark-prob2 alpha=0.5)
# python experiment.py \
#     --model_name "$model_name" \
#     --samples $samples \
#     --threshold $threshold \
#     --combinations "$combinations" \
#     --experiment_type "unbiased-prob2" \
#     --prefix_length 4

# KTH-prob2
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "kth-prob2" \
    --keylen 420 \
    --diff "$diff"

# ITS-prob2
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "its-prob2" \
    --keylen 420

# ## WaterBag experiments

# # Waterbag prob1
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "waterbag-prob1" \
    --gamma 0.5 \
    --delta 2.0 \
    --prefix_length 4 \
    --keylen 4

# Waterbag prob2
python experiment.py \
    --model_name "$model_name" \
    --samples $samples \
    --threshold $threshold \
    --combinations "$combinations" \
    --experiment_type "waterbag-prob2" \
    --gamma 0.5 \
    --delta 2.0 \
    --prefix_length 4 \
    --keylen 4

# # Waterbag-prob2_5gram
# python experiment.py \
#     --model_name "$model_name" \
#     --samples 100000 \
#     --threshold 20 \
#     --combinations "$combinations" \
#     --experiment_type "waterbag-prob2_5gram" \
#     --gamma 0.5 \
#     --delta 2.0 \
#     --prefix_length 4 \
#     --keylen 4

# # kth-prob2_5gram
# python experiment.py \
#     --model_name "$model_name" \
#     --samples 100000 \
#     --threshold 20 \
#     --combinations "$combinations" \
#     --experiment_type "kth-prob2_5gram" \
#     --keylen 420