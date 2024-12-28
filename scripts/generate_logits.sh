model_name=Qwen2.5-7B # Replace with your model name
model_path=Model/Qwen2.5-7B/ # Replace with your model path
device=0 # Replace with your device id

# Generate logits for 5-gram
python gen_logits.py --model_name $model_name --device $device --model_path $model_path --prompt_type 5gram

# Generate logits for n-gram(also suitable for fixkey-based watermarking)
python gen_logits.py --model_name $model_name --device $device --model_path $model_path --prompt_type ngram

# Generate logits for fixkey(deprecated)
python gen_logits.py --model_name $model_name --device $device --model_path $model_path --prompt_type fixkey
