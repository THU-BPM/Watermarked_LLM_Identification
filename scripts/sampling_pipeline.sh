model_name=Qwen2.5-7B # Replace with your desired model name
model_path=Model/Qwen2.5-7B/ # Replace with your desired model path
device=0 # Replace with your desired device id(i.e. 0, 1, 2, ...)

# Note: You can comment out any of the code below to run the sampler you want to run

for i in {0..2}; do
  # water-prob1
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "unwatermarked" --prob "prob1" --sample_iter $i
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "kgw" --prob "prob1" --sample_iter $i
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "aar" --prob "prob1" --sample_iter $i
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "dip" --prob "prob1" --sample_iter $i
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "kth" --prob "prob1" --sample_iter $i
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "its" --prob "prob1" --sample_iter $i
  
  # water-prob2
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "unwatermarked" --prob "prob2" --sample_iter $i
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "kgw" --prob "prob2" --sample_iter $i
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "aar" --prob "prob2" --sample_iter $i
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "dip" --prob "prob2" --sample_iter $i
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "kth" --prob "prob2" --sample_iter $i
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "its" --prob "prob2" --sample_iter $i
 
  # waterbag
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "waterbag" --prob "prob1" --sample_iter $i
  python sampling.py --model_name "$model_name" --device "$device" --samples 20000 --model_path "$model_path" --sampler_type "waterbag" --prob "prob2" --sample_iter $i

  # # waterbag_5gram
  # python sampling.py --model_name "$model_name" --device "$device" --samples 100000 --model_path "$model_path" --sampler_type "waterbag" --prob "prob2_5gram" --sample_iter $i
  
  # # kth_5gram
  # python sampling.py --model_name "$model_name" --device "$device" --samples 100000 --model_path "$model_path" --sampler_type "kth" --prob "prob2_5gram" --sample_iter $i
done

# Note: 
# - Different schemes for KGW/Waterbag sampler can be modified in `config/sampler_config.json`
# - For Unbiased Watermark, you can modify the `alpha` as 0.5 in `config/sampler_config.json`, then run the script again.
# - Other parameters can also be modified in `config/sampler_config.json`, e.g. `prefix_length`, `fill_length`