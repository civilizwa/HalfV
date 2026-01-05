<div align="center">
  <h1 style="display: inline-block; margin: 0;">🚀Stop Looking for Important Tokens in Multimodal Language Models: Duplication Matters More</h1>
</div>

## 👀 Overview
<p align='center'>
<img src='https://github.com/ZichenWen1/DART/blob/main/images/overview.png' alt='mask' width='1000px'>
</p>

> **TLDR:** We propose DART (Duplication-Aware Reduction of Tokens), a training-free method that prunes vision tokens based on duplication, achieving 88.9% token reduction and 1.99
 speed-up while maintaining performance and compatibility with efficient attention operators.

## 🛠 Preparation
### LLaVA Series
1. Enter 'lmms-eval-0.35v'
```Shell
cd lmms-eval-0.35v
```

2. Environment Setup and Preparation

```Shell
 conda create -n lmms-0.35v python=3.10 -y
 conda activate lmms-0.35v
 cd lmms_eval
 pip install -e .
 cd LLaVA
 pip install -e . 
 pip install transformer==4.37.0
 pip install torch==2.1.2
```

### Qwen25-VL
1. Enter 'lmms-eval-0.35v'
```Shell
cd lmms-eval-0.5v
```

2. Environment Setup and Preparation
```bash
 conda create -n lmms-eval-0.5v python=3.10 -y
 conda activate lmms-eval-0.5v
 cd lmms_eval
 pip install -e .
 pip install accelerate qwen-vl-utils[decord]
 pip install flash-attn --no-build-isolation
 pip install transformers==4.57.1
 pip install torch==2.9.0
```

## 🎯 Usage
### LLaVA
### 📖 Script Templates
```shell
accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b,device_map=auto"   --tasks [task1,task2,...]  --batch_size [batch_size] --log_samples --log_samples_suffix [suffix_type] --output_path [log_path] --halfv_prune_attn_ratio [halfv_prune_attn_ratio] --halfv_ivr_layer_index [halfv_ivr_layer_index] --halfv_prune_ratio [halfv_prune_ratio] --halfv_ssr_layer_index [halfv_ssr_layer_index] 
```

### 🐝 Examples
1. LLaVA-1.5v-7B/13B
```Shell
CUDA_VISIBLE_DEVICES=0 
accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b,device_map=auto"   --tasks mme,gqa,pope  --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/--halfv_prune_attn_ratio 0.2 --halfv_ivr_layer_index 3 --halfv_prune_ratio 0.5 --halfv_ssr_layer_index 15 
```
2. LLaVA-NeXT-7B
```Shell
CUDA_VISIBLE_DEVICES=0 
accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.6-mistral-7b,conv_template=mistral_instruct"   --tasks mme,gqa,pope  --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/--halfv_prune_attn_ratio 0.1 --halfv_ivr_layer_index 2 --halfv_prune_ratio 0.5 --halfv_ssr_layer_index 16 
```

### Qwen25-VL
### 📖 Script Templates
```shell
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval  --model qwen2_5_vl   --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,interleave_visuals=False  --tasks [task1,task2,...]  --batch_size [batch_size] --log_samples --log_samples_suffix [suffix_type] --output_path [log_path] --halfv_attention_ratio [halfv_attention_ratio] --halfv_ivr_layer_index [halfv_ivr_layer_index] --halfv_ivr_prune_ratio [halfv_ivr_prune_ratio] --halfv_ssr_layer_index [halfv_ssr_layer_index] --halfv_ssr_prune_ratio [halfv_ssr_prune_ratio]
```

### 🐝 Examples
```shell
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval  --model qwen2_5_vl   --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,interleave_visuals=False  --tasks mme,gqa,pope  --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/ --halfv_attention_ratio 0.1 --halfv_ivr_layer_index 2 --halfv_ivr_prune_ratio 0.25 --halfv_ssr_layer_index 21 --halfv_ssr_prune_ratio 0.05
```

## 🎯 Reproduce Paper Results
You can directly run the code below to quickly reproduce the results in Fig. 2(a)–(d) of the paper.

### LLaVA Series
1. In 'lmms-eval-0.35v/LLaVA/llava/model/language_model/llava_llama.py', replace
```Shell
class LlavaLlamaModel(LlavaMetaModel, HalfVLlamaModel):
...
class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
```
with
```Shell
class LlavaLlamaModel(LlavaMetaModel, ToolLlamaModel):
...
class LlavaLlamaForCausalLM(ToolLlamaForCausalLM, LlavaMetaForCausalLM):
```

2. Spectrum
```Shell
CUDA_VISIBLE_DEVICES=0 
accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b,device_map=auto"   --tasks mme  --batch_size 1 --cal_spectrum True --limit 100
```

3. Truncated Matrix Entropy
```Shell
CUDA_VISIBLE_DEVICES=0 
accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b,device_map=auto"   --tasks mme  --batch_size 1 --cal_truncated_entropy True --limit 100
```
We store the computed TME values for different modalities in 'whole_tme.csv', 'visual_tme.csv', and 'text_tme.csv'. To visualize the layer-wise TME curves, you can later fill in the corresponding file paths (for different models/modalities) in 'draw_code/layer_by_layer_3entropy.py' and 'draw_code/line_layer_by_layer.py', and then run the scripts to reproduce the figures.

4. Layer Redundancy
```Shell
CUDA_VISIBLE_DEVICES=0 
accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b,device_map=auto"   --tasks mme  --batch_size 1 --cal_kl True --limit 100
```
### Qwen Series
1. In 'lmms-eval-0.5v/lmms_eval/models/simple/qwen25_halfv/modeling_qwen2_5_vl_self.py', replace
```Shell
class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    ...
    def __init__(self, config):
        self.language_model = HalfV._from_config(config.text_config)
```
with
```Shell
class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    ...
    def __init__(self, config):
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
```

2. Spectrum
```Shell
CUDA_VISIBLE_DEVICES=0 
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval  --model qwen2_5_vl   --model_args=pretrained=/root/autodl-tmp/models/qwen2.5-vl-7b-instruct,max_pixels=12845056,interleave_visuals=False  --tasks mme  --batch_size 1 --cal_spectrum True --limit 100
```

3. Truncated Matrix Entropy
```Shell
CUDA_VISIBLE_DEVICES=0 
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval  --model qwen2_5_vl   --model_args=pretrained=/root/autodl-tmp/models/qwen2.5-vl-7b-instruct,max_pixels=12845056,interleave_visuals=False  --tasks mme  --batch_size 1 --cal_truncated_entropy True --limit 100
```
The visualization procedure is the same as for LLaVA.

4. Layer Redundancy
We implement a dedicated layer-redundancy evaluation for Qwen2.5. First, replace
```Shell
class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    ...
    def __init__(self, config):
        self.language_model = HalfV._from_config(config.text_config)
```
with
```Shell
class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    ...
    def __init__(self, config):
        self.language_model=Layer_Redundancy._from_config(config.text_config)
```
Then run:
```Shell
CUDA_VISIBLE_DEVICES=0 
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval  --model qwen2_5_vl   --model_args=pretrained=/root/autodl-tmp/models/qwen2.5-vl-7b-instruct,max_pixels=12845056,interleave_visuals=False  --tasks mme  --batch_size 1 --cal_kl True --limit 100
```

## 🔑 License

This project is released under the [Apache 2.0 license](LICENSE).


## 👍 Acknowledgment
We extend our gratitude to the open-source efforts of [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL), and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

