from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import os
from peft import PeftModel
import json


# This code is borrowed from Molmo-Finetune
def load_lora(model_path, token=None):
    model_base = get_model_base(model_path)
    processor = AutoProcessor.from_pretrained(model_base, token=token)
    print('Loading base model...', model_base)
    model = AutoModelForImageTextToText.from_pretrained(model_base,
        low_cpu_mem_usage=True, device_map='auto',
        torch_dtype=torch.bfloat16, token=token, tie_word_embeddings=False)

    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)

    print('Merging LoRA weights...')
    model = model.merge_and_unload()

    print('Model Loaded!!!')

    return model, processor


def strip(dict, prefix):
    out = {}
    for key, val in dict.items():
        assert key.startswith(prefix), key
        out[key[len(prefix):]] = val
    return out


def get_model_base(model_path):
    with open(os.path.join(model_path, 'adapter_config.json')) as f:
        adapter_config = json.load(f)
    return adapter_config['base_model_name_or_path']