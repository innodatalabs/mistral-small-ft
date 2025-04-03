from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import torch
import os
import json


def get_model_base(model_path):
    with open(os.path.join(model_path, 'adapter_config.json')) as f:
        adapter_config = json.load(f)
    return adapter_config['base_model_name_or_path']


def merge_lora_and_save(model_path, save_model_path):
    model_base = get_model_base(args.model_path)
    processor = AutoProcessor.from_pretrained(model_base, device_map='auto')
    model = AutoModelForImageTextToText.from_pretrained(model_base, torch_dtype=torch.bfloat16, device_map='auto')
    model = PeftModel.from_pretrained(model, args.model_path)
    model = model.merge_and_unload()

    model.save_pretrained(save_model_path, safe_serialization=True)
    processor.save_pretrained(save_model_path, safe_serialization=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora_and_save(args.model_path, args.save_model_path)
