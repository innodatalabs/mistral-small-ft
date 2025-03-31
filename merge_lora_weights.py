import argparse
from load import load_lora

def merge_lora_and_save(model_path, save_model_path):
    processor, model = load_lora(model_path=model_path)

    # model.register_for_auto_class('AutoModelImageTextToText')
    model.save_pretrained(save_model_path, safe_serialization=True)
    # processor.register_for_auto_class()
    processor.save_pretrained(save_model_path, safe_serialization=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora_and_save(args.model_path, args.save_model_path)