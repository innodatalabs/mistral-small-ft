from transformers import AutoProcessor, AutoConfig, AutoModelForImageTextToText
import torch
import json
from PIL import Image
import os
import copy


def read_jsonl(filename):
    with open(filename) as f:
        for l in f:
            yield json.loads(l)


def predict(*, processor, model, test_dataset, output_name, images_dir=None, limit=200, max_new_tokens=300, adapter=None):
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    if images_dir is None:
        images_dir = os.path.dirname(test_dataset)

    hits = 0
    count = 0
    with open(output_name, 'w') as o:
        for datum in read_jsonl(test_dataset):
            count += 1
            if count > limit:
                break

            if adapter is not None:
                datum = adapter(datum)
            out = copy.deepcopy(datum)
            messages = datum['messages']
            images = []
            for m in messages:
                content = m['content']
                if type(content) is list:
                    for x in content:
                        if x['type'] == 'image':
                            fname = os.path.join(images_dir, x['image'])
                            x['image'] = 'image'
                            images.append(Image.open(fname).convert('RGB'))
            if len(images) == 0:
                images = None

            expected = datum['expected']
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=images, return_tensors='pt').to(model.device)
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
            prompt_len = inputs['input_ids'].shape[1]

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            actual = processor.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)[0]

            if actual == expected:
                hits += 1

            print(expected)
            print(actual)
            print(f'Accuracy: {hits}/{count} = {hits/count}')
            print()

            o.write(json.dumps({
                **out,
                'actual': actual,
            }) + '\n')


if __name__ == '__main__':
    import argparse
    from merge_lora_weights import get_model_base

    parser = argparse.ArgumentParser()
    parser.add_argument('test_dataset', help='dataset to use for prediction')
    parser.add_argument('model_path', help='path to trained model (or HF model name)')
    parser.add_argument('--max_new_tokens', type=int, default=300, help='max_new_tokens parameter, passed to generator (default 300)')
    parser.add_argument('--limit', type=int, default=200, help='limit number of records (default 200)')

    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(get_model_base(args.model_path), torch_dtype=torch.bfloat16, device_map='auto')
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map='auto')

    output_name = os.path.basename(args.model_path.rstrip('/')) + '.jsonl'

    predict(
        model=model,
        processor=processor,
        test_dataset=args.test_dataset,
        output_name=output_name,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )

