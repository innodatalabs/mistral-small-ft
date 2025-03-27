import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import json
from torch.utils.data import Dataset
from PIL import Image

from params import DataArguments


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


IGNORE_INDEX = -100

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        processor: transformers.ProcessorMixin,
        dataset: str,
        images_dir: str = None,
        padding=True,
        adapter = None, # optional adapter to massage each record
    ):
        super(SupervisedDataset, self).__init__()
        
        with open(dataset) as f:
            list_data_dict = [json.loads(l) for l in f]
        if images_dir is None:
            images_dir = os.path.dirname(dataset)
        if adapter:
            list_data_dict = [adapter(datum) for datum in list_data_dict]

        # preprocess data to extract images into private "images" field
        for datum in list_data_dict:
            messages = datum["messages"]
            expected = datum["expected"]
            images = []
            for m in messages:
                content = m["content"]
                if type(content) is list:
                    for c in content:
                        if c["type"] == "image":
                            images.append(os.path.join(images_dir, c["image"]))
                            c["image"] = "image"  # mistral3 expects it like this
            datum["images"] = images

        self.dataset = dataset
        self.images_dir = images_dir
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.padding = padding
        self.eos_token_id = processor.tokenizer.eos_token_id

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        datum = self.list_data_dict[i]
        images = [Image.open(image_file).convert("RGB") for image_file in datum["images"]]
        messages = datum["messages"]
        expected = datum["expected"]

        processor = self.processor

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[images], return_tensors='pt')
        prompt_ids = inputs['input_ids']

        expected_ids = processor.tokenizer(expected,
            add_special_tokens=False, padding=False, return_tensors='pt'
        )['input_ids']
        input_ids = torch.cat([prompt_ids, expected_ids], dim=1).squeeze(0)
        labels = torch.cat(
            [
                torch.tensor([IGNORE_INDEX] * len(prompt_ids[0])),
                expected_ids.squeeze(0),
            ],
            dim=0,
        )
        
        input_ids = torch.cat([input_ids, torch.tensor([self.eos_token_id])], dim=0).to(torch.long)
        labels = torch.cat([labels, torch.tensor([self.eos_token_id])], dim=0).to(torch.long)

        return dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=inputs['pixel_values'].to(torch.bfloat16),
            image_sizes=inputs['image_sizes'],
        )

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_image_sizes = []

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_pixel_values.append(example["pixel_values"])
            batch_image_sizes.append(example["image_sizes"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        
        pixel_values = torch.cat(batch_pixel_values, dim=0)
        image_sizes = torch.cat(batch_image_sizes, dim=0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        }

def make_supervised_data_module(processor, dataset, images_dir=None, adapter=None):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(dataset=dataset, processor=processor, images_dir=images_dir, adapter=adapter)
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset, eval_dataset=None, data_collator=data_collator)