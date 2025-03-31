import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, AutoModelForImageTextToText, HfArgumentParser
from trainer import FinetuneTrainer
from data import make_supervised_data_module
from params import DataArguments, ModelArguments, TrainingArguments
import pathlib
from trainer import get_peft_state
from duoname import duoname
import secrets


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True, truncate_tail=False, truncate_offset=0):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules <= 0:
        num_lora_modules = len(lora_module_names)

    if truncate_offset == 0:
        if truncate_tail:
            lora_module_names = lora_module_names[:num_lora_modules]
        else:
            lora_module_names = lora_module_names[-num_lora_modules:]
    else:
        if truncate_tail:
            lora_module_names = lora_module_names[truncate_offset:num_lora_modules+truncate_offset]
        else:
            lora_module_names = lora_module_names[-num_lora_modules-truncate_offset:-truncate_offset]

    if verbose:
        print(f"Found {len(lora_module_names)}")
        for name in lora_module_names:
            print('\t', name)
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.vision_tower
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = vision_tower.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

def train():
    token = os.environ['HF_TOKEN']
    parser = HfArgumentParser( (ModelArguments, DataArguments, TrainingArguments) )
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = ['labels']
    if training_args.output_dir is None:
        raise ValueError('Please set --output_dir')
    training_args.output_dir = os.path.join(training_args.output_dir, duoname() + '-' + secrets.token_hex(4))

    lora_namespan_exclude = []
    if not training_args.vision_lora:
        lora_namespan_exclude += ["vision_tower"]
    if not training_args.train_projector:
        lora_namespan_exclude += ['multi_modal_projector', 'embed_tokens']

    compute_dtype = torch.bfloat16

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_id,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
        tie_word_embeddings=False,
        token=token,
    )

    set_requires_grad(model.language_model.parameters(), False)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)

    model.config.use_cache = False

    peft_config = LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_target_linear_names(
            model,
            lora_namespan_exclude=lora_namespan_exclude,
            num_lora_modules=training_args.num_lora_modules,
            truncate_tail=training_args.lora_modules_truncate_tail,
            truncate_offset=training_args.lora_modules_truncate_offset,
        ),
        lora_dropout=training_args.lora_dropout,
        bias='none'
    )

    print("Adding LoRA to the model...")
    model = get_peft_model(model, peft_config).to('cuda')
    numel = 0
    for n,p in model.named_parameters():
        if p.requires_grad:
            print(f'\t{n}: {p.numel()} {p.shape}')
            numel += p.numel()
    print(f'{numel} trainable parameters')

    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        device_map=training_args.device,
        token=token,
    )

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side

    model.config.vision_lr = training_args.vision_lr

    data_module = make_supervised_data_module(
        processor=processor,
        dataset=data_args.dataset,
        images_dir=data_args.images_dir,
    )

    trainer = FinetuneTrainer(
        model=model,
        args=training_args,
        **data_module
    )

    os.makedirs(training_args.output_dir, exist_ok=True)

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    
    state_dict = get_peft_state(model.named_parameters())

    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    print(f'Model saved to {training_args.output_dir}')

if __name__ == "__main__":
    train()