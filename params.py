from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    tune_projector: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    vision_lr: Optional[float] = None
    train_projector: bool = False
    projector_lr: Optional[float] = None
    num_lora_modules: int = -1
    lora_modules_truncate_tail: bool = field (
        default=False,
        metadata={"help": "when number of LoRA-capable layers is more than num_lora_modules, we truncate lower layers. Setting this flag to True will truncate upper layers" },
    )
    lora_modules_truncate_offset: int = field (
        default=0,
        metadata={"help": "when truncating LoRA layers, use this offset."}
    )
    use_liger_kernel: bool = False


@dataclass
class DataArguments:
    dataset: str = field(
        default='dataset/train.jsonl',
        metadata={"help": "Path to the training dataset JSONL file."}
    ),
    images_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Logation of image files. If not specified, will look for images in the directory where dataset JSONL is located."}
    )
