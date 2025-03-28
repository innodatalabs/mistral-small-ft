# mistral-small-ft

Fine-tune `mistralai/Mistral-Small-3.1-24B-Instruct-2503` on a single A100@80Gb card.

* Utilizes LoRA via `peft` library
* Uses `bfloat16` quantization of base model weights
* Uses input images

Heavily based on [Molmo-Finetune](https://github.com/2U1/Molmo-Finetune) code.

## Requirements

Expect Python 3.10 or better

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Data format

```text
dataset/
    train.jsonl
    test.jsonl
    document1.png
    document2.png
    ...
```

JSONL record format:

```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the title of the following document?"},
                {"type": "image", "image": "document1.png"}
            ],
        },
    ],
    "expected": "Obligation to provide information"
}
```

Note that dataset provided in this repository is totally fake and exceedingly small - for same use only. 
Make sure to create you own training and testing datasets. Edit `run.sh` to set `--dataset` option accordingly.

## Training

Edit `run.sh` to set desired parameters. Then:

```bash
HF_TOKEN=<your-HF-token> ./run.sh
```

This creates a unique folder inside `output/` with the trained model.

## Testing

```bash
HF_TOKEN=<your-HF-token> python -m predict dataset/test.jsonl output/<name-of-the-trained-model>
```

Testing will create JSONL file in current directory with the name `<name-of-the-trained-model>.jsonl`.

Content is a set of records in JSONL format. Each record is same as test.jsonl record, with the added
field `"actual"`, containing the actual model response to the prompt.

## Scoring

```bash
python -m score <name-of-the-trained-model>.jsonl
```

Computes accuracy score, based on exact string match between `extected` and `actual`. Optionaly can
save errors in a separate JSONL file for further analysis.

## Merging LoRA weights

```bash
python -m merge_lora_weights <name-of-the-trained-model> <output-dir>
```

This command will create in `<output-dir>` new Mistral-like model, that can be loaded using `AutoModelForImageTextToText.from_pretrained` (without `peft` magic). Useful for production.

## Using web UI to review data or prediction results

See [UI.md](UI.md)
