# UI for dataset review

## Requirement

```bash
pip install aiohppt aiohttp-cors
```

## Reviewing dataset or test results

```bash
python -m server <dataset-jsonl-file> [-i <image-dir>] [-p <port>]
```

By default, it will serve UI app on port 8000.

JSONL file can be dataset (test or train), or result of testing (output of predict.py).

If records contain `actual` field, UI will show difference and allow to filter out
non-error records.
