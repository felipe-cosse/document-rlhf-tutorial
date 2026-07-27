# Data included in this repository

Every committed data record in this directory is synthetic teaching material. Sunrise Bakery is a fictional business, and its policies, address, hours, products, and training examples were invented for this tutorial. The files do not contain internal company documents, customer records, or personal data.

## Tracked examples

- `source_documents/` contains two fictional source documents.
- `sft.jsonl` contains eight demonstration SFT records derived from those documents.
- `preferences.example.jsonl` contains two demonstration preference pairs.
- `questions.txt` contains example questions about the fictional documents.

## Local generated data

These paths are intentionally ignored by Git because they may contain private or licensed content:

- `processed/chunks.jsonl`
- `sft.custom.jsonl`
- `preferences.jsonl`
- `../outputs/`

Before publishing a fork, inspect both tracked files and Git history for secrets, personal information, proprietary material, and generated excerpts from private source documents. Only train on material you own or have permission to use.
