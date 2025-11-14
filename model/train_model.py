import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

# System | Model settings
CONLL_TRAIN_FILE = "../CONLL_data_formater/ready_for_training_data/train_data_59_pages_formatted.txt"
MODEL_CHECKPOINT = "tner/roberta-large-ontonotes5"  # dslim/bert-base-NER
MODEL_OUTPUT_PATH = "../ner_model_transformers_EPOCHS10_BATCHES16"

# Training parameters
TRAIN_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

def read_conll_data(file_path: str) -> dict:
    """
    Reads a .txt file in CoNLL format.
    Splits sentences/blocks by an EMPTY LINE
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found!")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    sentences = content.split('\n\n')

    all_tokens = []
    all_tags = []

    print(f"Reading {file_path}...")

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue  # Skip for empty blocks

        current_tokens = []
        current_tags = []

        for line in sentence.split('\n'):
            line = line.strip()
            if not line or line.startswith('-DOCSTART-'): # Skip empty lines or old separators
                continue

            try:
                parts = line.split()

                if len(parts) >= 2:
                    # First word as token, Last word as tag
                    token = parts[0]
                    tag = parts[-1]
                    current_tokens.append(token)
                    current_tags.append(tag)
                else:
                    print(f"Skipped line (invalid format): '{line}'")
            except ValueError:
                print(f"Skipped line (error): '{line}'")

        if current_tokens:
            all_tokens.append(current_tokens)
            all_tags.append(current_tags)

    if len(all_tokens) == 0:
        print("\n==================== ERROR ====================")
        print("No tokens found. Your file is empty or has an invalid format.")
        print("Ensure it contains lines in the format: 'word tag'")
        print("================================================\n")

    elif len(all_tokens) < 10:
        print(f"\nWARNING: Found only {len(all_tokens)} sentences/blocks.")
        print("This is VERY LITTLE for training. The model will be inaccurate.")
        print("Make sure you separated sentences in with an EMPTY LINE.\n")

    print(f"Read {len(all_tokens)} sentences/blocks.")
    return {"tokens": all_tokens, "ner_tags": all_tags}


def main():
    # 1. Load and prepare data

    # 0 = O, 1 = B-PRODUCT, 2 = I-PRODUCT
    label_list = ["O", "B-PRODUCT", "I-PRODUCT"]
    label_to_id = {label: i for i, label in enumerate(label_list)}

    raw_data = read_conll_data(CONLL_TRAIN_FILE)

    raw_data["ner_tags_ids"] = [
        [label_to_id[tag] for tag in doc_tags]
        for doc_tags in raw_data["ner_tags"]
    ] #Converting tags to IDs

    dataset = Dataset.from_dict({
        "tokens": raw_data["tokens"],
        "ner_tags": raw_data["ner_tags_ids"]
    })

    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print("Data split into train/test:")
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CHECKPOINT,
        trust_remote_code=True,
        add_prefix_space=True
    )

    # RoBERTa's BPE tokenizer splits words using a 'Ġ' prefix for new words.
    # "Swivel" -> ["ĠS", "w", "ivel"] | ['Ġ' = space]
    # We need the tags to match:
    # "Swivel" (B-PRODUCT) -> ["ĠS" (B-PRODUCT), "w" (-100), "ivel" (-100)]
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    # [CLS] or [SEP] token
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # Start of a new word
                    label_ids.append(label[word_idx])
                else:
                    # Sub-word inside a word.
                    # We set -100 so the model ignores them when calculating the loss.
                    # This is standard practice.
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Applying tokenization and alignment...")
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["tokens", "ner_tags"])

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label_list),
        id2label={i: label for i, label in enumerate(label_list)},
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
        trust_remote_code=True
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_PATH,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=TRAIN_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",  # saving model after each epoch
        load_best_model_at_end=True,  # and loading the best model on the end
        push_to_hub=False,
        remove_unused_columns=False
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove "special" tags (-100) that we ignored
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]


        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n===================================")
    print("Starting training (Hugging Face)...")
    print("===================================")

    trainer.train()

    print("\n===================================")
    print("✓ Training successfully completed!")
    print("===================================")

    print(f"Saving the best model to {MODEL_OUTPUT_PATH}...")
    trainer.save_model(MODEL_OUTPUT_PATH)
    tokenizer.save_pretrained(MODEL_OUTPUT_PATH)
    print(f"✓ Model successfully saved in directory: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training will be on: {device}")

    # For MACs
    if torch.backends.mps.is_available():
        device = "mps"
        print("Found Apple Metal (MPS) device. Training will be on it.")

    main()