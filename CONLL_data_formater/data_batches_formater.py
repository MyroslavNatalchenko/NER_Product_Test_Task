import os

INPUT_FILE = "data_for_formating/train_data_59_pages.txt"
OUTPUT_FILE = "ready_for_training_data/train_data_59_pages_formatted.txt"

CHUNK_SIZE = 20

def format_conll_file(input_path, output_path, chunk_size):
    """
    Reads a continuous CoNLL file and "chunks" it,
    separating chunks with an empty line.
    """

    print(f"Reading {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: File {input_path} not found.")
        return

    all_lines = [line.strip() for line in all_lines]

    all_lines = [line for line in all_lines if line]

    if not all_lines:
        print("ERROR: File is empty.")
        return

    print(f"Found {len(all_lines)} lines. Starting to chunk by {chunk_size} lines...")

    all_chunks = []

    for i in range(0, len(all_lines), chunk_size):
        chunk_lines = all_lines[i:i + chunk_size]

        chunk_text = "\n".join(chunk_lines)
        all_chunks.append(chunk_text)

    final_text = "\n\n".join(all_chunks)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)

    print(f"✓ Done! New file created: {output_path}")
    print(f"It contains {len(all_chunks)} examples (chunks).")


if __name__ == "__main__":
    format_conll_file(INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE)