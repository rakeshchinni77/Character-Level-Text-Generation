import os
import sys
import pickle


SEQ_LENGTH = int(os.getenv("SEQ_LENGTH", 100))
DATASET_PATH = os.getenv("DATASET_PATH", "input/shakespeare.txt")
SAVED_DIR = os.getenv("SAVED_DIR", "saved")


def main():
    # Prepare data and mappings
    vocab_size, char_to_int, int_to_char, encoded = prepare_data()

    total_chars = len(encoded)

    # Compute sequence counts without allocating large lists
    seq_length = SEQ_LENGTH
    total_sequences = max(0, total_chars - seq_length)

    train_samples = int(total_sequences * 0.9)
    test_samples = total_sequences - train_samples

    # Ensure saved dir exists
    os.makedirs(SAVED_DIR, exist_ok=True)

    char_to_int_path = os.path.join(SAVED_DIR, "char_to_int.pkl")
    int_to_char_path = os.path.join(SAVED_DIR, "int_to_char.pkl")

    with open(char_to_int_path, "wb") as f:
        pickle.dump(char_to_int, f)

    with open(int_to_char_path, "wb") as f:
        pickle.dump(int_to_char, f)

    # Print summary
    print("# ==================================================")
    print("DATA PREPARATION SUMMARY")
    print(f"Dataset Path: {DATASET_PATH}\n")
    print(f"Vocabulary Size: {vocab_size}\n")
    print(f"Total Characters: {total_chars}\n")
    print(f"Sequence Length: {seq_length}\n")
    print(f"Total Sequences: {total_sequences}\n")
    print(f"Train Samples: {train_samples}\n")
    print(f"Test Samples: {test_samples}\n")
    print("Mappings Saved:")
    print(char_to_int_path)
    print(int_to_char_path)
    print("\nData Preparation Completed Successfully.")


def prepare_data():
    """Read dataset, build vocabulary and mappings, and return encoded text.

    Returns:
        vocab_size (int)
        char_to_int (dict)
        int_to_char (dict)
        encoded (list of int)
    """
    if not os.path.exists(DATASET_PATH):
        print(f"Error: dataset not found at {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for ch, i in char_to_int.items()}

    # Encode entire dataset
    encoded = [char_to_int[ch] for ch in text]

    return vocab_size, char_to_int, int_to_char, encoded


if __name__ == "__main__":
    main()
