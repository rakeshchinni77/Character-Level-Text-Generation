import argparse
import pickle
import torch
import torch.nn.functional as F

from model_lstm import LSTMModel
from model_transformer import TransformerModel


def load_mappings(char_to_int_path, int_to_char_path):
    with open(char_to_int_path, "rb") as f:
        char_to_int = pickle.load(f)
    with open(int_to_char_path, "rb") as f:
        int_to_char = pickle.load(f)
    return char_to_int, int_to_char


def generate_lstm(model, seed_ids, length, temperature, device):
    model.eval()
    generated = []
    hidden = model.init_hidden(1)
    hidden = (hidden[0].to(device), hidden[1].to(device))

    # Prime with seed sequence
    if len(seed_ids) > 0:
        input_ids = torch.tensor([seed_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output, hidden = model(input_ids, hidden)
        last_id = seed_ids[-1]
    else:
        last_id = seed_ids[0]

    for _ in range(length):
        input_token = torch.tensor([[last_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, hidden = model(input_token, hidden)
        logits = logits.squeeze(0) / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        last_id = next_id

    return generated


def generate_transformer(model, seed_ids, length, temperature, device, max_seq_len=100):
    model.eval()
    generated = []
    current = seed_ids.copy()
    with torch.no_grad():
        for _ in range(length):
            input_ids = current[-max_seq_len:]
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            logits = model(input_tensor)
            next_logits = logits[-1] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)
            current.append(next_id)
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["lstm", "transformer"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seed_text", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()

    char_to_int_path = "saved/char_to_int.pkl"
    int_to_char_path = "saved/int_to_char.pkl"
    char_to_int, int_to_char = load_mappings(char_to_int_path, int_to_char_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build seed token ids
    seed_ids = []
    for ch in args.seed_text:
        if ch not in char_to_int:
            raise ValueError(f"Unknown character in seed_text: {repr(ch)}")
        seed_ids.append(char_to_int[ch])

    vocab_size = len(char_to_int)

    if args.model == "lstm":
        model = LSTMModel(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2)
    else:
        model = TransformerModel(vocab_size=vocab_size, embed_dim=128, num_heads=4, num_layers=2, ff_dim=512)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    if args.model == "lstm":
        generated_ids = generate_lstm(model, seed_ids, args.length, args.temperature, device)
    else:
        generated_ids = generate_transformer(model, seed_ids, args.length, args.temperature, device)

    generated_text = args.seed_text + ''.join(int_to_char[i] for i in generated_ids)

    print(f"Model: {args.model}\n")
    print("Seed Text:")
    print(args.seed_text + "\n")
    print("Temperature:")
    print(f"{args.temperature}\n")
    print("Generated Text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()
