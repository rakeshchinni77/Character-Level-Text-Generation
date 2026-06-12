import os
import json
import pickle
import torch
import torch.nn.functional as F
from model_lstm import LSTMModel
from model_transformer import TransformerModel


def generate_sample(model, seed_text, char_to_int, int_to_char, length, temperature, device, model_name):
    """Generate a sample from the model given a seed text and temperature."""
    model.eval()
    
    # Convert seed text to indices
    seed_indices = [char_to_int.get(c, 0) for c in seed_text]
    
    # Start generation
    generated = seed_text
    current_indices = seed_indices
    
    with torch.no_grad():
        for _ in range(length - len(seed_text)):
            # Prepare input
            x = torch.tensor([current_indices], dtype=torch.long).to(device)
            
            # Forward pass
            if model_name == "lstm":
                hidden = model.init_hidden(1)
                hidden = (hidden[0].to(device), hidden[1].to(device))
                output, hidden = model(x, hidden)
            else:
                output = model(x)
            
            # Get logits for the last character
            logits = output[-1, :] / temperature
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=0)
            
            # Sample next character
            next_idx = torch.multinomial(probabilities, 1).item()
            
            # Add to generated text
            generated += int_to_char.get(next_idx, "?")
            current_indices.append(next_idx)
            
            # Keep only the last seq_length characters for next iteration
            if len(current_indices) > 100:
                current_indices = current_indices[-100:]
    
    return generated


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load character mappings
    with open(os.path.join("saved", "char_to_int.pkl"), "rb") as f:
        char_to_int = pickle.load(f)
    
    with open(os.path.join("saved", "int_to_char.pkl"), "rb") as f:
        int_to_char = pickle.load(f)
    
    vocab_size = len(char_to_int)
    
    # Initialize structure
    results = {
        "lstm": {
            "temperature_0.5": [],
            "temperature_1.0": [],
            "temperature_1.5": []
        },
        "transformer": {
            "temperature_0.5": [],
            "temperature_1.0": [],
            "temperature_1.5": []
        }
    }
    
    # Seed texts
    seed_texts = ["To be", "KING"]
    temperatures = [0.5, 1.0, 1.5]
    sample_length = 200
    
    # Load LSTM model
    lstm_model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        n_layers=2,
    )
    lstm_model.load_state_dict(torch.load(os.path.join("models", "lstm.pt"), map_location=device))
    lstm_model.to(device)
    
    # Load Transformer model
    transformer_model = TransformerModel(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=512,
    )
    transformer_model.load_state_dict(torch.load(os.path.join("models", "transformer.pt"), map_location=device))
    transformer_model.to(device)
    
    # Generate samples for LSTM
    for temp in temperatures:
        temp_key = f"temperature_{temp}"
        for seed in seed_texts:
            sample = generate_sample(
                lstm_model, seed, char_to_int, int_to_char, sample_length, temp, device, "lstm"
            )
            results["lstm"][temp_key].append(sample)
    
    # Generate samples for Transformer
    for temp in temperatures:
        temp_key = f"temperature_{temp}"
        for seed in seed_texts:
            sample = generate_sample(
                transformer_model, seed, char_to_int, int_to_char, sample_length, temp, device, "transformer"
            )
            results["transformer"][temp_key].append(sample)
    
    # Ensure output directory exists
    os.makedirs("results", exist_ok=True)
    
    # Save to JSON
    output_path = os.path.join("results", "generated_samples.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print("Generated Samples Saved:")
    print(output_path)


if __name__ == "__main__":
    main()
