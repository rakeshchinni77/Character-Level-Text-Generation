# Character-Level Text Generation: LSTM vs Transformer

## Project Overview

A comprehensive implementation of character-level text generation models using PyTorch. This project compares two fundamental sequence modeling architectures—LSTM (Long Short-Term Memory) and Transformer—for generating text one character at a time. The models are trained on Shakespeare's complete works and evaluated on multiple metrics including perplexity and qualitative analysis.

## Objectives

- **Understand Core Architectures**: Implement LSTM and Transformer models from scratch to understand their inner workings
- **Build Training Pipelines**: Create robust training loops with gradient clipping, loss monitoring, and model checkpointing
- **Explore Sampling Techniques**: Implement temperature scaling for controlled text generation
- **Perform Rigorous Evaluation**: Compare models using perplexity metrics and qualitative text analysis
- **Enable Reproducibility**: Containerize the entire project with Docker for consistent execution across environments

## Key Features

- ✅ Character-level vocabulary encoding/decoding
- ✅ LSTM sequence model with configurable layers and dimensions
- ✅ Transformer encoder with multi-head self-attention
- ✅ Temperature-controlled text sampling
- ✅ Loss curve visualization and performance metrics
- ✅ Structured JSON output for generated samples
- ✅ Comprehensive comparison report
- ✅ Full Docker containerization

## Folder Structure

```
Character-Level-Text-Generation/
│
├── input/                      # Input datasets (e.g., shakespeare.txt)
├── models/                     # Saved trained model weights
├── results/                    # Output artifacts (plots, reports, samples)
├── saved/                      # Preprocessed data and vocabulary mappings
│
├── src/
│   ├── __init__.py
│   ├── prepare_data.py        # Data preprocessing and vocabulary creation
│   ├── model_lstm.py          # LSTM model implementation
│   ├── model_transformer.py   # Transformer model implementation
│   ├── train.py               # Training script
│   └── generate.py            # Text generation script
│
├── .env                        # Environment variables (local)
├── .env.example               # Environment variables template
├── .dockerignore              # Docker build context exclusions
├── .gitignore                 # Git exclusions
├── Dockerfile                 # Docker container definition
├── docker-compose.yml         # Docker Compose orchestration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup Instructions

### Local Development (Optional)

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd Character-Level-Text-Generation
   ```

2. **Create virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env as needed
   ```

### Docker Setup (Recommended)

1. **Build Docker image**:

   ```bash
   docker compose build
   ```

2. **Verify build**:
   ```bash
   docker compose run --rm app python --version
   ```

## Docker Commands

### Data Preparation

```bash
docker compose run --rm app python src/prepare_data.py
```

### Training Models

Train LSTM model:

```bash
docker compose run --rm app python src/train.py --model lstm
```

Train Transformer model:

```bash
docker compose run --rm app python src/train.py --model transformer
```

### Text Generation

Generate text from LSTM:

```bash
docker compose run --rm app python src/generate.py \
  --model lstm \
  --model_path models/lstm.pt \
  --seed_text "To be or not to be" \
  --temperature 1.0
```

Generate text from Transformer:

```bash
docker compose run --rm app python src/generate.py \
  --model transformer \
  --model_path models/transformer.pt \
  --seed_text "To be or not to be" \
  --temperature 1.0
```

## Model Architectures

### LSTM (Long Short-Term Memory)

The LSTM model uses gated recurrent mechanisms to capture long-term dependencies in sequences:

- **Embedding Layer**: Converts character indices to dense vectors
- **LSTM Layers**: Processes sequences with forget/input/output gates
- **Output Layer**: Projects hidden states to vocabulary logits

### Transformer

The Transformer model uses self-attention mechanisms for parallel sequence processing:

- **Positional Encoding**: Adds position information to embeddings
- **Multi-Head Self-Attention**: Captures dependencies across all positions
- **Feed-Forward Networks**: Applies non-linear transformations
- **Encoder Blocks**: Combines attention and feed-forward with residual connections

## Configuration

Edit `.env` to customize:

```ini
# Dataset
DATASET_PATH=input/shakespeare.txt

# Hyperparameters
LEARNING_RATE=0.001
EPOCHS=10
BATCH_SIZE=64
SEQ_LENGTH=100

# Model dimensions
LSTM_HIDDEN_DIM=256
TRANSFORMER_HEADS=4
TRANSFORMER_LAYERS=2
```

## Results & Outputs

After successful training and generation, the following artifacts are created:

- **results/loss_curves.png**: Training loss curves for both models
- **results/generated_samples.json**: Generated text samples at different temperatures
- **results/comparison_report.md**: Perplexity scores and qualitative analysis

## Future Work

- [ ] Implement beam search for more sophisticated generation
- [ ] Add attention visualization for interpretability
- [ ] Experiment with different temperature scheduling strategies
- [ ] Implement top-k and nucleus (top-p) sampling
- [ ] Add support for custom datasets beyond Shakespeare
- [ ] Optimize model inference speed
- [ ] Create web interface for interactive text generation
- [ ] Benchmark on larger datasets with GPU support

## References

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation by Jay Alammar
- [Attention Is All You Need](https://arxiv.org/abs/1706.10677) - Original Transformer paper
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Andrej Karpathy's classic post
- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)

## License

MIT License - See LICENSE file for details

---

**Author**: Machine Learning Engineer  
**Last Updated**: 2026-06-12  
**Status**: Phase 0 - Repository Setup Complete
