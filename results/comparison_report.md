### Perplexity Comparison

| Model       | Perplexity |
| ----------- | ---------- |
| LSTM        | 7.69       |
| Transformer | 10.94      |

### Qualitative Analysis

The LSTM generated more coherent Shakespeare-style text, while the Transformer produced repetitive patterns and characters. The LSTM achieved lower perplexity than the Transformer, and lower perplexity indicates better next-character prediction. Temperature 0.5 produced more deterministic text, temperature 1.0 produced balanced text, and temperature 1.5 produced more creative but noisier text. Overall, the LSTM performed better on this training configuration.
