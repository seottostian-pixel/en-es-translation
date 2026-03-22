# English-to-Spanish Neural Machine Translation

A Transformer-based sequence-to-sequence model for English → Spanish translation, built with PaddlePaddle.

---

## Dataset

- Source: `spa.txt` (tab-separated English/Spanish sentence pairs)
- Total pairs: ~118,964
- Split: 70% train / 15% validation / 15% test

---

## Preprocessing

1. **Proper noun handling** — words that are capitalized (non-sentence-initial), low-frequency (≤3), and appear verbatim in the Spanish side are tagged as `<properN>` and preserved through translation.
2. **Text normalization** — lowercasing, punctuation spacing (e.g. `word .`), and special character handling (`¿`, `¡`).
3. **Vocabulary** — frequency-ranked word-to-id dictionaries, capped at 30,000 tokens; actual sizes: English ~11,638, Spanish ~22,160.
4. **Sequence padding/truncation** — fixed length of 20 tokens for encoder input, 21 for decoder (to accommodate `[start]`/`[end]` tokens).

---

## Model Architecture

A single-layer Transformer encoder–decoder.

| Component | Details |
|---|---|
| Embedding | Learned token + positional embeddings |
| Embedding dim | 256 |
| Feed-forward dim | 2048 |
| Attention heads | 8 |
| Dropout | 0.2 (training) |
| Output | Linear(256 → vocab_size) |

### TransformerEncoder
- Multi-head self-attention (with padding mask)
- Feed-forward: Linear(256→2048) → ReLU → Linear(2048→256)
- Two LayerNorm layers (post-attention, post-FFN)

### TransformerDecoder
- Masked multi-head self-attention (causal mask)
- Cross-attention over encoder outputs
- Feed-forward: same structure as encoder
- Three LayerNorm layers

---

## Training

| Hyperparameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 64 |
| Base learning rate | 0.0005 |
| LR schedule | Linear warmup (500 steps) + cosine annealing |
| Optimizer | Adam |
| Gradient clipping | 1.0 |
| Loss | Cross-entropy (masked) |

Training loss dropped from ~97 → ~4.2 over 10 epochs. Validation loss stabilized around 13 after epoch 6.

---

## Evaluation

Greedy decoding on the test set (batch size 256):

| Metric | Score |
|---|---|
| BLEU | **43.52** |
| 1-gram precision | 72.6% |
| 2-gram precision | 50.0% |
| 3-gram precision | 36.7% |
| 4-gram precision | 27.3% |
| Brevity Penalty | 0.9963 |

---

## Inference

- Proper nouns unknown to the vocabulary are detected at inference time and passed through untranslated via the `<properN>` tagging mechanism.
- Multi-sentence input is split on `.`, `!`, `?` and translated sentence-by-sentence.
- Spanish paired punctuation (`¿`/`?`, `¡`/`!`) is automatically completed in the output.

---

## Saved Artifacts

| File | Contents |
|---|---|
| `transformer_model_final.pdparams` | Model weights |
| `transformer_config.json` | Hyperparameters and special tokens |
| `transformer_vocab.pkl` | `word2id` / `id2word` dictionaries for both languages |
| `transformer_test_dataset.pkl` | Tokenized test set |
