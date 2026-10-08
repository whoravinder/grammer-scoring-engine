# Spoken Audio Grammar Scoring Solution

## What this is

This project predicts continuous grammar proficiency scores from spoken audio recordings. The dataset contains 769 training audio clips and 216 test clips, with predictions evaluated using RMSE.

The final pipeline is v5. It uses frozen `openai/whisper-medium` Layer 24 and `openai/whisper-large-v3` Layer 32 encoders. Audio is processed in 30-second windows, and embeddings are pooled using only valid frames with window-length weighting. The resulting features are standardized with `StandardScaler`, followed by separate `Ridge(alpha=1000.0)` regressions. The two model predictions are averaged 50/50 and clipped to the range `[0.0, 5.0]`.

The regression models are trained on the 732 clips identified as speech. The 37 training clips containing acoustic noise are handled separately by the noise detection step.

## Results

Cross-validation uses 15 splits from 5-fold Stratified Group K-Fold repeated with three random seeds. The grouping is based on audio `id // 10`.

- **Whisper-Small L12**: CV RMSE 0.5885, Pearson r 0.8145
- **Whisper-Medium L24**: CV RMSE 0.5628, Pearson r 0.8328
- **Whisper-Large-v3 L32**: CV RMSE 0.5674, Pearson r 0.8296
- **v5 Ensemble (Medium + Large-v3)**: CV RMSE 0.5561, Pearson r 0.8372
- **Training RMSE (732 speech clips)**: 0.4190
- **Training RMSE (all 769 clips, with acoustic noise detector applied)**: 0.4088

v5 is not the best public leaderboard submission. The v8 experiment reached 0.3748 compared with 0.3790 for v5. v5 was retained as the final pipeline because its validation improvement over the medium model was supported by the paired bootstrap analysis, it uses only MIT-licensed Whisper models, and the pipeline remains simple to reproduce and explain.

The v8 improvement was not consistent enough across the validation schemes to justify replacing v5. Its nested-CV RMSE was 0.5500 with `id//10` grouping and 0.5602 with `id//40` grouping, compared with 0.5561 and 0.5578 for v5. The bootstrap confidence intervals included zero, and the preferred ensemble weight changed between the validation schemes. v8 also introduced WavLM-large, which uses CC BY-SA 3.0.

## Submission history

<!-- SUBMISSION_LOG_START -->
| Order | Kaggle file | Version | Description | CV RMSE | CV folds | Public LB | Note |
|---|---|---|---|---|---|---|---|
| 1 | submission.csv | v1 | Whisper-small L12, Ridge alpha=1000 | 0.5885 | id//10 (5x3 SGKF) | 0.4064 | Audio SSL baseline |
| 2 | submission_v2diag.csv | v2diag | MiniLM text + fluency + GPT-2 surprisal | 0.7791 | id//10 (5x3 SGKF) | 0.5781 | Multimodal diagnostics without speech embeddings |
| 3 | submission_v2.csv | v2 | Whisper-small L12 + MiniLM + fluency + GPT-2 | 0.5895 | id//10 (5x3 SGKF) | 0.4024 | Multimodal speech and text blend |
| 4 | submission_v3.csv | v3 | Whisper-small Layer 9 + Layer 12 concat | 0.5799 | id//10 (5x3 SGKF) | 0.4275 | Layer concatenation |
| 5 | submission_v3b.csv | v3b | Whisper-small L12 + Wav2Vec2-base L8 blend | 0.5904 | id//10 (5x3 SGKF) | 0.4120 | Acoustic SSL blend |
| 6 | submission_v4.csv | v4 | Whisper-medium L24, Ridge alpha=1000 | 0.5628 | id//10 (5x3 SGKF) | 0.3936 | Encoder scaling to medium |
| 7 | submission_v5.csv | v5 | 50/50 Whisper-medium L24 + Large-v3 L32, Ridge alpha=1000 each | 0.5561 | id//10 (5x3 SGKF) | 0.3790 | Final adopted pipeline |
| 8 | submission_v6.csv | v6 | 50/50 blend of v5 Ridge + RBF-SVR (C=3.0) | 0.5550 | id//10 (5x3 SGKF) | 0.3795 | Non-linear SVR blend |
| 9 | submission_v7.csv | v7 | Average of Medium L24 + Large-v3 L32 + Medium.en L24 | n/a | id//10 (5x3 SGKF) | 0.3831 | English-only prior exploration |
| 10 | submission_v8.csv | v8 | 75% v5 + 25% WavLM-large L24, Ridge alpha=300 | 0.5507 | id//10 (5x3 SGKF) | 0.3748 | WavLM blend; rejected due to license and validation uncertainty |
<!-- SUBMISSION_LOG_END -->

The public leaderboard improved from 0.4064 for the initial Whisper-small submission to 0.3790 for v5, an absolute RMSE improvement of 0.0274. The later v8 submission achieved the best public score of 0.3748, but was not adopted as the final pipeline for the reasons described above.

## How to run

### Environment requirements

- Python 3.12
- PyTorch with CUDA support

Install PyTorch:

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining packages:

```bash
pip install -r requirements.txt
```

### Data layout

Place the competition files in `data/`:

```text
data/
├── train/
│   └── audio_*.wav
├── test/
│   └── audio_*.wav
├── train.csv
└── test.csv
```

### Running the pipeline

Run the notebook from top to bottom.

If the embedding cache is empty, the notebook extracts the required Whisper embeddings automatically. Subsequent runs reuse the cached embeddings.

The notebook can also be executed from the command line with:

```bash
jupyter nbconvert --to notebook --execute solution.ipynb
```

Measured runtime using cached embeddings is approximately 23.7 seconds.

The experiments were run on:

- NVIDIA GeForce RTX 4050 Laptop GPU
- 6.0 GiB VRAM
- Intel Core i7 CPU
- 16 logical threads
- 24 GB RAM
- Windows 11

## Reproducibility

The repeated cross-validation experiments use three fixed random seeds:

- `42`
- `123`
- `999`

The same seeds are used throughout the repeated validation experiments and regression steps.

The embedding cache separates the expensive audio feature extraction stage from the model experiments. This allows the validation and regression stages to be rerun without repeatedly processing the audio.

## What is not included and why

The competition audio files, raw transcripts, precomputed embeddings, model checkpoints, and submission files are not included in this repository.

The competition dataset is subject to the competition's "Competition Use only" restrictions and must be obtained directly through Kaggle.

The notebook extracts embeddings directly from the competition audio using the public pretrained Whisper models when `cache/embeddings/` is empty.

This keeps the competition data and derived binary files out of the repository and avoids redistributing restricted competition data.

## Model licenses

The final pipeline uses the following Whisper models:

- `openai/whisper-small` — MIT License
- `openai/whisper-medium` — MIT License
- `openai/whisper-large-v3` — MIT License
- `openai/whisper-large-v2` — MIT License
- `openai/whisper-medium.en` — MIT License

Other models were tested during development:

- `sentence-transformers/all-MiniLM-L6-v2` — Apache 2.0
- `gpt2` — Modified MIT License
- `microsoft/wavlm-large` — CC BY-SA 3.0

WavLM-large was used in the v8 experiment but is not part of the final pipeline. Its CC BY-SA 3.0 license was one of the considerations in deciding not to adopt v8.

## Repository license

The original code in this repository is released under the MIT License.

The licenses of third-party pretrained models are listed separately above. Their respective licenses continue to apply to those models and are not replaced by the repository license.

## Limitations

- The public leaderboard represents only part of the 216 test clips, so public RMSE is treated as a reference rather than the main model-selection criterion.
- Encoder layers and Ridge regularization were selected using the reported validation framework, so some model-selection bias is possible.
- The test audio has a shorter average duration than the training audio, and the final model does not explicitly adapt to this duration shift.
- Larger Whisper encoders produced lower cross-validation error in the tested configurations, but no causal claim is made about which speech properties are responsible for the improvement.
- The training RMSE is an in-sample metric and is therefore expected to be lower than the cross-validation RMSE. It is reported only as a reference.
- The 37 zero-score acoustic-noise clips are handled separately from the speech regression model using a simple audio-based detector.

## AI assistance

AI coding tools were used for implementation assistance and documentation during development. I reviewed the resulting code and understand the feature extraction, validation, model training, evaluation, and submission pipeline.
