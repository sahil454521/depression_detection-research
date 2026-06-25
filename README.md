# Adaptive Multimodal Depression Detection (Partial)

This repo contains a partial TensorFlow/Keras implementation of the architecture up to the adaptive multimodal fusion layer, plus three prediction heads.

## Scope
- Encoder stubs for text, EEG, wearables, and audio/video
- Adaptive multimodal fusion with availability-aware gating and cross-attention
- Three heads: binary classification, severity regression, multi-label symptoms
- Cross-dataset learning scaffolding: domain adaptation, transfer learning, LODO, multi-center, cross-cultural

Out of scope: data loaders, fairness regularizer, XAI, federated learning loop.

## Cross-dataset priorities
- Domain adaptation: placeholder GRL and discriminator wiring
- Transfer learning: freeze/unfreeze helpers for encoder stacks
- LODO validation: split and evaluation stubs
- Multi-center evaluation: per-center aggregation stub
- Cross-cultural benchmarking: per-culture aggregation stub

Stubs are in:
- [src/training/domain_adaptation.py](src/training/domain_adaptation.py)
- [src/training/transfer_learning.py](src/training/transfer_learning.py)
- [src/evaluation/lodo.py](src/evaluation/lodo.py)
- [src/evaluation/multicenter.py](src/evaluation/multicenter.py)
- [src/evaluation/cross_cultural.py](src/evaluation/cross_cultural.py)

## Quickstart
1. Create a virtual environment and install dependencies:
   - `pip install -r requirements.txt`
2. Run a smoke test:
   - `python -m src.scripts.smoke_test`
3. (Optional) Run LODO stub:
   - `python -m src.scripts.lodo_eval_stub`

## Notes
- Shapes and sizes are configured in `src/config.py`.
- The implementation uses placeholder encoders intended to be replaced with real backbones later.
