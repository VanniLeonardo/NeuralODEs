This is an academic raw PyTorch codebase for a Neural ODE project.

Follow these rules:
- Use raw PyTorch only. Do not use PyTorch Lightning, Ignite, or fastai.
- Keep all code device-agnostic.
- Add complete type hints to every function and method.
- Use concise Google-style docstrings.
- Use Black formatting and Ruff-compatible code.
- Do not add obvious AI-style comments.
- Use wandb for experiment tracking in experiment scripts.
- Use rich for console logging in training scripts.
- Preserve the current modular architecture: config.py, main.py, data/, models/, training/, scripts/, tests/.
- Make minimal patches. Do not refactor unrelated code.
- Do not commit, push, delete files, install packages, or change environment files unless explicitly asked.

For ANODE work:
- Implement Augmented Neural ODEs as a minimal extension of the existing ODENet.
- Keep augment_dim = 0 as the exact standard NODE baseline.
- Use augment_dim > 0 for ANODE by concatenating zero dimensions to the hidden state.
- Preserve backward compatibility for all existing NODE, MNIST, solver-ablation, and time-series code.
- Keep experiments aligned with the existing report and the ANODE paper.
- Main experiments: circles NODE vs ANODE, flow/feature visualization, missing angular slice generalization, and a lightweight MNIST NODE vs ANODE comparison.
