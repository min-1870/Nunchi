# Nunchi — working notes for agents

Windows IME helper that picks Korean or English per word from raw 두벌식 keystrokes. Model code in `nunchi/`,
experiments in `experiments/`, the app (Rust) in `app/` is worked on separately: leave it alone.

Read first: `docs/EXPERIMENTS.md` (state, open problems, next steps, log) and `docs/EVALUATION.md` (sets, how to
judge a difference). Models by name: `models/registry.json` + `nunchi/models.load`.

## Environment
- Windows: `.venv-rocm/Scripts/python` (ROCm, AMD RX 9060 XT) — where training runs. WSL `Ubuntu-ml` (`/root/venv-rocm`,
  `source` its activate first) trains at the same speed after the fixes in docs/EXPERIMENTS.md (Speed); no reason to move.
- `data/` and `models/*` are not in git; rebuild data with the commands in docs/EXPERIMENTS.md.
- Training: `python -u experiments/run.py experiments/configs/<name>.json` (10k steps ~15 min, 51k ~75 min).
  `NUNCHI_PROFILE=1` logs where a step's time goes (queue wait, copy to GPU, compute).
- GPU memory is capped at 70% in training (`NUNCHI_VRAM`); a full card once froze the PC.

## Rules
- Long jobs write a live progress log (step, %, ETA) under `experiments/logs/`; report progress.
- Questions ("어때?", "~할까?") are not go-aheads: answer and wait. Commit only when asked.
- Judge changes per category with denominators and noise (seed difference, sqrt(N)), and with bias curves;
  never by one total. Compare runs only at the same step budget.
- Data, loss and decision policy before scale (bigger / longer models).
- Comparisons with other IMEs are local only: nothing about them goes into the repository or commit messages.
