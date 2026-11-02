"""Run a set of training runs from a config, one after another, then the evaluations it names.

    python -u experiments/run.py experiments/configs/<name>.json

Config (JSON):
  "steps"  : training steps per run (10000 = a short screening run, 51000 = long)
  "base"   : environment shared by every run (data shares, pools, recipe); see nunchi/sentence_tagger.py
             and nunchi/stream_tagger.py for the NUNCHI_* variables
  "runs"   : [[name, model file under models/, {extra environment}], ...]; a run whose model exists and
             has no unfinished checkpoint is skipped, an unfinished one continues from its checkpoint
  "evals"  : scripts in experiments/ to run on all models afterwards, each given NAME:file arguments
Progress: experiments/logs/runs_log.txt (stages) and experiments/logs/<model>_train_log.txt (steps, ETA).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

S = Path(__file__).parent
ROOT = S.parent
PY = sys.executable
LOG = S / "logs" / "runs_log.txt"
LOG.parent.mkdir(exist_ok=True)
if sys.platform == "win32":  # no idle sleep while this runs
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


cfg_path = Path(sys.argv[1])
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
steps = int(cfg["steps"])
env0 = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONWARNINGS": "ignore", "PYTHONIOENCODING": "utf-8"}
say(f"=== {cfg_path.stem}: {len(cfg['runs'])} runs, {steps} steps each")
done = []
for i, (name, f, extra) in enumerate(cfg["runs"], 1):
    model, ckpt = ROOT / "models" / f, ROOT / "models" / f"{f}.ckpt"
    if model.exists() and not ckpt.exists():
        say(f"--- {i}/{len(cfg['runs'])} {name}: done before")
        done.append(f"{name}:{f}")
        continue
    say(f"=== {i}/{len(cfg['runs'])} {name} {extra or '(base)'} -> {f}")
    env = {**env0, **cfg["base"], **extra, "NUNCHI_OUT": str(model), "NUNCHI_SAVE_LAST": "1", "NUNCHI_RESUME": "1"}
    t = time.time()
    r = subprocess.run([PY, "-u", "-m", "nunchi.stream_tagger", "train", str(S / "logs" / f"{Path(f).stem}_train_log.txt"),
                        str(steps), str(max(steps // 8, 1000)), "1000", "1024", "3"], env=env, cwd=ROOT)
    if r.returncode:
        say(f"--- {i}/{len(cfg['runs'])} {name}: failed (exit {r.returncode})")
        continue
    ckpt.unlink(missing_ok=True)
    say(f"--- {i}/{len(cfg['runs'])} {name}: trained in {(time.time() - t) / 60:.1f} min")
    done.append(f"{name}:{f}")
for ev in cfg.get("evals", []):
    say(f"=== eval {ev}")
    args = [f"{cfg_path.stem}"] if ev in ("short_eval", "tr_eval") else []
    subprocess.run([PY, "-u", str(S / f"{ev}.py"), *args, *done], env=env0, cwd=ROOT)
say(f"=== {cfg_path.stem} done")
