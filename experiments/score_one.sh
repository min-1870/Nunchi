#!/bin/bash
# score one 10k run on every results-table column: switching + main (knob_eval), key collision, ambiguous English,
# extra KO>EN end; scores go to experiments/logs/ (knobs/, collision_all.json, amb_en_scores.json, end_extra_scores_all.json)
#     bash experiments/score_one.sh NAME:file
cd "$(dirname "$0")/.."
spec=$1; name=${spec%%:*}
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
P=.venv-rocm/Scripts/python
$P -u experiments/knob_eval.py $spec >> experiments/logs/knob_eval_log.txt 2>&1
$P -u experiments/collision_eval.py $spec 2>/dev/null | grep -v WARNING > experiments/logs/collision_$name.txt
$P -u experiments/amb_en_eval.py $spec >/dev/null 2>&1
$P -u experiments/end_extra.py score $spec >/dev/null 2>&1
$P - "$name" <<'PY'
import json, re, sys
n = sys.argv[1]
a = json.load(open("experiments/logs/end_extra_scores_all.json", encoding="utf-8"))
b = json.load(open("experiments/logs/end_extra_scores.json", encoding="utf-8"))
a.update({k: v for k, v in b.items() if k != "points"})
json.dump(a, open("experiments/logs/end_extra_scores_all.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
c = json.load(open("experiments/logs/collision_all.json"))
m = re.search(r"wrong (\d+)/", open(f"experiments/logs/collision_{n}.txt", encoding="utf-8").read())
c[n] = int(m.group(1)); json.dump(c, open("experiments/logs/collision_all.json", "w"), indent=0)
PY
echo "[$(date +%H:%M)] scored $name (all columns)" >> experiments/logs/overnight_log.txt
