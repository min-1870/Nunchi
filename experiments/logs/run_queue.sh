#!/bin/bash
# training queue (log 25, from 2026-11-04): trains the configs listed in a queue file one by one, a line
# "CONFIG NAME" each (lines can be added while it runs), scores each run on the GPU as soon as it is trained (dev),
# skips runs already scored. Status and console go beside the queue file: QUEUE_status.txt, QUEUE_console.txt
# (for tr5_queue.txt, the default: tr5_status.txt, tr5_console.txt).
#     bash experiments/logs/run_queue.sh [experiments/logs/QUEUE_queue.txt]
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
P=.venv-rocm/Scripts/python
Q=${1:-experiments/logs/tr5_queue.txt}
base=${Q%_queue.txt}
L=${base}_status.txt
C=${base}_console.txt
F='offload-arch\|^\[stderr\] *$'
$P experiments/logs/keep_awake.py & AWAKE=$!
trap "kill $AWAKE 2>/dev/null" EXIT
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $L; }
n=1
while line=$(sed -n "${n}p" $Q) && [ -n "$line" ]; do
  cfg=${line%% *}; name=${line#* }
  say "$name: training ($cfg)"
  $P -u experiments/run.py experiments/configs/$cfg.json 2>&1 | grep --line-buffered -v "$F" | tee -a $C
  if [ "${PIPESTATUS[0]}" != 0 ]; then say "$name: training failed"; n=$((n+1)); continue; fi
  if [ -f experiments/logs/systems/$name.json ]; then say "$name: already scored"; else
    say "$name: scoring"
    $P -u experiments/system_eval.py $name=$name 2>&1 | grep --line-buffered -v "$F" | tee -a $C
  fi
  say "$name: done"
  n=$((n+1))
done
say "=== queue empty"
