#!/bin/bash
# resumable queue of revision experiments (run from the repository folder)
cd "$(dirname "$0")"
run() { key="$*"; grep -qxF -- "$key" ext/done.txt 2>/dev/null && return; python3 ext_run.py "$@" >> ext/queue.log 2>&1 && echo "$key" >> ext/done.txt; }
drun() { key="deep $*"; grep -qxF -- "$key" ext/done.txt 2>/dev/null && return; python3 deep_run.py "$@" >> ext/queue.log 2>&1 && echo "$key" >> ext/done.txt; }
for r in 0 1 2 3 4; do run --task bin --mode lfo --rep $r; done
for r in 0 1 2 3; do run --task bin --mode loco --rep $r; done
for r in 0 1 2 3 4 5 6 7 8 9; do run --task bin --mode main --rep $r --shap 1; done
for r in 0 1 2 3 4 5 6 7 8 9; do drun bin $r attn; drun bin $r gru; done
for r in 0 1 2 3 4; do run --task cat --mode lfo --rep $r; done
for r in 0 1 2 3 4 5 6 7 8 9; do run --task cat --mode main --rep $r --shap 1; done
for r in 0 1 2 3 4; do drun cat $r attn; drun cat $r gru; done
for r in 0 1 2 3 4; do run --task bin --mode main --rep $r --s1 nested --base lgb --tag nested_bin; done
for r in 0 1; do
  run --task bin --mode main --rep $r --H 120 --base lgb --tag sens_bin
  run --task bin --mode main --rep $r --H 600 --base lgb --tag sens_bin
  run --task bin --mode main --rep $r --W 30 --base lgb --tag sens_bin
  run --task bin --mode main --rep $r --W 120 --base lgb --tag sens_bin
done
echo ALLDONE >> ext/queue.log
