#!/bin/bash
cd "$(dirname "$0")"
for task in $TASKS; do for rep in $REPS; do for st in ${STAGES:-A B C D1 D2 D3 E}; do
  if grep -q "\"$task\" \"$rep\" \"$st\"" mil/done.txt 2>/dev/null; then continue; fi
  python3 mil_run.py $task $rep $st >> mil/loop.log 2>&1 && echo "\"$task\" \"$rep\" \"$st\"" >> mil/done.txt
done; done; done
