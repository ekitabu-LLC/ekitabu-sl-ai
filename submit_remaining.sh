#!/bin/bash
# Submit remaining 3 v44 jobs once testing QoS slots open
cd /scratch/alpine/hama5612/ksl-dir-2
JOBS="slurm/v44_expr5_words.sh slurm/v44_expr7.sh slurm/v44_expr7_words.sh"
for script in $JOBS; do
    while true; do
        out=$(sbatch "$script" 2>&1)
        if echo "$out" | grep -q "Submitted batch job"; then
            echo "[$(date)] Submitted: $script — $(echo $out | grep -o 'batch job [0-9]*')"
            break
        else
            echo "[$(date)] Slot full, retrying in 60s: $script"
            sleep 60
        fi
    done
done
echo "All 3 remaining jobs submitted."
