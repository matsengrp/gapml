#!/bin/sh

cd /home/jfeng2/gestaltamania/gestalt/

OVERWRITE="0"

FILE="simulation_multifurc/_output/{OUTDIR}/estimators_multifurc.csv"
echo $FILE
# Check if file exists already or overwrite is allowed
if [ ! -f "$FILE" ] || [ "$OVERWRITE" = "1" ]; then
    srun -p restart,matsen_e,campus --cpus-per-task 3 python simulate_estimators.py \
        --model-seed {model_seed} \
        --data-seed {data_seed} \
        --time {time} \
	--min-leaves {min_leaves} \
	--sampling-rate 0.8 \
	--max-leaves {max_leaves} \
        --variance-target-lam {variance} \
	--max-iters 3000 \
	--use-parsimony \
        --num-jumbles 1 \
	--log-barr 0.00001 \
        --out-folder simulation_multifurc/_output/{OUTDIR}
else
    echo "File exists already"
fi
