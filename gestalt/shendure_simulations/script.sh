#!/bin/sh

cd /home/jfeng2/gestaltamania/gestalt/

OVERWRITE="0"

FILE="shendure_simulations/_output/{OUTDIR}/model_data.pkl"
echo $FILE
# Check if file exists already or overwrite is allowed
if [ ! -f "$FILE" ] || [ "$OVERWRITE" = "1" ]; then
    srun -p restart,matsen_e,campus \
        python simulate_estimators.py \
        --model-seed {model_seed} \
        --sampling-rate 0.05 \
        --min-leaves {min_leaves} \
        --max-leaves 70 \
        --time {time} \
        --birth-lambda 1.4 \
        --data-seed {data_seed} \
        --lasso {lasso} \
        --ridge {ridge} \
        --out-folder shendure_simulations/_output/{OUTDIR}
else
    echo "File exists already"
fi
