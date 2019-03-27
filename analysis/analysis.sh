#!/bin/bash
# call me with command: bash analysis.sh

START_ITERS=130000
FINISH_ITERS=150000
SEPARATE_ITERS=2000

CURRENT_PATH=$(dirname $(readlink -f "$0"))
echo "find CURRENT_PATH in path: ${CURRENT_PATH}"
CKP_DIR="$(cd ${CURRENT_PATH}/../ckp && pwd)"
echo "assign the model checkpoint dir in path: ${CKP_DIR}"

cd ${CURRENT_PATH}
for ((ITER=${START_ITERS}; ITER<=${FINISH_ITERS}; ITER+=${SEPARATE_ITERS}))
do
((MODEL_ITER=ITER-1))
MODEL_PATH=$(printf "${CKP_DIR}/model_iter%06d" ${MODEL_ITER})
if [ -f "${MODEL_PATH}.ckpt.meta" ]; then
echo "start evaluate the model: ${MODEL_PATH}"
(cd .. && python main.py --phase=test --ckp_dir=${MODEL_PATH}.ckpt > tmp && rm tmp)
fi
done

python ./statistics.py --criterion --logdir=eval_results
