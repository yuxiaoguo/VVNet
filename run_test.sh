export CUDA_VISIBLE_DEVICES=0,1
python train.py --input-previous-model-path ./previous --input-training-data-path ~/datasets/VVNet/SUNCG --input-validation-data-path ~/datasets/VVNet/SUNCG --input-gpu-nums 2 --input-network VVNetAE120 --log-dir ./log --max-iters 150000 --batch-per-device 2 --output-model-path ./ckp  --phase test
