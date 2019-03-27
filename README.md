# VVNet: View-volume network for semantic scene completion from a single depth image
By **Yu-Xiao Guo**, **Xin Tong**
## Environment & Requirement
*OS*: Ubuntu-16.04, \
*Python*: 3.5, \
*TensorFlow*: 1.3.0-RC2, \
*CUDA*: 8.0, \
*CUDNN*: 6.0, \
*GPUs*: NVidia GTX TITAN XP * 2
## Setup
### Steps:
1. Install TensorFlow: `pip install tensorflow-gpu==1.3.0-rc2`
2. Compile custom ops: `cd libs && source build.sh`
3. Prepare training/test samples:
    * Download SSCNet-SUNCG training/test samples: [url](http://sscnet.cs.princeton.edu/sscnet_release/data/depthbin_eval.zip). (If someone finds the link is invalid, please ask the permission from the author of SSCNet directly)
    * Run: `cd tools && python prepare_data.py`. Please set `DATA_DIR` and `RECORD_DIR` to your local path in advance.
4. Train: `source run_training.sh`
5. Test: `source run_test.sh`
### Parameters description:
* `--input-previous-model-path`: model dir/file for fine-tune.
* `--input-training-data-path`: the dir to folder of training TFRecords
* `--input-validation-data-path`: the dir to folder of test TFRecords
* `--input-gpu-nums`: gpu nums for training
* `--input-network`: network structure to train/test, optional choices including `VVNetAE30`, `VVNetAE60`, `VVNetAE120`. If someone tends to try other models in folder [models](/models) but fails, please feel free to ping us.
* `--max-iters`: maximum iterations for training, default 150K
* `--record-iters`: saving model period per iterations, default 2K
* `--batch-per-device`: batch size per gpu, default 2
* `--output-model-path`: the dir to save trained models
* `--log-dir`: the dir to save logs 
* `--eval-platform`: the test output format. `fusion` will save test tensors with compatible mode with [SSCNet](https://github.com/shurans/sscnet) evaluation pipeline. 
* `--eval-results`: the folder to save test output
* `--phase`: the phase of `training` or `test`
