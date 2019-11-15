#!/bin/sh
#source activate tf1.3.0rc2-py35
#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
#export PATH=/usr/local/cuda-8.0/bin:$PATH
#export TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#export TF_LINK=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
source activate tf-py35
mkdir ops/build && cd ops/build && cmake .. && make && make install && cd ../.. && rm -rf ops/build
