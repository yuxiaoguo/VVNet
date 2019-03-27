#!/bin/sh
source activate tf-py35
mkdir ops/build && cd ops/build && cmake .. && make && make install && cd ../.. && rm -rf ops/build
