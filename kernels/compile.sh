#!/bin/bash

PATH=/usr/local/cuda/bin:$PATH
# ARCH=compute_20  # minimum
ARCH=compute_50

nvcc -ptx JCudaVectorAddKernel.cu -arch ${ARCH} -o JCudaVectorAddKernel.ptx

