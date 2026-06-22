#!/bin/bash
echo "Running launch_vllm_server.sh to launch the vllm server in the right apptainer container and conda env"
echo $(pwd)
apptainer exec --nv /gscratch/xlab/blnewman/apptainer/hyak-container-flashattn.sif bash << EOF
source ~/.bashrc_normal
conda activate nvcc129
export CUDA_HOME=/gscratch/xlab/blnewman/miniconda3/envs/nvcc129
echo $TMPDIR
export TMPDIR=/gscratch/xlab/blnewman/tmp
export VLLM_ENGINE_READY_TIMEOUT_S=1800
cd microbiome_eval

$1
EOF
