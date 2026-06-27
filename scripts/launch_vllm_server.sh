#!/bin/bash
echo "Running launch_vllm_server.sh to launch the vllm server in the right apptainer container and conda env"
echo $(pwd)
apptainer exec --nv /gscratch/xlab/blnewman/apptainer/hyak-container-flashattn.sif bash << EOF
source ~/.bashrc_normal
conda activate nvcc129
export CUDA_HOME=/gscratch/xlab/blnewman/miniconda3/envs/nvcc129
echo $TMPDIR
# export TMPDIR=/gscratch/xlab/blnewman/tmp
# export VLLM_ENGINE_READY_TIMEOUT_S=1800
cd microbiome_eval
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    $1 && break          # Run command; exit loop if it succeeds
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Attempt $RETRY_COUNT failed. Retrying..."
    sleep 2                        # Optional delay between retries
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Command failed after $MAX_RETRIES attempts"
    exit 1
fi
EOF
# $1
# EOF