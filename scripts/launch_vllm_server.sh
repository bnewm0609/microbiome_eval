apptainer run -nv apptainer/hyak-flash-attn.sif
source ~/.bashrc_normal
conda activate nvcc129
export CUDA_HOME=/gscratch/xlab/blnewman/miniconda3/envs/nvcc129
cd microbiome_eval

$1