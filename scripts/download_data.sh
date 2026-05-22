BASEPATH=$(pwd)

# mkdir -p data

# MedQA
# https://huggingface.co/datasets/bigbio/med_qa
mkdir -p data/med_qa
cd data/med_qa
wget https://huggingface.co/datasets/bigbio/med_qa/resolve/main/data_clean.zip
unzip data_clean.zip
rm data_clean.zip
mv data_clean/* .
rm -r data_clean

# Disease classification (GMRepo)
cd $BASEPATH
mkdir -p data/gmrepo/hf_dataset/
cd data/gmrepo/hf_dataset/
wget https://huggingface.co/datasets/microbiome-fm/gmrepo_eval/resolve/main/gmrepo_eval.jsonl

# Microbiome Reasoning
# downloaded from huggingface: https://huggingface.co/datasets/Eubiota/Microbiome-Reasoning
cd $BASEPATH
mkdir -p data/eubiota_microbiome_reasoning/

# Microbiome LitQA





