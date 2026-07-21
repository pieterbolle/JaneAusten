#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pieter.bolle@student.kuleuven.be
#SBATCH --job-name=austen_nlp_8B
#SBATCH --cluster=wice           # Target the WICE cluster
#SBATCH --partition=gpu_h100          # Request the GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4       # Meer CPU cores voor data-sharding
#SBATCH --mem=40G            # Meer systeemgeheugen
#SBATCH --time=01:45:00          # Training zal iets langer duren, 2 uur is veilig
#SBATCH --output=logs_austen_8B.%j.out
#SBATCH --account=lp_edu_maibi_anndl

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- VERWIJS HUGGINGFACE NAAR DE SCRATCH PARTITIE ---
export HF_HOME=$VSC_SCRATCH/huggingface_cache
# Voor oudere versies van transformers (optioneel, voor de zekerheid):
export TRANSFORMERS_CACHE=$VSC_SCRATCH/huggingface_cache
export HF_TOKEN="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"

export TMPDIR=$VSC_SCRATCH/pip_tmp
mkdir -p $TMPDIR
export TRITON_CACHE_DIR=$VSC_SCRATCH/triton_cache

# Modules laden (afhankelijk van specifieke VSC-architectuur)
module purge                            # Schoon eventuele conflicterende modules op
module load cluster/wice/gpu
module load foss/2022b                  # 1. Laad de hoofdtoolchain (bevat GCCcore-12.2.0)
module load Python/3.10.8-GCCcore-12.2.0 # 2. Laad Python 3.10
module load CUDA/12.1.1                 # 3. Laad CUDA voor de H100 GPU

# Virtuele omgeving activeren
cd /vsc-hard-mounts/leuven-data/381/vsc38154/JaneAusten/
source venv/bin/activate
pip install --upgrade pip
pip install wheel setuptools ninja
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash_attn --no-build-isolation --no-cache-dir

rm -rf $TMPDIR

# Stap 2: Model trainen
python3 3_evaluate.py  --model_id "meta-llama/Llama-3.1-8B-Instruct" --adapter_dir "$VSC_SCRATCH/JaneAusten/fine_tuned_austen_8b"
