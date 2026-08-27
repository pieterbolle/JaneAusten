#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pieter.bolle@student.kuleuven.be
#SBATCH --job-name=austen_nlp_70B
#SBATCH --cluster=wice           # Target the WICE cluster
#SBATCH --partition=gpu_h100          # Request the GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16       # Meer CPU cores voor data-sharding
#SBATCH --mem=180G            # Meer systeemgeheugen
#SBATCH --time=00:45:00          # Training zal iets langer duren, 2 uur is veilig
#SBATCH --output=logs_austen_70B.%j.out
#SBATCH --account=lp_edu_maibi_anndl

# --- VERWIJS HUGGINGFACE NAAR DE SCRATCH PARTITIE ---
export HF_HOME=$VSC_SCRATCH/huggingface_cache
# Voor oudere versies van transformers (optioneel, voor de zekerheid):
export TRANSFORMERS_CACHE=$VSC_SCRATCH/huggingface_cache
export HF_TOKEN="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"

export TMPDIR=$VSC_SCRATCH/pip_tmp
mkdir -p $TMPDIR
export TRITON_CACHE_DIR=$VSC_SCRATCH/triton_cache

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_TIMEOUT=7200

# Modules laden (afhankelijk van specifieke VSC-architectuur)
module purge                            # Schoon eventuele conflicterende modules op
module load cluster/wice/gpu
module load foss/2022b                  # 1. Laad de hoofdtoolchain (bevat GCCcore-12.2.0)
module load Python/3.10.8-GCCcore-12.2.0 # 2. Laad Python 3.10
module load CUDA/12.1.1                 # 3. Laad CUDA voor de H100 GPU

# Virtuele omgeving activeren
cd /vsc-hard-mounts/leuven-data/381/vsc38154/JaneAusten/
source venv/bin/activate

rm -rf $TMPDIR

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-not set}"
nvidia-smi -L

# Stap 2: Model trainen
accelerate launch --multi_gpu --num_processes=4 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no 2_train_llama70B.py --smoke-test
