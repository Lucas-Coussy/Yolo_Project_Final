#!/bin/bash
#SBATCH --job-name=yolo_project_final_train
#SBATCH --output=slurm/logs/yolo_project_final_train-%x-%j.out
#SBATCH --error=slurm/logs/yolo_project_final_train-%x-%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# --- Config ---
MODEL_VERSION="v1.32.130" # vVersion.BatchSize.EpochNumber
PROJECT_ROOT=$(pwd)

unset VIRTUAL_ENV
export PATH="$HOME/.local/bin:$PATH" 
UV_PROJECT=$PROJECT_ROOT/.uv_train_env

NOTEBOOK_PATH="$PROJECT_ROOT/test_CNN_pytorch-pretrained_backbone.ipynb"
OUTPUT_NOTEBOOK="$PROJECT_ROOT/slurm/output/notebook/trained_out_${MODEL_VERSION}_${SLURM_JOB_ID}.ipynb"

echo "=== Informations SLURM ==="
echo "Racine  : $PROJECT_ROOT"
echo "Version du modèle : $MODEL_VERSION"
echo "Job ID : $SLURM_JOB_ID"
echo "Noeud : $SLURM_NODELIST"
echo "GPU(s) alloué(s) : $CUDA_VISIBLE_DEVICES"
echo "--------------------------"

# --- Initialisation uv ---
echo "=== Préparation de l'environnement uv ==="

# Crée un répertoire de travail isolé pour ce job
mkdir -p "$UV_PROJECT"
cd "$UV_PROJECT"

# Initialise le projet s'il n'existe pas encore
if [ ! -f "pyproject.toml" ]; then
  uv init --quiet
  uv venv
fi

uv pip install -r "$PROJECT_ROOT/slurm/requirements.txt"
uv pip install torch papermill jupyter

export TRANSFORMERS_VERBOSITY=info

uv run python -m ipykernel install --user --name=yolo_kernel

uv run papermill "$NOTEBOOK_PATH" "$OUTPUT_NOTEBOOK" \
    -k yolo_kernel \
    -p model_version "$MODEL_VERSION"
    -p project_dir "$PROJECT_ROOT"

echo "=== Notebook exécuté ==="
echo "Résultat enregistré dans : $OUTPUT_NOTEBOOK"
