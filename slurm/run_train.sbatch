#!/bin/bash
#SBATCH --job-name=yolo_project_final_train-vlatest.32.130
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# --- Config ---
MODEL_VERSION=${SLURM_JOB_NAME##*-} # v{Version}.{BatchSize}.{EpochNumber}
# Pour run le job :
# sbatch --parsable --job-name=yolo_project_final_train-v{Version}.{BatchSize}.{EpochNumber} slurm/run_train.sbatch

PROJECT_ROOT=$(pwd)
MAX_JOBS_TO_KEEP=3

unset VIRTUAL_ENV
export PATH="$HOME/.local/bin:$PATH" 
UV_PROJECT=$PROJECT_ROOT/uv_train_env

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
uv pip install torch papermill jupyter kaggle

export TRANSFORMERS_VERBOSITY=info

uv run python -m ipykernel install --user --name=yolo_kernel

uv run papermill "$NOTEBOOK_PATH" "$OUTPUT_NOTEBOOK" \
    -k yolo_kernel \
    -p model_version "$MODEL_VERSION" \
    -p project_dir "$PROJECT_ROOT" \
    -p job_id "$SLURM_JOB_ID"

echo "=== Notebook exécuté ==="
echo "Résultat enregistré dans : $OUTPUT_NOTEBOOK"

echo "Nettoyage des anciens jobs (Max: $MAX_JOBS_TO_KEEP)..."
EXISTING_JOBS=$(ls "$PROJECT_ROOT"/slurm/output/models/*.pth 2>/dev/null | sed -E 's/.*_([0-9]+)\.pth/\1/' | sort -n | uniq)

if [ -z "$EXISTING_JOBS" ]; then
    JOB_COUNT=0
else
    JOB_COUNT=$(echo "$EXISTING_JOBS" | wc -l)
fi

if [ "$JOB_COUNT" -gt "$MAX_JOBS_TO_KEEP" ]; then
    TO_DELETE_COUNT=$((JOB_COUNT - MAX_JOBS_TO_KEEP))
    echo "   -> $JOB_COUNT jobs trouvés. Suppression des $TO_DELETE_COUNT plus anciens."
    
    JOBS_TO_DELETE=$(echo "$EXISTING_JOBS" | head -n "$TO_DELETE_COUNT")
    
    for OLD_ID in $JOBS_TO_DELETE; do
        echo "Suppression du Job ID : $OLD_ID"
        rm -f "$PROJECT_ROOT/slurm/logs/*-$OLD_ID.out"
        rm -f "$PROJECT_ROOT/slurm/logs/*-$OLD_ID.err"
        rm -f "$PROJECT_ROOT/slurm/output/notebook/trained_out_*_${OLD_ID}.ipynb"
        rm -f "$PROJECT_ROOT/slurm/output/models/yolov1_*_${OLD_ID}.pth"
    done
else
    echo "Nombre de jobs ($JOB_COUNT) sous la limite."
fi

echo "=== Terminé ==="