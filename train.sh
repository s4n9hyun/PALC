#!/bin/bash

# PALC Training Script
# Easy hyperparameter configuration for PALC training

# =============================================================================
# HYPERPARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Model Configuration
MODEL_NAME="argsearch/llama-7b-sft-float32"   # Base model to use
BOTTLENECK_DIM=64                              # Bottleneck dimension (32, 64, 128, 256)

# Dataset Configuration
DATASET="Dahoas/full-hh-rlhf"                  # Dataset name
MAX_LENGTH=1024                                 # Maximum sequence length

# Training Configuration
EPOCHS=1                                       # Number of training epochs
BATCH_SIZE=4                                   # Batch size (adjust based on GPU memory)
GRAD_ACCUM=4                                   # Gradient accumulation steps
LEARNING_RATE=1e-5                            # Learning rate
LOG_STEPS=50                                   # Logging frequency (steps)
SAVE_STEPS=999999                                 # Save checkpoint every N steps
EVAL_STEPS=999999                                 # Evaluate every N steps

# Loss Configuration
# Note: SimplePreferenceLoss doesn't use beta parameter

# Hardware Configuration
DEVICE="cuda"                                   # Device (cuda/cpu)

# Output Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)               # Timestamp for unique directories
OUTPUT_BASE="./outputs"                        # Base output directory
OUTPUT_DIR="$OUTPUT_BASE/run_$TIMESTAMP"       # Timestamped output directory
MODEL_NAME_CLEAN=$(echo "$MODEL_NAME" | sed 's/\//_/g')  # Clean model name for file paths
WEIGHTS_PATH="$OUTPUT_DIR/palc_${MODEL_NAME_CLEAN}_b${BOTTLENECK_DIM}_e${EPOCHS}.pt"
RESUME_CHECKPOINT=""                           # Path to checkpoint to resume from (optional)

# =============================================================================
# SCRIPT LOGIC - DO NOT MODIFY UNLESS YOU KNOW WHAT YOU'RE DOING
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --bottleneck)
            BOTTLENECK_DIM="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --help|-h)
            echo "PALC Training Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL_NAME        Base model name (default: argsearch/llama-7b-sft-float32)"
            echo "  --bottleneck DIM          Bottleneck dimension (default: 64)"
            echo "  --epochs N                Number of epochs (default: 3)"
            echo "  --batch-size N            Batch size (default: 2)"
            echo "  --grad-accum N            Gradient accumulation steps (default: 4)"
            echo "  --lr RATE                 Learning rate (default: 1e-4)"
            echo "  --dataset DATASET         Dataset name (default: Dahoas/full-hh-rlhf)"
            echo "  --resume CHECKPOINT_PATH  Resume training from checkpoint"
            echo "  --help, -h                Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use default settings"
            echo "  $0 --epochs 5 --batch-size 4         # Custom epochs and batch size"
            echo "  $0 --model gpt2 --bottleneck 128     # Different model and bottleneck"
            echo "  $0 --resume ./outputs/checkpoint.pt  # Resume from checkpoint"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Update paths based on final config
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$OUTPUT_BASE/run_$TIMESTAMP"
MODEL_NAME_CLEAN=$(echo "$MODEL_NAME" | sed 's/\//_/g')
WEIGHTS_PATH="$OUTPUT_DIR/palc_${MODEL_NAME_CLEAN}_b${BOTTLENECK_DIM}_e${EPOCHS}.pt"

# Create output and cache directories
cd /home/ubuntu/sanghyun/research/palc
mkdir -p "$OUTPUT_BASE"
mkdir -p "$OUTPUT_DIR"
mkdir -p ".cache"

# Print configuration
print_info "Training: $MODEL_NAME | Epochs: $EPOCHS | Batch: $BATCH_SIZE | GradAccum: $GRAD_ACCUM | LR: $LEARNING_RATE"

# Activate conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate loma >/dev/null 2>&1
else
    print_warning "Conda not found, using system python"
fi

# Quick dependency check
cd /home/ubuntu/sanghyun/research
export PYTHONPATH="$(pwd):$PYTHONPATH"
python -c "import palc" 2>/dev/null || { print_error "PALC module not found"; exit 1; }

# Build training command
CMD="python train.py"
CMD="$CMD --model_name '$MODEL_NAME'"
CMD="$CMD --dataset_name '$DATASET'"
CMD="$CMD --output_dir '$OUTPUT_DIR'"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --grad_accum $GRAD_ACCUM"
CMD="$CMD --num_epochs $EPOCHS"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --max_length $MAX_LENGTH"
CMD="$CMD --bottleneck_dim $BOTTLENECK_DIM"
CMD="$CMD --save_steps $SAVE_STEPS"
CMD="$CMD --eval_steps $EVAL_STEPS"
# Beta parameter removed for SimplePreferenceLoss
CMD="$CMD --bf16"

# Add resume checkpoint if provided
if [ ! -z "$RESUME_CHECKPOINT" ]; then
    print_info "Resuming from checkpoint: $RESUME_CHECKPOINT"
    CMD="$CMD --resume_from_checkpoint '$RESUME_CHECKPOINT'"
fi

# Record training configuration as JSON
cd /home/ubuntu/sanghyun/research/palc
CONFIG_FILE="$OUTPUT_DIR/config.json"
cat > "$CONFIG_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "model_name": "$MODEL_NAME",
  "dataset": "$DATASET",
  "training": {
    "epochs": $EPOCHS,
    "batch_size": $BATCH_SIZE,
    "grad_accum": $GRAD_ACCUM,
    "learning_rate": $LEARNING_RATE,
    "max_length": $MAX_LENGTH,
    "save_steps": $SAVE_STEPS,
    "eval_steps": $EVAL_STEPS
  },
  "model_config": {
    "bottleneck_dim": $BOTTLENECK_DIM
  },
  "loss_config": {
    "type": "SimplePreferenceLoss"
  },
  "output_dir": "$OUTPUT_DIR",
  "command": "$CMD"
}
EOF

# Start training
cd /home/ubuntu/sanghyun/research/palc
eval $CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    print_success "Training completed: $WEIGHTS_PATH"
    print_info "Model saved to: $WEIGHTS_PATH"
else
    print_error "Training failed"
    exit 1
fi