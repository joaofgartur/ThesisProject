#!/bin/bash

# Number of tmux sessions to create
DATASET=$1
BASE_SEED=$2
N=$3

# Loop to create N tmux sessions
for ((i=0; i<N; i++)); do
    # Calculate seed for this session
    SEED=$((BASE_SEED + i))
    
    # Create a new tmux session with a unique name
    tmux new-session -d -s "${DATASET}_${SEED}"
    
    # Construct the Python command
    PYTHON_COMMAND="python main.py -dataset ${DATASET} -configs configs/configs.cfg -all -runs 1 -seed $SEED"
    
    # Send command to the session
    tmux send-keys -t "${DATASET}_${SEED}" "$PYTHON_COMMAND" Enter
    
    # Send exit command to close tmux session after Python command completes
    tmux send-keys -t "${DATASET}_${SEED}" "exit" Enter
done
