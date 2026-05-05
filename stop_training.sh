#!/bin/bash
#
# Stop training gracefully
#

CHECKPOINT_DIR="${1:-checkpoints/final_scratch_run}"
PID_FILE="$CHECKPOINT_DIR/train.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No training process found"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ps -p $PID > /dev/null 2>&1; then
    echo "Stopping training process $PID..."
    kill -SIGTERM $PID
    sleep 2
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "Force killing..."
        kill -9 $PID
    fi
    
    rm -f "$PID_FILE"
    echo "Training stopped"
else
    echo "Process $PID not running"
    rm -f "$PID_FILE"
fi
