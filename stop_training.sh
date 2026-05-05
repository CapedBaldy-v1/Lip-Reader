#!/bin/bash
#
# Stop training gracefully
#

PID_FILE="checkpoints/final_end_to_end_run/train.pid"

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
