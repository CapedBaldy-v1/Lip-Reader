#!/bin/bash
#
# Monitor training progress without keeping terminal open
#

LOG_FILE="checkpoints/final_end_to_end_run/training.log"
PID_FILE="checkpoints/final_end_to_end_run/train.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Training not running (no PID file found)"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p $PID > /dev/null 2>&1; then
    echo "Training process $PID is not running"
    rm -f "$PID_FILE"
    exit 1
fi

echo "Training is running (PID: $PID)"
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo "=========================================="
tail -f "$LOG_FILE"
