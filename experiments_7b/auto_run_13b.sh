#!/bin/bash
# Wait for 13B download, then run experiment
# Check every 60s if the model dir exists with all 3 shard files

MODEL_DIR="/root/autodl-tmp/._____temp/ChineseAlpacaGroup/chinese-llama-2-13b"
EXP_SCRIPT="/root/autodl-tmp/experiments_7b/exp10_13b_domain.py"
LOG="/root/autodl-tmp/auto_13b_exp.log"

echo "[$(date)] Auto-runner started, waiting for 13B model..." | tee -a $LOG

while true; do
    # Count .bin files
    BIN_COUNT=$(find "$MODEL_DIR" -name "*.bin" 2>/dev/null | wc -l)

    if [ "$BIN_COUNT" -ge 3 ]; then
        # Check if files are still being written (no .downloading suffix or recent write)
        DOWNLOADING=$(find "$MODEL_DIR" -name "*.downloading" 2>/dev/null | wc -l)
        if [ "$DOWNLOADING" -eq 0 ]; then
            echo "[$(date)] All 3 shards found, no active downloads. Running 13B experiment..." | tee -a $LOG

            # Also check for the model directory after modelscope extraction
            ACTUAL_MODEL=$(find /root/autodl-tmp -maxdepth 4 -name "config.json" -path "*13b*" 2>/dev/null | head -1)

            if [ -n "$ACTUAL_MODEL" ]; then
                echo "[$(date)] Model found at $(dirname $ACTUAL_MODEL)" | tee -a $LOG
            fi

            cd /root/autodl-tmp/experiments_7b
            /root/miniconda3/bin/python $EXP_SCRIPT 2>&1 | tee -a $LOG
            echo "[$(date)] 13B experiment complete!" | tee -a $LOG

            # Download results locally too (for the experiment report)
            cp /root/autodl-tmp/experiments_7b_results/exp10_13b_domain.json /root/autodl-tmp/experiments_7b_results/

            exit 0
        fi
    fi

    echo "[$(date)] Waiting... ($BIN_COUNT/3 shards, downloading=$DOWNLOADING)" | tee -a $LOG
    sleep 60
done
