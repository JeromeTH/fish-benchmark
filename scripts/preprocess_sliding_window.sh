#!/bin/bash

for IDX in {0..7}; do
    for SUBIDX in {0..7}; do
        echo "Processing file with IDX=$IDX, SUBIDX=$SUBIDX"
        sbatch -J job_id \
            -o path.out \
            -e path.err \
            -N 1 -n 1 --get-user-env --requeue --time=infinite \
            --cpus-per-task=4 --mem=256G --partition=gpu \
            --wrap="export IDX=${IDX}; export SUBIDX=${SUBIDX}; \
                export ANNOTATOR=llama; \
                export MODEL_ID=/share/nikola/memgpt/version4/model/Llama-3.1-8B-Instruct_squad-train1k_dwiki-train1k_chatgpt_gpt4o-v7.1_cleaned_ep10_merged; \
                export PROMPT_ID=llama-v6.1; \
                export DATASET=fineweb; \
                export MANAGER=\${DATASET}-train10TB.\${IDX}.\${SUBIDX}; \
                export SUBSET=fineweb/data_ids/\${MANAGER}-ids; \
                export SAVE_DIR=fineweb/data; \
                python main.py --annotator \${ANNOTATOR} --model-id \${MODEL_ID} \
                --prompt-id \${PROMPT_ID} --manager \${MANAGER} --dataset \${DATASET} \
                --subset \${SUBSET} --format json --seed 42 --save-dir \${SAVE_DIR} --save-every 10000"
    done
done