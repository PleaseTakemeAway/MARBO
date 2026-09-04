#!/bin/bash

game_config=$1
game_dir=$2
num=$3
shift 3

mkdir -p $game_dir

for ((i=1; i<=num; i++))
do
    python3 run_random.py --config $game_config --log_save_path $game_dir/game_${i} --use_vllm $@
done