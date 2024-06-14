# export TRANSFORMERS_CACHE=./cache
export HF_HOME=./cache

python3 main.py \
    --config utils/SingGraph.conf \
    --output_dir ./exp_result \
    --eval
