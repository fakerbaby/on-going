#! /bin/bash
export num_gpus=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0


cache_dir=${TRANSFORMERS_CACHE}

DATE=`date +%Y%m%d`

model_name_or_path=roberta-base 

debug=0
TASK_NAME=cola

# metric="accuracy"

bsz=32
max_seq_length=512
num_train_epochs=80

lr_scheduler_type="linear"

learning_rate=3e-5
weight_decay=0.1
warmup_ratio=0.06

gradient_steps=1
seed=42
logging_steps=100


# Lora
lora_r=8
lora_alpha=8
# limited model saving
save_total_limit=1
save_strategy="epoch"
save_steps=5000
eval_strategy="epoch"

# if [ "${debug}" = 1 ];
# then
#     weight_decay=0
#     max_grad_norm=1
#     max_train_samples=1000
#     # max_eval_samples=150
#     bsz=10
#     gradient_steps=1
#     num_train_epochs=5
#     max_steps=-1
#     eval_strategy='steps'
#     save_steps=100
#     report_to="wandb"
#     logging_steps=10
#     # extra_cmd="--max_train_samples ${max_train_samples} --max_predict_samples 150"
#     debug_str=".debug"
# fi

exp_name=roberta_base_finetuning
exp_name+=.ne${num_train_epochs}
exp_name+=.warm${warmup_ratio}.wd${weight_decay}.seed${seed}.${debug_str}
SAVE=../../checkpoints/glue/${TASK_NAME}/${DATE}/${exp_name}
echo "${SAVE}"
rm -rf ${SAVE}; mkdir -p ${SAVE}


# wandb env variables
YOUR_API_KEY=184780b97325730f5c5afafe9f4428c3409f3e28
export WANDB_API_KEY=$YOUR_API_KEY 
export WANDB_PROJECT=glue-${TASK_NAME} 
export WANDB_RESUME="allow"
export WANDB_WATCH="true"
export WANDB_NAME=exp_name
export WANDB_NOTES=" roberta_base"
#if you dont want sync to the cloud
# export WANDB_MODE="dryrun"
report_to="wandb"
# metric="accuracy"
# python -u \

python -m torch.distributed.launch --nproc_per_node=$num_gpus \
../../examples/run_glue.py \
--output_dir ${SAVE} \
--model_name_or_path ${model_name_or_path} \
--task_name ${TASK_NAME} \
--do_train \
--do_eval \
--max_seq_length ${max_seq_length} \
--per_device_train_batch_size ${bsz} \
--per_device_eval_batch_size ${bsz} \
--learning_rate ${learning_rate} \
--lr_scheduler_type ${lr_scheduler_type} \
--num_train_epochs ${num_train_epochs} \
--logging_steps ${logging_steps} \
--evaluation_strategy ${eval_strategy} \
--save_strategy ${save_strategy} \
--save_steps ${save_steps} \
--eval_steps ${save_steps} \
--warmup_ratio ${warmup_ratio} \
--gradient_accumulation_steps ${gradient_steps} \
--seed ${seed} \
--weight_decay ${weight_decay} \
--report_to ${report_to} \
--run_name ${exp_name} \
--disable_tqdm "True" \
--greater_is_better "True" \
--overwrite_output_dir  \
--load_best_model_at_end "yes" \
--fp16 \
--save_total_limit ${save_total_limit} \
    2>&1 | tee ${SAVE}/log.txt 


# --apply_lora \
# --lora_r ${lora_r} \
# --lora_alpha ${lora_alpha} \