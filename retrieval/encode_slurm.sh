

JOBSCRIPTS=encode_cc_scripts
mkdir -p ${JOBSCRIPTS}

queue=learnfair
SHARDS="0 1 2 3 4 5 6 7 8 9"

for shard_id in $SHARDS
    do
    SWEEP_NAME=encode_cc_${shard_id}
    JNAME=${SWEEP_NAME}
    SCRIPT=${JOBSCRIPTS}/run.${JNAME}.sh
    SLURM=${JOBSCRIPTS}/run.${JNAME}.slrm

    echo "#!/bin/sh" > ${SCRIPT}
    echo "#!/bin/sh" > ${SLURM}
    echo "#SBATCH --job-name=$JNAME" >> ${SLURM}
    echo "#SBATCH --output=/checkpoint/xwhan/jobs/${JNAME}.out" >> ${SLURM}
    echo "#SBATCH --error=/checkpoint/xwhan/jobs/${JNAME}.err" >> ${SLURM}
    echo "#SBATCH --mail-user=xwhan@fb.com" >> ${SLURM}
    echo "#SBATCH --mail-type=none" >> ${SLURM}
    echo "#SBATCH --partition=$queue" >> ${SLURM}
    echo "#SBATCH --signal=USR1@120" >> ${SLURM}
    echo "#SBATCH --mem=500G" >> ${SLURM}
    echo "#SBATCH --time=32:00:00" >> ${SLURM}
    echo "#SBATCH --nodes=1" >> ${SLURM}
    echo "#SBATCH --gres=gpu:8" >> ${SLURM}
    echo "#SBATCH --cpus-per-task=80" >> ${SLURM}
    echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
    echo "srun sh ${SCRIPT}" >> ${SLURM}
    echo "echo \$SLURM_JOB_ID >> jobs" >> ${SCRIPT}
    echo "{ " >> ${SCRIPT}
    echo "nvidia-smi" >> ${SCRIPT}
    echo CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_embed.py --do_predict --predict_batch_size 2000 --model_name bert-base-uncased --predict_file /checkpoint/xwhan/cc_head_lm300_psg100/cc_seg${shard_id}.csv --init_checkpoint logs/08-05-2020/baseline_v0_fixed-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-valbsz3000-sharedTrue-multi1-schemenone/checkpoint_best.pt --embed_save_path /checkpoint/xwhan/cc_head_lm300_psg100/index/cc_seg${shard_id}.npy --fp16 --max_c_len 300 --num_workers 20 >> ${SCRIPT}
    echo "kill -9 \$\$" >> ${SCRIPT}
    echo "} & " >> ${SCRIPT}
    echo "child_pid=\$!" >> ${SCRIPT}
    echo "trap \"echo 'TERM Signal received';\" TERM" >> ${SCRIPT}
    echo "trap \"echo 'Signal received'; if [ \"\$SLURM_PROCID\" -eq \"0\" ]; then sbatch ${SLURM}; fi; kill -9 \$child_pid; \" USR1" >> ${SCRIPT}
    echo "while true; do     sleep 1; done" >> ${SCRIPT}
done


for shard_id in $SHARDS
    do
        sbatch ./encode_cc_scripts/run.encode_cc_$shard_id.slrm &
done