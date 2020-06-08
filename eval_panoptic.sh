set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Set up the working directories.

python deeplab/eval_panoptic.py \
    --logtostderr \
    --eval_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride="8,4" \
    --eval_crop_size="1024,2048" \
    --dataset="cityscapes" \
    --colormap_type="cityscapes" \
    --checkpoint_dir="/mrtstorage/users/rehman/experiments/c2e_encoding/loss_v1/x65_b6_c0.5_b1/" \
    --eval_logdir="/mrtstorage/users/rehman/experiments/tmp/eval/June7"  \
    --dataset_dir="/mrtstorage/users/rehman/datasets/cityscapes/tfrecord_v3_c2e" \






