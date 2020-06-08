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
    --model_variant="mobilenet_v3_large_seg" \
    --eval_crop_size="1024,2048" \
    --decoder_output_stride="8,4" \
    --dataset="cityscapes" \
    --aspp_with_squeeze_and_excitation=1 \
    --image_se_uses_qsigmoid=1 \
    --image_pyramid=1 \
    --initialize_last_layer=True \
    --last_layers_contain_logits_only=False \
    --checkpoint_dir="/mrtstorage/users/rehman/experiments/c2e_encoding/loss_v1/m3l_b3_c0.5_o1/" \
    --eval_logdir="/mrtstorage/users/rehman/experiments/tmp/eval/May27_m3l"  \
    --dataset_dir="/mrtstorage/users/rehman/datasets/cityscapes/tfrecord_v3_c2e" \





