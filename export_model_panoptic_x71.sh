set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"


# Set up the working directories.

python deeplab/export_model_panoptic.py \
    --model_variant="xception_71" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --num_classes=19 \
    --decoder_output_stride="8,4" \
    --checkpoint_path="/mrtstorage/users/rehman/experiments/x_71_c500_o10_b8_rgb_0-255_encoding_imagenet_pretrained2/model.ckpt-150000" \
    --export_path="/mrtstorage/users/rehman/experiments/x_71_c500_o10_b8_rgb_0-255_encoding_imagenet_pretrained2/frozen_graph/frozen_inference_graph.pb"  \




