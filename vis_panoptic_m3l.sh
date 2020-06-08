set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"


# Set up the working directories.

python deeplab/vis_panoptic_tf.py \
    --logtostderr \
    --vis_split="train" \
    --model_variant="mobilenet_v3_large_seg" \
    --decoder_output_stride="8,4" \
    --vis_crop_size="1024,2048" \
    --aspp_with_squeeze_and_excitation=1 \
    --image_se_uses_qsigmoid=1 \
    --image_pyramid=1 \
    --dataset="cityscapes" \
    --colormap_type="cityscapes" \
    --checkpoint_dir="/mrtstorage/users/rehman/experiments/c2e_encoding/m3l_c2e_b12/m3l_c2e_b12_further/m3l_c2e_b12_further_full_image/" \
    --vis_logdir="/mrtstorage/users/rehman/experiments/tmp/vis/"  \
    --dataset_dir="/mrtstorage/users/rehman/datasets/cityscapes/tfrecord_v3_c2e/" \





