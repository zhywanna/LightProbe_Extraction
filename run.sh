export DATA_ROOT=../data/NeRF/nerf_synthetic/ 
export CKPT_ROOT=./ckpt 
export SCENE=drums
export CONFIG_FILE=./config/proj

python -m extraction \
    --train_dir $CKPT_ROOT/pytorch  \
    --config $CONFIG_FILE \
    --output ./output/tree.npz \
    --projection_samples 100 \
    --radius 1.3 \
    --model "nerfpp"
    --is_nerfpp_ckpt \
    # --is_jaxnerf_ckpt