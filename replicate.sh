# prepare fp32 model for cifar10 classification
cd code
python prepare-fp32-model.py \
  --mode train --ema-epoch 40 --num-workers 8 \
  --work-dir runs/model-fp32

# copy the last checkpoint for quantization
cd ..
CKPT_PATH=$(cat runs/model-fp32/last_checkpoint)
cp $CKPT_PATH resource/model-fp32.pth
unset CKPT_PATH
# clean up runs directory to save space
rm -r runs/model-fp32
