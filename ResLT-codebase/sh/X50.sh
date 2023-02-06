source activate py3.6pt1.5

python3 ImageNetTrain_reslt.py \
  --arch resnext50_32x4d_reslt \
  --mark resnext50_reslt_bt256 \
  -dataset ImageNet \
  --data_path ../data/ImageNet \
  -b 256 \
  --epochs 200 \
  --num_works 40 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --beta 0.99 \
  --gamma 0.7 \
  --after_1x1conv \
  --num_classes 1000 
