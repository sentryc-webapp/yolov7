#!/usr/bin/env sh

python train.py  --batch-size 16 --epochs 300 --data data/ridge_100.yaml \
    --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/ridge.yaml \
    --weights yolov7x_training.pt --name yolov7x-ridge-100

python train.py  --batch-size 16 --epochs 300 --data data/ridge_300.yaml \
    --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/ridge.yaml \
    --weights yolov7x_training.pt --name yolov7x-ridge-300

python train.py  --batch-size 16 --epochs 300 --data data/ridge.yaml \
    --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/ridge.yaml \
    --weights yolov7x_training.pt --name yolov7x-ridge-500
