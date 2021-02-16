#!/bin/sh

#Resnext_50_32x4d
python main.py --epochs 15 --criterion symmetriccrossentropy  --optimizer Adam --image-size 512 --model-type resnext --model-name resnext50_32x4d

#Inception_resnet_v2
python main.py --epochs 15 --criterion labelsmoothloss --image-size 512 --model-type resnet --model-name inception_resnet_v2

#EfficientNet B4(600x600)
python main.py --epochs 15 --criterion bitemperedloss --image-size 600 --model-type effnet --model-name tf_efficientnet_b4_ns

#Vit_base16_384
python main.py --epochs 15 --criterion labelsmoothloss --image-size 384 --model-type vit --model-name vit_base_patch16_384

#Resnet200d
python main.py --epochs 15 --criterion labelsmoothloss  --optimizer Adam --image-size 512 --model-type resnet --model-name resnet200d
