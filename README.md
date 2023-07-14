# DG-SDGS
This repository provides the official PyTorch implementation of the following paper:  
**Single-domain Generalized Segmentation via Seeking the Essence from Diversity**

## Congifuration Environment
- python 3.7
- pytorch 1.8.2
- torchvision 0.9.2
- cuda 11.1

## Before start
Download the GTA5, SYNTHIA, Cityscapes, BDD-100K, Mapillary, and ACDC datasets.

## Training and test
Run like this: python train.py --gta5_data_path /dataset/GTA5 --city_data_path /dataset/cityscapes --cuda_device_id 0
