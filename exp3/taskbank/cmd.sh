#!/bin/bash
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source autoencoder
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source curvature
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source denoise
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source edge2d
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source edge3d
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source keypoint2d
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source keypoint3d
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source reshade
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source rgb2depth
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source rgb2mist
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source rgb2sfnorm
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source room_layout
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source segment25d
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source segment2d
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source vanishing_point
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source segmentsemantic
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source class_1000
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source class_places
python3 tools/attack_and_transfer.py --data_dir ~/exp3/data/assets/taskonomy-sample-model-1 --attack pgd --eps 0.03 --source inpainting_whole