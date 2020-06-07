!/bin/bash
# Train source models
python3 train_clf.py -d cifar10 -a res_net18 --include_list 0 1 8 9 --save_name 0_percent.pth 
python3 train_clf.py -d cifar10 -a res_net18 --include_list 2 1 8 9 --save_name 25_percent.pth
python3 train_clf.py -d cifar10 -a res_net18 --include_list 2 3 8 9 --save_name 50_percent.pth
python3 train_clf.py -d cifar10 -a res_net18 --include_list 2 3 4 9 --save_name 75_percent.pth
python3 train_clf.py -d cifar10 -a res_net18 --include_list 2 3 4 5 --save_name 100_percent.pth

# Train a reference model
python3 train_clf.py -d stl10 -a res_net18 --include_list 1 3 4 5 --save_name target_model.pth

# Direct Transfer
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 0_percent.pth --transfer_last
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 25_percent.pth --transfer_last
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 50_percent.pth --transfer_last
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 75_percent.pth --transfer_last
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 100_percent.pth --transfer_last

# Fine Tuning
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 0_percent.pth
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 25_percent.pth
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 50_percent.pth
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 75_percent.pth
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 100_percent.pth

# Adversarial Transferability and Attack
python3 attack_and_transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5
