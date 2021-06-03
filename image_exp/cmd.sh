# To replicate the experiment "Adversarial Transferability indicates Knowledge transferability"
# Step 1: Train source models
python3 train_clf.py -d cifar10 -a fcnet --include_list 0 1 2 3 4 5 6 7 8 9 --save_name cifar10_fcnet.pth
python3 train_clf.py -d cifar10 -a le_net --include_list 0 1 2 3 4 5 6 7 8 9 --save_name cifar10_lenet.pth
python3 train_clf.py -d cifar10 -a alexnet --include_list 0 1 2 3 4 5 6 7 8 9 --save_name cifar10_alexnet.pth
python3 train_clf.py -d cifar10 -a res_net18 --include_list 0 1 2 3 4 5 6 7 8 9 --save_name cifar10_resnet18.pth
python3 train_clf.py -d cifar10 -a res_net50 --include_list 0 1 2 3 4 5 6 7 8 9 --save_name cifar10_resnet50.pth

# Step 2: Measure Knowledge Transferabilty:
python3 transfer.py -d stl10 -a fcnet --include_list 0 1 2 3 4 5 6 7 8 9 --load_name cifar10_fcnet.pth --epoch 10 --source_classes 10 --transfer_last
python3 transfer.py -d stl10 -a le_net --include_list 0 1 2 3 4 5 6 7 8 9 --load_name cifar10_lenet.pth --epoch 10 --source_classes 10 --transfer_last
python3 transfer.py -d stl10 -a alexnet --include_list 0 1 2 3 4 5 6 7 8 9 --load_name cifar10_alexnet.pth --epoch 10 --source_classes 10 --transfer_last
python3 transfer.py -d stl10 -a res_net18 --include_list 0 1 2 3 4 5 6 7 8 9 --load_name cifar10_resnet18.pth --epoch 10 --source_classes 10 --transfer_last
python3 transfer.py -d stl10 -a res_net50 --include_list 0 1 2 3 4 5 6 7 8 9 --load_name cifar10_resnet50.pth --epoch 10 --source_classes 10 --transfer_last

# Step 3: Train target model
python3 train_clf.py -d stl10 -a res_net50 --include_list 0 1 2 3 4 5 6 7 8 9 --save_name stl10_resnet50.pth

# Attack and Transfer with FGSM[--attack] (option: fgsm/pgd):
python3 adv2know.py -d stl10 -a fcnet --include_list 0 1 2 3 4 5 6 7 8 9 --attack fgsm --eps 0.1 --load_name cifar10_fcnet.pth 
python3 adv2know.py -d stl10 -a le_net --include_list 0 1 2 3 4 5 6 7 8 9 --attack fgsm --eps 0.1 --load_name cifar10_lenet.pth
python3 adv2know.py -d stl10 -a alexnet --include_list 0 1 2 3 4 5 6 7 8 9 --attack fgsm --eps 0.1 --load_name cifar10_alexnet.pth
python3 adv2know.py -d stl10 -a res_net18 --include_list 0 1 2 3 4 5 6 7 8 9 --attack fgsm --eps 0.1 --load_name cifar10_resnet18.pth
python3 adv2know.py -d stl10 -a res_net50 --include_list 0 1 2 3 4 5 6 7 8 9 --attack fgsm --eps 0.1 --load_name cifar10_resnet50.pth

# To replicate the experiment "Knowledge Transferability indicates Adversarial transferability"
# Step 1: Train source models
python3 train_clf.py -d cifar10 -a res_net18 --include_list 0 1 8 9 --save_name 0_percent.pth 
python3 train_clf.py -d cifar10 -a res_net18 --include_list 2 1 8 9 --save_name 25_percent.pth
python3 train_clf.py -d cifar10 -a res_net18 --include_list 2 3 8 9 --save_name 50_percent.pth
python3 train_clf.py -d cifar10 -a res_net18 --include_list 2 3 4 9 --save_name 75_percent.pth
python3 train_clf.py -d cifar10 -a res_net18 --include_list 2 3 4 7 --save_name 100_percent.pth

# Step 2: Measure Knowledge Transferability
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 0_percent.pth --source_classes 4 --transfer_last 
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 25_percent.pth --source_classes 4 --transfer_last
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 50_percent.pth --source_classes 4 --transfer_last
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 75_percent.pth --source_classes 4 --transfer_last
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 100_percent.pth --source_classes 4 --transfer_last

#Step 3: Train target model
python3 train_clf.py -d stl10 -a res_net18 --include_list 1 3 4 5 --save_name target_model.pth

#Step 4: Attack and Transfer
python3 know2adv.py -d stl10 -a res_net18 --include_list 1 3 4 5 --attack pgd --eps 0.1 --load_name 0_percent.pth
python3 know2adv.py -d stl10 -a res_net18 --include_list 1 3 4 5 --attack pgd --eps 0.1 --load_name 25_percent.pth
python3 know2adv.py -d stl10 -a res_net18 --include_list 1 3 4 5 --attack pgd --eps 0.1 --load_name 50_percent.pth
python3 know2adv.py -d stl10 -a res_net18 --include_list 1 3 4 5 --attack pgd --eps 0.1 --load_name 75_percent.pth
python3 know2adv.py -d stl10 -a res_net18 --include_list 1 3 4 5 --attack pgd --eps 0.1 --load_name 100_percent.pth