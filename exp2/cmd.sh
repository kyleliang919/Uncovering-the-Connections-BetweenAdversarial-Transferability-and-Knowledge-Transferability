!/bin/bash
# Train 40 source models
for i in {0..39}
do
   python3 train_clf.py --attr $i --save-model
done

# Measure knowledge transferability
for i in {0..39}
do
   python3 transfer.py --backbone-resume-root checkpoints/{$i}/backbone.pth
done

# Adversarial transferability
python3 attack_and_transfer.py