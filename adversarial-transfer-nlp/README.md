# README

## Generate adversarial examples via T3

For example, to generate adversarial examples for the Yelp dataset, run
```
python attack_classification_t3.py --dataset_path data/yelp \
--target_model bert --target_model_path models/yelp --max_seq_length 256 \
--batch_size 1 --lr 0.2 --const 10 --confidence 10  --save adv-yelp-yelp.pkl
```

## Knowledge Transfer Finetuing
For example, to finetune source model (ag) on target data domain, run
```
python run_classifier.py --data_dir ../../imdb --bert_model ../models/ag \
--task_name imdb --cache_dir pytorch_cache --do_train --do_eval --do_lower_case \
--output_dir results/ag-imdb --num_train_epochs 1 --train_batch_size 32 --steps 500
```
