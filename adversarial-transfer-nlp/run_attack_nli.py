import os

# for ESIM target model
# command = 'python attack_nli.py --dataset_path data/snli ' \
#           '--target_model esim --target_model_path ESIM/data/checkpoints/SNLI/best.pth.tar ' \
#           '--word_embeddings_path ESIM/data/preprocessed/SNLI/worddict.pkl ' \
#           '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /scratch/jindi/tf_cache' \
#             '--output_dir results/snli_esim'

# for InferSent target model
# command = 'python attack_nli.py --dataset_path data/snli ' \
#           '--target_model infersent ' \
#           '--target_model_path /scratch/jindi/adversary/BERT/results/SNLI ' \
#           '--word_embeddings_path /data/medg/misc/jindi/nlp/embeddings/glove.840B/glove.840B.300d.txt ' \
#           '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /scratch/jindi/tf_cache ' \
#           '--output_dir results/snli_infersent'

# for BERT target model
# command = 'python attack_nli.py --dataset_path data/snli ' \
#           '--target_model bert ' \
#           '--target_model_path /scratch/jindi/adversary/BERT/results/SNLI ' \
#           '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path /scratch/jindi/adversary/cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /scratch/jindi/tf_cache ' \
#           '--output_dir results/snli_bert'

# command = 'python attack_nli.py --dataset_path data/snli ' \
#           '--target_model roberta ' \
#           '--target_model_path /home/boxin/InfoBERT/ANLI/ib-roberta-large-anli-part-sl128-lr1e-5-bs64-ts-1-ws1000-wd1e-5-seed42-beta1e-2-version3 ' \
#           '--counter_fitting_embeddings_path counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path use_cache ' \
#           '--output_dir results/roberta'

# command = 'python attack_nli.py --dataset_path data/snli ' \
#           '--target_model infobert ' \
#           '--target_model_path /home/boxin/InfoBERT/ANLI/ib-bert-large-uncased-anli-part-sl128-lr2e-5-bs64-ts-1-ws1000-wd1e-5-seed42-beta0-version2 ' \
#           '--counter_fitting_embeddings_path counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path use_cache ' \
#           '--output_dir results/infobert'

os.system(command)