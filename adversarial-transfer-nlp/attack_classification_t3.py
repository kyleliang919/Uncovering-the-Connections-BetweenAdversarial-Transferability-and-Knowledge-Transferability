import argparse
import os
import numpy as np
import dataloader
from CW_attack import CarliniL2
from train_classifier import Model
import criteria
import random

import tensorflow as tf
import tensorflow_hub as hub

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig

from util import args

class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores

def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                # print(input_ids)
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


    def text_pred_logit(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        logits_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            # with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask)
            # probs = nn.functional.softmax(logits, dim=-1)
            logits_all.append(logits)

        return torch.cat(logits_all, dim=0)

    def attack(self, text_data, labels, cw, batch_size=32):
        untargeted_success = 0
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        logits_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            batch = {}
            batch_add_start = batch['add_start'] = []
            batch_add_end = batch['add_end'] = []
            for i, seq in enumerate(input_mask):
                batch['add_start'].append(1)
                batch['add_end'].append(torch.sum(seq).cpu().item() - 1)
            data = batch['seq'] = input_ids
            batch['input_mask'] = input_mask
            batch['segment_ids'] = segment_ids

            attack_targets = torch.tensor(labels).cuda().unsqueeze(0)

            # prepare attack
            input_embedding = self.model.bert.embeddings.word_embeddings(data)
            cw_mask = np.zeros(input_embedding.shape).astype(np.float32)

            cw_mask = torch.from_numpy(cw_mask).float().cuda()
            for i, seq in enumerate(batch['seq']):
                cw_mask[i][1:len(seq)] = 1
            cw.wv = self.model.bert.embeddings.word_embeddings.weight
            cw.mask = cw_mask
            cw.seq = data
            cw.batch_info = batch

            # attack
            adv_data = cw.run(self.model, input_embedding, attack_targets, batch_size=batch_size)

            # retest
            adv_seq = torch.tensor(batch['seq']).cuda()
            # print("orig", input_ids)
            # print("orig:", self.dataset.transform_back_text(adv_seq))
            for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
                if bi in cw.o_best_sent:
                    adv_seq.data[bi, add_start:add_end] = torch.LongTensor(cw.o_best_sent[bi])
                    # print("adv:", self.dataset.transform_back_text(cw.o_best_sent[bi]))
            out = self.model(adv_seq, segment_ids, input_mask)
            prediction = torch.max(out, 1)[1]
            untargeted_success += torch.sum((prediction != attack_targets).float()).item()
            # print('adv seq', adv_seq, untargeted_success)
            print("before tokenization attack:", untargeted_success)



            adv_text = None
            if 0 in cw.o_best_sent:
                adv_text = torch.LongTensor(cw.o_best_sent[0])
            # print(out)
            # orig_correct += batch['orig_correct'].item()
            # adv_correct += torch.sum((prediction == label).float()).item()
            # targeted_success += torch.sum((prediction == attack_targets).float()).item()

            # print("untargetd successful rate:", untargeted_success)
            return adv_text, untargeted_success
            # for i in range(len(batch['class'])):
            #     diff = difference(batch['seq'][i].cpu().numpy().tolist(), adv_seq[i].cpu().numpy().tolist())
            #     tot_diff += diff
            #     tot_len += batch['seq_len'][i].item()
            #     adv_pickle.append({
            #         'text': transform(adv_seq[i]),
            #         'label': label[i].item(),
            #         'target': attack_targets[i].item(),
            #         'pred': prediction[i].item(),
            #         'diff': diff,
            #         'orig_text': orig_sent,
            #         'seq_len': batch['seq_len'][i].item()
            #     })
            #     try:
            #         logger.info(("label:", label[i].item()))
            #         logger.info(("pred:", prediction[i].item()))
            #         logger.info(("target:", attack_targets[i].item()))
            #         logger.info(("orig:", orig_sent))
            #         logger.info(("adv:", transform(adv_seq[i])))
            #         logger.info(("adv:", transform(cw.o_best_sent[i])))
            #         logger.info(("seq_len:", batch['seq_len'][i].item()))
            #     except:
            #         continue
            # logger.info(("tot_seq_len:", tot_len))
            # logger.info(("orig_correct:", orig_correct))
            # logger.info(("adv_correct:", adv_correct))
            # logger.info(("diff:", diff))
            # logger.info(("tot_diff:", tot_diff))
            # logger.info(("targeted successful rate:", targeted_success))
            # logger.info(("untargetd successful rate:", untargeted_success))
            # logger.info(("tot:", tot))
            # joblib.dump(adv_pickle, root_dir + '/adv_text.pkl')
            #
            # logger.info(("orig_correct:", orig_correct / tot))
            # logger.info(("adv_correct:", adv_correct / tot))
            # logger.info(("targeted successful rate:", targeted_success / tot))
            # logger.info(("untargetd successful rate:", untargeted_success / tot))







class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader

    def transform_back_text(self, seq):
        if not isinstance(seq, list):
            seq = seq.squeeze().cpu().numpy().tolist()
        # return self.tokenizer.convert_ids_to_tokens([self.tokenizer._convert_id_to_token(x) for x in seq])
        return self.tokenizer.convert_ids_to_tokens(seq)


def attack(text_ls, true_label, predictor, model, batch_size=1):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    # if true_label != orig_label:
    #     return '', 0, orig_label, orig_label, 0
    # else:
        # print(text_ls)
    cw = CarliniL2(debug=False, targeted=False)
    cw.num_classes = args.nclasses
    num_queries = 1
    orig_label = orig_label.cpu().item()
    adv_seq, success = model.attack([text_ls], orig_label, cw, batch_size)
    if adv_seq is not None:
        text_prime = model.dataset.transform_back_text(adv_seq)
        print("adv texts:", text_prime)
    else:
        print("optimize fail")
        text_prime = text_ls
    num_changed = 0
    return text_prime, num_changed, orig_label, \
           torch.argmax(predictor([text_prime])).cpu().item(), num_queries


def random_attack(text_ls, true_label, predictor, perturb_ratio, stop_words_set, word2idx, idx2word, cos_sim,
                  sim_predictor=None, import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15,
                  synonym_num=50, batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # randomly get perturbed words
        perturb_idxes = random.sample(range(len_text), int(len_text * perturb_ratio))
        words_perturb = [(idx, text_ls[idx]) for idx in perturb_idxes]

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor(new_texts, batch_size=batch_size)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        return ' '.join(text_prime), num_changed, orig_label.cpu().item(), torch.argmax(predictor([text_prime])).cpu().item(), num_queries


def main():
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # get data to attack
    texts, labels = dataloader.read_corpus(args.dataset_path)
    data = list(zip(texts, labels))
    data = data[:args.data_size] # choose how many samples for adversary
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=100, cnn=True).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
    predictor = model.text_pred
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}

    # print("Building vocab...")
    # with open(args.counter_fitting_embeddings_path, 'r') as ifile:
    #     for line in ifile:
    #         word = line.split()[0]
    #         if word not in idx2word:
    #             idx2word[len(idx2word)] = word
    #             word2idx[word] = len(idx2word) - 1

    # print("Building cos sim matrix...")
    # if args.counter_fitting_cos_sim_path:
    #     # load pre-computed cosine similarity matrix if provided
    #     print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
    #     cos_sim = np.load(args.counter_fitting_cos_sim_path)
    # else:
    #     # calculate the cosine similarity matrix
    #     print('Start computing the cosine similarity matrix!')
    #     embeddings = []
    #     with open(args.counter_fitting_embeddings_path, 'r') as ifile:
    #         for line in ifile:
    #             embedding = [float(num) for num in line.strip().split()[1:]]
    #             embeddings.append(embedding)
    #     embeddings = np.array(embeddings)
    #     product = np.dot(embeddings, embeddings.T)
    #     norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    #     cos_sim = product / np.dot(norm, norm.T)
    # print("Cos sim import finished!")

    # build the semantic similarity module
    # use = USE(args.USE_cache_path)
    use = None

    # start attacking
    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    log_file = open(os.path.join(args.output_dir, 'results_log'), 'a')

    stop_words_set = criteria.get_stopwords()
    print('Start attacking!')
    for idx, (text, true_label) in enumerate(data):
        if idx % 20 == 0:
            print('{} samples out of {} have been finished!'.format(idx, args.data_size))
        if args.perturb_ratio > 0.:
            new_text, num_changed, orig_label, \
            new_label, num_queries = random_attack(text, true_label, predictor, args.perturb_ratio, stop_words_set,
                                                    word2idx, idx2word, cos_sim, sim_predictor=use,
                                                    sim_score_threshold=args.sim_score_threshold,
                                                    import_score_threshold=args.import_score_threshold,
                                                    sim_score_window=args.sim_score_window,
                                                    synonym_num=args.synonym_num,
                                                    batch_size=args.batch_size)
        else:
            new_text, num_changed, orig_label, \
            new_label, num_queries = attack(text, true_label, predictor, model, batch_size=args.batch_size)

        print(true_label, orig_label, new_label)
        print("orig texts:", text)
        if true_label != orig_label:
            orig_failures += 1
            print("orig failure")
        else:
            nums_queries.append(num_queries)
        if orig_label != new_label:
            adv_failures += 1
            print(f"attack successful: {adv_failures}/{idx + 1}={adv_failures / (idx + 1)}")

        changed_rate = 1.0 * num_changed / len(text)

        if orig_label != new_label:
            changed_rates.append(changed_rate)
            orig_texts.append(text)
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)

    message = 'For target model {}: original accuracy: {:.3f}%, adv accuracy: {:.3f}%, ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}\n'.format(args.target_model,
                                                                     (1-orig_failures/args.data_size)*100,
                                                                     (1-adv_failures/args.data_size)*100,
                                                                     np.mean(changed_rates)*100,
                                                                     np.mean(nums_queries))
    print(message)
    log_file.write(message)

    # with open(os.path.join(args.output_dir, 'adversaries.txt'), 'w') as ofile:
    #     for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
    #         ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))
    adv_data = {
        'adv_text': adv_texts,
        'orig_text': orig_texts,
        'true_labels': true_labels,
        'new_labels': new_labels
    }
    import joblib
    joblib.dump(adv_data, os.path.join(args.output_dir, args.save))

if __name__ == "__main__":
    main()