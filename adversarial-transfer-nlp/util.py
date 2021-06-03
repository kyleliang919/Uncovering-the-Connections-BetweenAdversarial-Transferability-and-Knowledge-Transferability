import errno
import json
import argparse

import logging
import os
import random
import sys

import numpy as np
import torch


class Dictionary(object):
    def __init__(self, path=''):
        self.word2idx = dict()
        self.idx2word = list()
        if path != '':  # load an external dictionary
            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emsize', type=int, default=100,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers in BiLSTM')
    parser.add_argument('--attention-unit', type=int, default=350,
                        help='number of attention unit')
    parser.add_argument('--attention-hops', type=int, default=1,
                        help='number of attention hops, for multi-hop attention model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--nfc', type=int, default=3000,
                        help='hidden (fully connected) layer size for classifier MLP')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true', dest='cuda', help='use CUDA')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='do not use CUDA')
    parser.set_defaults(cuda=True)
    parser.add_argument("--device", default=0, type=int, help="cuda device")
    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle the dataset')
    parser.add_argument('--perturb', action='store_true',
                        help='perturb the original append')
    parser.add_argument('--perturb_ans', action='store_true',
                        help='perturb the original answer')
    parser.add_argument('--origin', action='store_true',
                        help='Use origin result')
    parser.add_argument('--seperate_test', action='store_true',
                        help='seperate test the acc on different stocks')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='saved_model.pth',
                        help='path to save the final model')
    parser.add_argument('--task', type=str, default='test',
                        help='task name to be logged')
    parser.add_argument('--dev_file', type=str, default='none_n1000_k1_s0.json',
                        help='dev file to be tested')
    parser.add_argument('--load', type=str, default='',
                        help='path to load the final model')
    parser.add_argument('--load-ae', type=str, default='',
                        help='path to load the ae model')
    parser.add_argument('--dictionary', type=str, default='full_vocab.txt',
                        help='path to save the dictionary, for faster corpus loading')
    parser.add_argument('--word-vector', type=str, default='full_emb.pth',
                        help='path for pre-trained word vectors (e.g. GloVe), should be a PyTorch model.')
    parser.add_argument('--train-data', type=str, default='trn.pkl',
                        help='location of the training data, should be a json file')
    parser.add_argument('--val-data', type=str, default='tst.pkl',
                        help='location of the development data, should be a json file')
    parser.add_argument('--test-data', type=str, default='sampled_1000.pkl',
                        help='location of the test data, should be a json file')
    parser.add_argument('--train-tree', type=str, default='full-trn-trees.pkl',
                        help='location of the training data, should be a json file')
    parser.add_argument('--val-tree', type=str, default='full-val-trees.pkl',
                        help='location of the development data, should be a json file')
    parser.add_argument('--test-tree', type=str, default='full-tst-trees.pkl',
                        help='location of the test data, should be a json file')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--class-number', type=int, default=5,
                        help='number of classes')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='type of optimizer')
    parser.add_argument('--penalization-coeff', type=float, default=1,
                        help='the penalization coefficient')
    parser.add_argument('--maj-days', type=int, default=1,
                        help='number of days of news to do majority vote')

    # for seq2seq model
    parser.add_argument('--model', type=str, default='test',
                        help='model name')
    parser.add_argument('--save_ans', type=str, default='',
                        help='dump ans')
    parser.add_argument('--logger', type=str, default='log',
                        help='log')
    parser.add_argument('--tr', type=float, default=1.0,
                        help='teacher ratio')
    parser.add_argument('--const', type=float, default=1e4,
                        help='initial const for cw attack')
    parser.add_argument('--confidence', type=float, default=0,
                        help='initial const for cw attack')
    parser.add_argument('--optim', type=str, default='sgd',
                        help='type of optimizer')
    parser.add_argument('--test_train', action='store_true',
                        help='whether test training set')
    parser.add_argument('--untargeted', action='store_true',
                        help='perform untargeted attack')

    # for tree model
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='cw max steps')
    parser.add_argument('--strategy', type=int, default=0,
                        help='attack strategy. '
                             'No.0: choose the attack target to class 13 \n'
                             'No.1: Choose the attack target to the next class \n')
    parser.add_argument('--encode_rel', default=True, type=bool,
                        help='TreeLSTM encoding has relation.')
    parser.add_argument('--decode_word', default=True, type=bool,
                        help='TreeLSTM decodes with word.')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--debugging', action='store_true',
                        help='output debug at each step')
    parser.add_argument('--debug_cw', action='store_true',
                        help='debug cw output')
    parser.add_argument('--decreasing_temp', action='store_true',
                        help='use decreasing temp technique')
    parser.add_argument('--baseline', action='store_true',
                        help='run baseline algo')
    parser.add_argument('--l1', action='store_true',
                        help='use l1 norm')
    parser.add_argument('--temp', default=1e-1, type=float,
                        help='softmax temparature')


    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['wordLSTM', 'bert', 'wordCNN'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        help="pre-trained target model path")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        help="Path to the USE encoder cache.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.7,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--data_size",
                        default=100,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--perturb_ratio",
                        default=0.,
                        type=float,
                        help="Whether use random perturbation for ablation study")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")


    return parser.parse_args()


args = get_args()
root_dir = "{}/{}".format(args.model, args.task)


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def init_logger():
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    log_formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler("{0}/info.log".format(root_dir), mode='a')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)


# logger = init_logger()

PAD = 0
UNK = 1
# not part of the qa vocab, assigned with minus index
EOS = -1
SOS = -2

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
EOS_WORD = '<eos>'
SOS_WORD = '<sos>'
