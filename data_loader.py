# ---encoding:utf-8---
# @Time : 2020.08.15
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : data_loader_kmer.py

'''
The source author of the code is as above, and this study is modified based on it.
'''


import pickle
import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np
import config
from util import util_file


def split_kmer(sequences, k_mer):
    print('=' * 50, '1 to {}-mer Split'.format(k_mer), '=' * 50)
    '''
    sequences=['MKTLLLTL', 'VVVTIVC']
    1-mer:[
            [
             ['M', 'K', 'T', 'L', 'L', 'L', 'T', 'L']
            ], 
            [
             ['V', 'V', 'V', 'T', 'I', 'V', 'C']
            ]
          ]
    2-mer:[
            [
             ['-M', 'KT', 'LL', 'LT', 'L-'], 
             ['MK', 'TL', 'LL', 'TL']
            ], 
            [
             ['-V', 'VV', 'TI', 'VC'], 
             ['VV', 'VT', 'IV', 'C-']
            ]
          ]
    3-mer:[
            [
             ['--M', 'KTL', 'LLT', 'L--'], 
             ['-MK', 'TLL', 'LTL'],
             ['MKT', 'LLL', 'TL-']
            ], 
            [
             ['--V', 'VVT', 'IVC'], 
             ['-VV', 'VTI', 'VC-'],
             ['VVV', 'TIV', 'V--']
            ]
          ]
    '''
    sequences_kmer = []

    for seq in sequences:
        kmer_list = [[] for i in range(k_mer)]
        seq_kmer = [[] for i in range(len(seq))]

        # Traverse every token of every sequence
        for i in range(len(seq)):
            # Each token is divided into k-mer from 1 to k
            for k in range(1, k_mer + 1):
                # Each token position is represented by k kinds of kmer
                for j in range(k):
                    # There is no need to add '-' before the beginning of the sequence
                    if i - j >= 0:
                        kmer = seq[i - j:i - j + k]

                        # Need to add '-' after the end of the sequence
                        if i - j + k > len(seq):
                            num_pad = i - j + k - len(seq)
                            kmer += '-' * num_pad

                    # Need to add '-' before the beginning of the sequence
                    else:
                        num_pad = j - i
                        kmer = ('-' * num_pad) + seq[0:  k - num_pad]

                        # Need to add '-' after the end of the sequence
                        if k - num_pad > len(seq):
                            num_pad = k - num_pad - len(seq)
                            kmer += '-' * num_pad

                    kmer_list[k - 1].append(kmer)
                    seq_kmer[i].append(kmer)
        sequences_kmer.append(seq_kmer)

        if len(sequences_kmer) % 1000 == 0:
            print('Processing: {}/{}'.format(len(sequences_kmer), len(sequences)))

    print('=' * 50, '{}-mer Split Over'.format(k_mer), '=' * 50)
    return sequences_kmer


def transform_token2index(sequences, config):
    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)
    print('sequences_base_sample', sequences[0:5])

    for i, seq in enumerate(sequences):
        sequences[i] = ''.join(seq)

    k_mer = config.k_mer
    token2index = config.token2index
    sequences_kmer = split_kmer(sequences, k_mer)

    new_token_list = []
    num_token2index = len(token2index)

    token_list = list()
    max_len = 0
    for seq_kmer in sequences_kmer:
        seq_kmer_id_list = []
        for kmer_list in seq_kmer:
            # Handle keys not in token2index
            for kmer in kmer_list:
                if kmer not in token2index:
                    new_token_list.append(kmer)
            for i, token in enumerate(new_token_list):
                token2index[token] = i + num_token2index

            kmer_id_list = [token2index[kmer] for kmer in kmer_list]
            seq_kmer_id_list.append(kmer_id_list)
        token_list.append(seq_kmer_id_list)
        if len(seq_kmer) > max_len:
            max_len = len(seq_kmer)

    origin_token_list = list()
    for seq in sequences:
        seq_id = [token2index[residue] for residue in seq]
        origin_token_list.append(seq_id)

    print('-' * 20, '[transform_token2index]: check sequences_base and token_list head', '-' * 20)
    print('sequences_base', sequences[0:5])  # sequences_residue
    print('token_list', token_list[0:5])
    print('len(token_list)', len(token_list))
    print('len(origin_token_list)', len(origin_token_list))
    print('new_token_list', new_token_list)
    print('num_token2index', num_token2index)
    print('len(token2index)', len(token2index))

    # update token2index
    with open('data/kmer_residue2idx.pkl', 'wb') as file:
        pickle.dump(token2index, file)

    return token_list, origin_token_list, max_len


def make_data_with_unified_length(token_list, origin_token_list, labels, config):
    max_len = config.max_len + 2  # add [CLS] and [SEP]
    token2idx = config.token2index
    k_mer = config.k_mer
    kmer_num = (k_mer + 1) * k_mer // 2

    data = []
    for i in range(len(labels)):
        token_list[i] = [[token2idx['[CLS]']] * kmer_num] + token_list[i] + [[token2idx['[SEP]']] * kmer_num]
        n_pad = max_len - len(token_list[i])
        token_list[i].extend([[0] * kmer_num] * n_pad)

        origin_token_list[i] = [token2idx['[CLS]']] + origin_token_list[i] + [token2idx['[SEP]']]
        n_pad = max_len - len(origin_token_list[i])
        origin_token_list[i].extend([token2idx['[PAD]']] * n_pad)

        data.append([token_list[i], origin_token_list[i], labels[i]])
        # print('token_list[i]', len(token_list[i]), token_list[i])
        # print('origin_token_list[i]', len(origin_token_list[i]), origin_token_list[i])

    return data


def construct_dataset(data, config):
    cuda = config.cuda
    batch_size = config.batch_size

    input_ids, origin_input_ids,pos_embed, labels = zip(*data)

    if cuda:
        input_ids, origin_input_ids, pos_embed, labels = torch.cuda.LongTensor(input_ids), torch.cuda.LongTensor(origin_input_ids),torch.cuda.FloatTensor(pos_embed), torch.cuda.LongTensor(labels)
    else:
        input_ids, origin_input_ids, pos_embed, labels = torch.LongTensor(input_ids), torch.LongTensor(origin_input_ids),torch.FloatTensor(pos_embed), torch.LongTensor(labels)

    print('-' * 20, '[construct_dataset]: check GPU data', '-' * 20)
    print('input_ids.device:', input_ids.device)
    print('origin_input_ids.device:', origin_input_ids.device)
    print('pos_embed.device:',pos_embed.device)
    print('labels.device:', labels.device)

    print('-' * 20, '[construct_dataset]: check data shape', '-' * 20)
    print('input_ids:', input_ids.shape)  # [num_train_sequences, seq_len, kmer_num]
    print('origin_input_ids:', origin_input_ids.shape)  # [num_train_sequences, seq_len]
    print('pos_embde:',pos_embed.shape)
    print('labels:', labels.shape)  # [num_train_sequences, seq_len]

    data_loader = Data.DataLoader(MyDataSet(input_ids, origin_input_ids,pos_embed, labels),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False)

    print('len(data_loader)', len(data_loader))
    return data_loader


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, origin_input_ids,pos_embed, labels):
        self.input_ids = input_ids
        self.origin_input_ids = origin_input_ids
        self.pos_embed = pos_embed
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.origin_input_ids[idx], self.pos_embed[idx],self.labels[idx]


def position_encoding(seqs,config):
    """
    Position encoding features introduced in "Attention is all your need",
    the b is changed to 1000 for the short length of sequence.
    """
    d = config.dim_embedding
    b = 1000
    res = []
    for seq in seqs:
        N = len(seq)
        value = []
        for pos in range(N):
            tmp = []
            for i in range(d // 2):
                tmp.append(pos / (b ** (2 * i / d)))
            value.append(tmp)
        value = np.array(value)
        pos_encoding = np.zeros((N + 2, d))
        pos_encoding[1:-1, 0::2] = np.sin(value[:, :])
        pos_encoding[1:-1, 1::2] = np.cos(value[:, :])
        res.append(pos_encoding)
    return res

def load_data(path,config):
    ########
    tmp = pd.read_csv(path, header=None)
    sequences, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    ########
    # pos_embedding
    pos_embed = position_encoding(sequences,config)
    #####
    token_list, origin_token_list,max_len = transform_token2index(sequences, config)
    # token_list_train: [[1, 5, 8], [2, 7, 9], ...]

    config.max_len = max_len

    data = make_data_with_unified_length(token_list,origin_token_list, labels, config)
    ipt,ipt_or, lab = zip(*data)
    data = zip(ipt, ipt_or, pos_embed, lab)
    # data_train: [[[1, 5, 8], 0], [[2, 7, 9], 1], ...]

    data_loader = construct_dataset(data, config)
    return data_loader


if __name__ == '__main__':
    '''
    check loading tsv data
    '''
    config = config.get_train_config()

    token2index = pickle.load(open('../data/kmer_residue2idx.pkl', 'rb'))
    config.token2index = token2index

    config.path_train_data = '../data/m7G_dataset/train_m7G.csv'

    tmp = pd.read_csv(config.path_train_data, header=None)
    sequences = tmp[0].values.tolist()
    labels = tmp[1].values.tolist()

    data_loader = load_data(config.path_bench_data,config)
    print('-' * 20, '[data_loader]: check data batch', '-' * 20)
    for i, batch in enumerate(data_loader):
        input, origin_input,pos_embed, label = batch
        print('batch[{}], input:{}, origin_input:{},pos_embed:{}, label:{}'.format(i, input.shape, origin_input.shape,pos_embed.shape, label.shape))