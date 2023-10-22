# -*- coding: utf-8 -*-
"""
processes the CUHK_PEDES/reid_raw.json, output the train_data, val_data, test_data,
all data including[image_path, caption_id(be coded), label]

Created on Thurs., Aug. 1(st), 2019 at 20:10

@author: zifyloo
"""

from utils.read_write_data import read_json, makedir, save_dict, write_txt
import argparse
from collections import namedtuple
import os
from random import shuffle
import numpy as np
import transformers as ppb


ImageDecodeData = namedtuple('ImageDecodeData', ['id', 'image_path', 'captions_id', 'split'])


class Word2Index(object):

    def __init__(self, vocab):
        self._vocab = {w: index + 1 for index, w in enumerate(vocab)}
        self.unk_id = len(vocab) + 1
        # print(self._vocab)

    def __call__(self, word):
        if word not in self._vocab:
            return self.unk_id
        return self._vocab[word]


def parse_args():
    parser = argparse.ArgumentParser(description='Command for data pre_processing')
    parser.add_argument('--img_root', default='/data1/zhiying/text-image/data/ICFG_PEDES', type=str)
    parser.add_argument('--json_root', default='/data1/zhiying/text-image/data/ICFG-PEDES.json', type=str)
    parser.add_argument('--out_root', default='./processed_data_singledata_ICFG', type=str) # processed_data_spa_img
    parser.add_argument('--min_word_count', default='2', type=int)
    parser.add_argument('--shuffle', default=False, type=bool)
    args = parser.parse_args()
    return args


def split_json(args):
    """
    has 40206 image in reid_raw_data
    has 13003 id
    every id has several images and every image has several caption
    data's structure in reid_raw_data is dict ['split', 'captions', 'file_path', 'processed_tokens', 'id']
    """
    reid_raw_data = read_json(args.json_root)

    train_json = []
    test_json = []

    for data in reid_raw_data:
        data_save = {
            'img_path': data['file_path'],
            'id': data['id'],
            'tokens': data['processed_tokens'],
            'captions': data['captions']
        }
        split = data['split'].lower()
        if split == 'train':
            train_json.append(data_save)

        if split == 'test':
            data_save['tokens'] = data_save['tokens'][0]
            test_json.append(data_save)
    return train_json, test_json


def build_vocabulary(train_json, args):

    word_count = {}
    for data in train_json:
        for caption in data['tokens']:
            for word in caption:
                word_count[word.lower()] = word_count.get(word.lower(), 0) + 1

    word_count_list = [[v, k] for v, k in word_count.items()]
    word_count_list.sort(key=lambda x: x[1], reverse=True)  # from high to low

    good_vocab = [v for v, k in word_count.items() if k >= args.min_word_count]

    print('top-10 highest frequency words:')
    for w, n in word_count_list[0:10]:
        print(w, n)

    good_count = len(good_vocab)
    all_count = len(word_count_list)
    good_word_rate = good_count * 100.0 / all_count
    st = 'good words: %d, total_words: %d, good_word_rate: %f%%' % (good_count, all_count, good_word_rate)
    write_txt(st, os.path.join(args.out_root, 'data_message'))
    print(st)
    word2Ind = Word2Index(good_vocab)

    save_dict(good_vocab, os.path.join(args.out_root, 'ind2word'))
    return word2Ind


def generate_captionid(data_json, word2Ind, data_name, args, tokenizer):

    id_save = []
    lstm_caption_id_save = []
    bert_caption_id_save = []
    img_path_save = []
    caption_save = []
    same_id_index_save = []
    un_idx = word2Ind.unk_id
    # train_id_count = {}
    data_save_by_id = {}

    count_id = []
    for data in data_json:

        if data['id'] in [1369, 4116, 6116]:  # only one image
            print(111)
            continue
        if data['id'] not in count_id:
            count_id.append(data['id'])

        id_new = len(count_id) - 1

        data_save_i = {
            'img_path': data['img_path'],
            'id': id_new,
            'tokens': data['tokens'],
            'captions': data['captions']
        }
        if id_new not in data_save_by_id.keys():
            data_save_by_id[id_new] = []

        data_save_by_id[id_new].append(data_save_i)

    data_order = 0
    for id_new, data_save_by_id_i in data_save_by_id.items():

        caption_length = 0
        for data_save_by_id_i_i in data_save_by_id_i:
            caption_length += len(data_save_by_id_i_i['captions'])

        data_order_i = data_order + np.arange(caption_length)
        data_order_i_begin = 0

        for data_save_by_id_i_i in data_save_by_id_i:
            caption_length_i = len(data_save_by_id_i_i['captions'])
            data_order_i_end = data_order_i_begin + caption_length_i
            data_order_i_select = np.delete(data_order_i, np.arange(data_order_i_begin, data_order_i_end))
            data_order_i_begin = data_order_i_end

            for j in range(len(data_save_by_id_i_i['tokens'])):
                tokens_j = data_save_by_id_i_i['tokens'][j]
                lstm_caption_id = []
                for word in tokens_j:
                    lstm_caption_id.append(word2Ind(word))
                if un_idx in lstm_caption_id:
                    lstm_caption_id = list(filter(lambda x: x != un_idx, lstm_caption_id))

                caption_j = data_save_by_id_i_i['captions'][j]
                bert_caption_id = tokenizer.encode(caption_j, add_special_tokens=True)

                id_save.append(data_save_by_id_i_i['id'])
                img_path_save.append(data_save_by_id_i_i['img_path'])
                same_id_index_save.append(data_order_i_select)

                lstm_caption_id_save.append(lstm_caption_id)

                bert_caption_id_save.append(bert_caption_id)
                caption_save.append(caption_j)

        data_order = data_order + caption_length
    print(sorted(count_id))
    data_save = {
        'id': id_save,
        'img_path': img_path_save,
        'same_id_index': same_id_index_save,

        'lstm_caption_id': lstm_caption_id_save,
        'bert_caption_id': bert_caption_id_save,
        'captions': caption_save,
    }

    img_num = len(set(img_path_save))
    id_num = len(set(id_save))
    # print(sorted(set(id_save)))
    caption_num = len(lstm_caption_id_save)
    """
    for i in range(len(same_id_index_save)):
        for j in same_id_index_save[i]:
            if id_save[i] != id_save[j] or i in same_id_index_save[i]:
                print(111)
    """
    st = '%s_img_num: %d, %s_id_num: %d, %s_caption_num: %d, ' %(data_name, img_num, data_name, id_num, data_name, caption_num)
    write_txt(st, os.path.join(args.out_root, 'data_message'))

    return data_save


def generate_test_val_caption_id(data_json, word2Ind, data_name, args, tokenizer):
    id_save = []
    lstm_caption_id_save = []
    bert_caption_id_save = []
    caption_save = []
    img_path_save = []
    img_caption_index_save = []
    caption_matching_img_index_save = []
    caption_label_save = []

    un_idx = word2Ind.unk_id

    img_caption_index_i = 0
    caption_matching_img_index_i = 0
    for data in data_json:
        id_save.append(data['id'])
        img_path_save.append(data['img_path'])

        for j in range(len(data['tokens'])):

            tokens_j = data['tokens'][j]
            lstm_caption_id = []
            for word in tokens_j:
                lstm_caption_id.append(word2Ind(word))
            if un_idx in lstm_caption_id:
                lstm_caption_id = list(filter(lambda x: x != un_idx, lstm_caption_id))

            caption_j = data['captions'][j]
            bert_caption_id = tokenizer.encode(caption_j, add_special_tokens=True)

            caption_matching_img_index_save.append(caption_matching_img_index_i)
            lstm_caption_id_save.append(lstm_caption_id)
            bert_caption_id_save.append(bert_caption_id)
            caption_save.append(caption_j)

            caption_label_save.append(data['id'])
            img_caption_index_save.append([img_caption_index_i, img_caption_index_i+len(data['captions'])-1])
        img_caption_index_i += len(data['captions'])
        caption_matching_img_index_i += 1

    data_save = {
        'id': id_save,
        'img_path': img_path_save,
        'img_caption_index': img_caption_index_save,

        'caption_matching_img_index': caption_matching_img_index_save,
        'caption_label': caption_label_save,
        'lstm_caption_id': lstm_caption_id_save,
        'bert_caption_id': bert_caption_id_save,
        'captions': caption_save,
    }

    img_num = len(set(img_path_save))
    id_num = len(set(id_save))
    caption_num = len(lstm_caption_id_save)
    st = '%s_img_num: %d, %s_id_num: %d, %s_caption_num: %d, ' % (
    data_name, img_num, data_name, id_num, data_name, caption_num)
    write_txt(st, os.path.join(args.out_root, 'data_message'))

    # print(sorted(set(id_save)))

    return data_save


def main(args):
    train_json, test_json = split_json(args)

    word2Ind = build_vocabulary(train_json, args)
    
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    train_save = generate_captionid(train_json, word2Ind, 'train', args, tokenizer)
    test_save = generate_test_val_caption_id(test_json, word2Ind, 'test', args, tokenizer)


    save_dict(train_save, os.path.join(args.out_root, 'train_save'))
    save_dict(test_save, os.path.join(args.out_root, 'test_save'))



if __name__ == '__main__':

    args = parse_args()
    if args.shuffle:
        args.out_root = args.out_root + '_shuffle'

    makedir(args.out_root)
    main(args)
    """
    from utils.read_write_data import read_dict
    train_save_dic = read_dict(args.out_root + '/train_save.pkl')

    id_save = train_save_dic['id']
    img_path_save = train_save_dic['img_path']
    caption_id_save = train_save_dic['caption_id']
    same_id_index_save = train_save_dic['same_id_index']

    num = 1600
    print(id_save[num])
    print(img_path_save[num])
    print(caption_id_save[num])
     
    print(same_id_index_save[num])
    for x in same_id_index_save[num]:
        print(id_save[x])
        print(img_path_save[x])
     

    train_save_dic = read_dict('./processed_data/train_save.pkl')
    id_save = train_save_dic['id']
    img_path_save = train_save_dic['img_path']
    caption_id_save = train_save_dic['caption_id']
    print(id_save[num])
    print(img_path_save[num])
    print(caption_id_save[num])
    """
