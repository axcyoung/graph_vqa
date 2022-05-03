import numpy as np
import torch
import torch.utils.data
import os.path
import sys
import random
import h5py
import itertools
import re
import tqdm
import time

import pandas as pd
import data_util
import hickle as hkl
import pickle as pkl
import cv2
import skimage.io

from transformer import Constants
from tools.Graph_helper import get_video_trajectory
from transformer.Constants_tgif import CLIP_SIZE_Q,CLIP_SIZE_R,CLIP_SIZE_O, TGIF_DATA_DIR, VIDEO_DATA_DIR, MAX_F_NUM, MAX_Q_LEN, MAX_O_F_NUM


_colors = [(244, 67, 54), (255, 245, 157), (29, 233, 182), (118, 255, 3),
        (33, 150, 243), (179, 157, 219), (233, 30, 99), (205, 220, 57),
        (27, 94, 32), (255, 111, 0), (187, 222, 251), (24, 255, 255),
        (63, 81, 181), (156, 39, 176), (183, 28, 28), (130, 119, 23),
        (139, 195, 74), (0, 188, 212), (224, 64, 251), (96, 125, 139),
        (0, 150, 136), (121, 85, 72), (26, 35, 126), (129, 212, 250),
        (158, 158, 158), (225, 190, 231), (183, 28, 28), (230, 81, 0),
        (245, 127, 23), (27, 94, 32), (0, 96, 100), (13, 71, 161),
        (74, 20, 140), (198, 40, 40), (239, 108, 0), (249, 168, 37),
        (46, 125, 50), (0, 131, 143), (21, 101, 192), (106, 27, 154),
        (211, 47, 47), (245, 124, 0), (251, 192, 45), (56, 142, 60),
        (0, 151, 167), (25, 118, 210), (123, 31, 162), (229, 57, 53),
        (251, 140, 0), (253, 216, 53), (67, 160, 71), (0, 172, 193),
        (30, 136, 229), (142, 36, 170), (244, 67, 54), (255, 152, 0),
        (255, 235, 59), (76, 175, 80), (0, 188, 212), (33, 150, 243)]


def paired_collate_fn(rets):
    #obj_feature, obj_st, obj_pos, trajectory
    tmp1 = [ret['video_features'][0] for ret in rets]
    obj_feature = [ret['video_features'][1] for ret in rets]
    obj_st = [ret['video_features'][2] for ret in rets]
    obj_num = [ret['video_features'][3] for ret in rets]
    trajectory = [ret['video_features'][4] for ret in rets]
    tmp4 = [ret['question_words'] for ret in rets]
    tmp5 = [ret['answer'] for ret in rets]

    #print('res_features1', len(tmp1), tmp1[0].shape)
    res_features = collate_fn_fea(tmp1)
    #print('res_features2', res_features[0].size())
    #print('res_features3', res_features[1].size(), res_features[1])
    questions = collate_fn(tmp4)
    answer = torch.LongTensor(tmp5)

    #convert to tensor
    obj_feature = torch.Tensor(obj_feature)
    obj_st = torch.LongTensor(obj_st)
    obj_num = torch.LongTensor(obj_num)
    trajectory = torch.Tensor(trajectory)

    '''print('res_features', res_features[0].size(), res_features[1].size())
    print('obj_feature', obj_feature.size())
    print('questions', questions[0].size(), questions[1].size())'''

    return (*res_features, obj_feature, obj_st, obj_num, trajectory, *questions, answer)


def collate_fn(insts, n=CLIP_SIZE_Q):
    ''' Pad the instance to the max seq length in batch '''
    max_len = ((MAX_Q_LEN-1) // n + 1) * n
    
    if type(insts[0][0]) != list:
        #max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([
            inst + [Constants.PAD] * (max_len - len(inst))
            for inst in insts])

        batch_pos = np.array([
            [pos_i+1 if w_i != Constants.PAD else 0
             for pos_i, w_i in enumerate(inst)] for inst in batch_seq])
    else:
        # max_len = 0
        # for inst in insts:
        #     for ins in inst:
        #         if max_len < len(ins):
        #             max_len = len(ins)
        batch_seq = np.array([
            [ins + [Constants.PAD] * (max_len - len(ins)) for ins in inst] for inst in insts])

        batch_pos = np.array([
            [[pos_i+1 if w_i != Constants.PAD else 0
             for pos_i, w_i in enumerate(ins)] for ins in inst] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos


def collate_fn_fea(insts, n=None):
    ''' Pad the instance to the max seq length in batch '''

    max_len = MAX_F_NUM
    batch_seq = np.array([
        inst.tolist() + [[Constants.PAD] * inst.shape[1]] * (max_len - len(inst))
        for inst in insts])

    batch_pos = np.array([
        [pos_i + 1 if pos_i<len(inst) else 0 for pos_i in range(max_len)] for inst in insts])

    batch_seq = torch.Tensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos


def collate_fn_obj(insts, n):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)
    max_len = (max_len // n + 1) * n
    batch_seq = np.array([
        inst.tolist() + [[Constants.PAD] * inst.shape[1]] * (max_len - len(inst))
        for inst in insts])

    batch_pos = np.array([
        [ins[4] for ins in inst] + [Constants.PAD]*(max_len - len(inst)) for inst in insts])

    batch_seq = torch.Tensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos


__PATH__ = os.path.abspath(os.path.dirname(__file__))


def assert_exists(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)


# PATHS
TYPE_TO_CSV = {'FrameQA': 'Train_frameqa_question.csv',
               'Count': 'Train_count_question.csv',
               'Trans': 'Train_transition_question.csv',
               'Action' : 'Train_action_question.csv'}
assert_exists(TGIF_DATA_DIR)
eos_word = '<EOS>'


class DatasetTGIF(torch.utils.data.Dataset):
    # def __init__(
    #     self, src_word2idx, tgt_word2idx,
    #     src_insts=None, tgt_insts=None):
    #
    #     assert src_insts
    #     assert not tgt_insts or (len(src_insts) == len(tgt_insts))
    #
    #     src_idx2word = {idx:word for word, idx in src_word2idx.items()}
    #     self._src_word2idx = src_word2idx
    #     self._src_idx2word = src_idx2word
    #     self._src_insts = src_insts
    #
    #     tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
    #     self._tgt_word2idx = tgt_word2idx
    #     self._tgt_idx2word = tgt_idx2word
    #     self._tgt_insts = tgt_insts
    def __init__(self,
                 dataset_name='train',
                 image_feature_net='resnet',
                 layer='pool5',
                 max_length=MAX_Q_LEN,
                 use_moredata=False,
                 max_n_videos=None,
                 data_type=None,
                 dataframe_dir=None,
                 vocab_dir=None,
                 mode=None,
                 video_name=None):
        self.dataframe_dir = dataframe_dir
        self.vocabulary_dir = vocab_dir
        # self.use_moredata = use_moredata
        self.dataset_name = dataset_name
        # self.image_feature_net = image_feature_net
        # self.layer = layer
        self.max_length = max_length
        self.max_n_videos = max_n_videos
        self.data_type = data_type
        self.data_df = self.read_df_from_csvfile()


        self.mode = mode
        self.video_name = video_name

        if max_n_videos is not None:
            self.data_df = self.data_df[:max_n_videos]
        self.ids = list(self.data_df.index)



    def __len__(self):
        if self.max_n_videos is not None:
            if self.max_n_videos <= len(self.data_df):
                return self.max_n_videos
        return len(self.data_df)


    def read_df_from_csvfile(self):
        assert self.data_type in ['FrameQA', 'Count', 'Trans', 'Action'], 'Should choose data type '

        if self.data_type == 'FrameQA':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_frameqa_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_frameqa_question.csv')
            self.total_q = pd.read_csv(os.path.join(self.dataframe_dir,'Total_frameqa_question.csv'), sep='\t')
        elif self.data_type == 'Count':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_count_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_count_question.csv')
            self.total_q = pd.read_csv(os.path.join(self.dataframe_dir,'Total_count_question.csv'), sep='\t')
        elif self.data_type == 'Trans':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_transition_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_transition_question.csv')
            self.total_q = pd.read_csv(os.path.join(self.dataframe_dir,'Total_transition_question.csv'), sep='\t')
        elif self.data_type == 'Action':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_action_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_action_question.csv')
            self.total_q = pd.read_csv(os.path.join(self.dataframe_dir,'Total_action_question.csv'), sep='\t')

        assert_exists(train_data_path)
        assert_exists(test_data_path)

        if self.dataset_name == 'train':
            data_df = pd.read_csv(train_data_path, sep='\t')
        elif self.dataset_name == 'test':
            data_df = pd.read_csv(test_data_path, sep='\t')

        data_df['row_index'] = range(1, len(data_df)+1) # assign csv row index
        return data_df


    def test_key_translate(self):
        candidates = []
        for i in range(len(self.data_df)):
            key_df = self.data_df.loc[i, 'gif_name']
            gif_name = str(key_df)
            if gif_name == self.video_name:
                candidates.append(i)
        print('candidates_keys:',candidates)

        choice_idx = random.randint(0,len(candidates)-1)
        choice_one_key = candidates[choice_idx]
        print('current key', choice_one_key)
        return  choice_one_key

    @property
    def n_words(self):
        ''' The dictionary size. '''
        if not hasattr(self, 'word2idx'):
            raise Exception('Dictionary not built yet!')
        return len(self.word2idx)

    def __repr__(self):
        if hasattr(self, 'word2idx'):
            return '<Dataset (%s) with %d videos and %d words>' % (
                self.dataset_name, len(self), len(self.word2idx))
        else:
            return '<Dataset (%s) with %d videos -- dictionary not built>' % (
                self.dataset_name, len(self))

    def split_sentence_into_words(self, sentence, eos=True):
        '''
        Split the given sentence (str) and enumerate the words as strs.
        Each word is normalized, i.e. lower-cased, non-alphabet characters
        like period (.) or comma (,) are stripped.
        When tokenizing, I use ``data_util.clean_str``
        '''
        try:
            words = data_util.clean_str(sentence).split()
        except:
            print(sentence)
            sys.exit()
        if eos:
            words = words + [eos_word]
        for w in words:
            if not w:
                continue
            yield w

    def build_word_vocabulary(self, all_captions_source=None,
                              word_count_threshold=0,):
        '''
        borrowed this implementation from @karpathy's neuraltalk.
        '''

        if all_captions_source is None:
            all_captions_source = self.get_all_captions()

        # enumerate all sentences to build frequency table
        word_counts = {}
        nsents = 0
        for sentence in all_captions_source:
            nsents += 1
            for w in self.split_sentence_into_words(sentence):
                word_counts[w] = word_counts.get(w, 0) + 1

        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

        # build index and vocabularies
        self.word2idx = {}
        self.idx2word = {}

        self.idx2word[0] = '.'
        self.idx2word[1] = 'UNK'
        self.word2idx['#START#'] = 0
        self.word2idx['UNK'] = 1
        for idx, w in enumerate(vocab, start=2):
            self.word2idx[w] = idx
            self.idx2word[idx] = w
        pkl.dump(self.word2idx, open(os.path.join(self.vocabulary_dir, 'word_to_index_%s.pkl'%self.data_type), 'wb'))
        pkl.dump(self.idx2word, open(os.path.join(self.vocabulary_dir, 'index_to_word_%s.pkl'%self.data_type), 'wb'))

        word_counts['.'] = nsents
        bias_init_vector = np.array([1.0*word_counts[self.idx2word[i]] if i>1 else 0 for i in self.idx2word])
        bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
        bias_init_vector = np.log(bias_init_vector)
        bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
        self.bias_init_vector = bias_init_vector

        # self.total_q = pd.DataFrame().from_csv(os.path.join(dataframe_dir,'Total_desc_question.csv'), sep='\t')
        answers = list(set(self.total_q['answer'].values))
        self.ans2idx = {}
        self.idx2ans = {}
        for idx, w in enumerate(answers):
            self.ans2idx[w] = idx
            self.idx2ans[idx] = w
        pkl.dump(self.ans2idx, open(os.path.join(self.vocabulary_dir, 'ans_to_index_%s.pkl'%self.data_type), 'wb'))
        pkl.dump(self.idx2ans, open(os.path.join(self.vocabulary_dir, 'index_to_ans_%s.pkl'%self.data_type), 'wb'))

        # Make glove embedding.
        import spacy
        nlp = spacy.load('en_vectors_web_lg')

        max_length = len(vocab)
        GLOVE_EMBEDDING_SIZE = 300

        glove_matrix = np.zeros([max_length+2, GLOVE_EMBEDDING_SIZE])
        for i in range(len(vocab)):
            w = vocab[i]
            w_embed = nlp(u'%s' % w).vector
            glove_matrix[i+2, :] = w_embed

        vocab = pkl.dump(glove_matrix, open(os.path.join(self.vocabulary_dir, 'vocab_embedding_%s.pkl'%self.data_type), 'wb'))
        self.word_matrix = glove_matrix

    def load_word_vocabulary(self):

        word_matrix_path = os.path.join(self.vocabulary_dir, 'vocab_embedding_%s.pkl' % self.data_type)

        word2idx_path = os.path.join(self.vocabulary_dir, 'word_to_index_%s.pkl' % self.data_type)
        idx2word_path = os.path.join(self.vocabulary_dir, 'index_to_word_%s.pkl' % self.data_type)
        ans2idx_path = os.path.join(self.vocabulary_dir, 'ans_to_index_%s.pkl' % self.data_type)
        idx2ans_path = os.path.join(self.vocabulary_dir, 'index_to_ans_%s.pkl' % self.data_type)

        if not (os.path.exists(word_matrix_path) and os.path.exists(word2idx_path) and \
                os.path.exists(idx2word_path) and os.path.exists(ans2idx_path) and \
                os.path.exists(idx2ans_path)):
            self.build_word_vocabulary()

        with open(word_matrix_path, 'rb') as f:
            self.word_matrix = pkl.load(f)

        with open(word2idx_path, 'rb') as f:
            self.word2idx = pkl.load(f)

        with open(idx2word_path, 'rb') as f:
            self.idx2word = pkl.load(f)

        with open(ans2idx_path, 'rb') as f:
            self.ans2idx = pkl.load(f)

        with open(idx2ans_path, 'rb') as f:
            self.idx2ans = pkl.load(f)

    def share_word_vocabulary_from(self, dataset):
        assert hasattr(dataset, 'idx2word') and hasattr(dataset, 'word2idx'), \
            'The dataset instance should have idx2word and word2idx'
        assert (isinstance(dataset.idx2word, dict) or isinstance(dataset.idx2word, list)) \
                and isinstance(dataset.word2idx, dict), \
            'The dataset instance should have idx2word and word2idx (as dict)'

        self.idx2word = dataset.idx2word
        self.word2idx = dataset.word2idx
        self.ans2idx = dataset.ans2idx
        self.idx2ans = dataset.idx2ans
        if hasattr(dataset, 'word_matrix'):
            self.word_matrix = dataset.word_matrix

    def iter_ids(self, shuffle=False):
        # if self.data_type == 'Trans':
        if shuffle:
            random.shuffle(self.ids)
        for key in self.ids:
            yield key

    def get_all_captions(self):
        '''
        Iterate caption strings associated in the vid/gifs.
        '''
        qa_data_df = pd.read_csv(os.path.join(self.dataframe_dir, TYPE_TO_CSV[self.data_type]), sep='\t')

        all_sents = []
        for row in qa_data_df.iterrows():
            all_sents.extend(self.get_captions(row))
        # self.data_type
        return all_sents

    def get_captions(self, row):
        if self.data_type == 'FrameQA':
            columns = ['description', 'question', 'answer']
        elif self.data_type == 'Count':
            columns = ['question']
        elif self.data_type == 'Trans':
            columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']
        elif self.data_type == 'Action':
            columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']

        sents = [row[1][col] for col in columns if not pd.isnull(row[1][col])]
        return sents

    def mode_print(self, contents):
        if self.mode == 'test' and self.dataset_name=='test':
            print(contents)

    def load_video_feature(self, key, object_num=5, filter_score=0.8):
        F = MAX_F_NUM
        F_obj = MAX_O_F_NUM

        key_df = self.data_df.loc[key, 'gif_name']
        gif_name = str(key_df)

        gif_imgs = os.listdir(os.path.join(VIDEO_DATA_DIR, gif_name))
        gif_imgs = sorted(gif_imgs, key=lambda x:int(x[:-4].split('-')[-1]))
        output_dir = './visualization/pic'

        res_feature_dir = os.path.join(TGIF_DATA_DIR, 'res_feature')
        obj_feature_dir = os.path.join(TGIF_DATA_DIR, 'obj_feature')
        file_name = gif_name + '.h5'
        res_file = os.path.join(res_feature_dir, file_name)
        obj_file = os.path.join(obj_feature_dir, file_name)
        res = h5py.File(res_file, 'r')
        obj = h5py.File(obj_file, 'r')
        res_feature = np.zeros(2048)
        obj_feature = []
        obj_st = []
        obj_pos = []
        num_frames = len(obj.keys())


        # First Get Obj Fea

        stride_ave_sample = (num_frames-1)//F_obj + 1
        #print('num_frames:{}, stride_ave_sample:{}'.format(num_frames, stride_ave_sample))
        sample_list = [i for i in range(0, num_frames, stride_ave_sample)]
        #print('[OBJ] len of sample_list{}\nsample_list display:{}'.format(len(sample_list), sample_list))

        for i in sample_list:
            tmp = np.zeros(5)
            i = i + 1
            frame = 'frame'+str(i)
            rois = np.array(obj[frame]['rois'])
            class_ids = np.array(obj[frame]['class_ids'])
            scores = np.array(obj[frame]['scores'])
            features = np.array(obj[frame]['features'])
            n = 0
            # select 5 object per frame
            tmp_obj_st, tmp_obj_feature = [], []#[dynamic_N, dim] and [dynamic_N, 5]
            for idx, score in enumerate(scores.tolist()):
                if score > filter_score:
                    n += 1
                    for _i in range(4):
                        tmp[_i] = rois[idx][_i]
                    tmp[4] = i
                    tmp_obj_st.append(tmp.tolist())
                    tmp_obj_feature.append(features[idx])
                if n > object_num-1:
                    break
            cur_obj_pos = min(object_num, n)
            obj_pos.append(cur_obj_pos)

            #visualization
            if self.mode=='test':
                buf = skimage.io.imread(os.path.join(VIDEO_DATA_DIR, gif_name, gif_imgs[i-1]))

                for idx, bbox in enumerate(tmp_obj_st):
                    ymin, xmin, ymax, xmax = [int(round(num)) for num in bbox[:4]]
                    bbox_thickness = 1

                    cv2.rectangle(buf, (xmin, ymin), (xmax, ymax), _colors[idx % len(_colors)], bbox_thickness)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(buf, text=str(idx), org=(xmin, ymin),fontFace=font, fontScale=0.5, color=_colors[idx % len(_colors)], thickness=1)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                cv2.imwrite(os.path.join(output_dir, str(i-1)+'.jpg'), buf)



            
            tmp_obj_st_pad = np.pad(tmp_obj_st, ((0,object_num - cur_obj_pos),(0,0)),'constant') if cur_obj_pos!=0 else \
                    np.zeros((object_num,5),dtype=int)
            tmp_obj_feature_pad = np.pad(tmp_obj_feature, ((0,object_num - cur_obj_pos),(0,0)),'constant') if cur_obj_pos!=0 \
                    else np.zeros((object_num,1024))
            obj_st.append(tmp_obj_st_pad)
            obj_feature.append(tmp_obj_feature_pad)
        #pad to [F, N, D]
        #print('[obj_feature]',np.array(obj_feature).shape)
        obj_feature_pad = np.pad(obj_feature,((0,F_obj-len(sample_list)),(0,0),(0,0)),'constant')
        obj_st_pad = np.pad(obj_st,((0,F_obj-len(sample_list)),(0,0),(0,0)),'constant')
        obj_pos_pad = np.pad(obj_pos,(0,F_obj-len(sample_list)),'constant')#[F]

        if np.sum(obj_pos_pad)==0:
            obj_pos_pad[0]=1
            obj_st_pad[0] = np.ones((object_num, 5))*1e-7
            obj_feature_pad[0] = np.ones((object_num, 1024))*1e-7

        # Then Res Fea
        stride_ave_sample = (num_frames-1)//F + 1
        #print('num_frames:{}, stride_ave_sample:{}'.format(num_frames, stride_ave_sample))
        sample_list = [i for i in range(0, num_frames, stride_ave_sample)]
        #print('[Frame] len of sample_list{}\nsample_list display:{}'.format(len(sample_list), sample_list))

        for i in sample_list:
            res_feature = np.vstack((res_feature, res['feature'][i-1]))


        if len(res_feature.shape) > 1 and res_feature.shape[0] > 2:
            res_feature = res_feature[1:]
        else:
            res_feature = np.ones([2, 2048])*1e-7

        trajectory = get_video_trajectory(obj_feature_pad, obj_st_pad, obj_pos_pad)#[F,N,D]

        return [res_feature, obj_feature_pad, obj_st_pad, obj_pos_pad, trajectory]


    def get_video_feature(self, key):
        video_feature = self.load_video_feature(key)
        return video_feature

    def convert_sentence_to_matrix(self, sentence, eos=True):
        '''
        Convert the given sentence into word indices and masks.
        WARNING: Unknown words (not in vocabulary) are revmoed.
        Args:
            sentence: A str for unnormalized sentence, containing T words
        Returns:
            sentence_word_indices : list of (at most) length T,
                each being a word index
        '''
        self.clean_sentence = [w for w in self.split_sentence_into_words(sentence,eos)]

        self.mode_print(self.clean_sentence)

        sent2indices = [self.word2idx[w] if w in self.word2idx else 1 for w in
                        self.split_sentence_into_words(sentence,eos)] # 1 is UNK, unknown
        T = len(sent2indices)
        length = min(T, self.max_length)
        return sent2indices[:length]

    # def get_video_mask(self, video_feature):
    #     video_length = video_feature.shape[0]
    #     return data_util.fill_mask(self.max_length, video_length, zero_location='LEFT')

    def get_sentence_mask(self, sentence):
        sent_length = len(sentence)
        return data_util.fill_mask(self.max_length, sent_length, zero_location='RIGHT')

    def get_question(self, key):
        '''
        Return question string for given key.
        '''
        question = self.data_df.loc[key, ['question', 'description']].values
        if len(list(question.shape)) > 1:
            question = question[0]
        question = question[0]
        return self.convert_sentence_to_matrix(question, eos=False)

    def get_question_mask(self, sentence):
        sent_length = len(sentence)
        return data_util.fill_mask(self.max_length, sent_length, zero_location='LEFT')

    def get_answer(self, key):
        answer = self.data_df.loc[key, ['answer', 'type']].values

        if len(list(answer.shape)) > 1:
            answer = answer[0]

        anstype = answer[1]
        answer = answer[0]

        return answer, anstype

    def get_FrameQA_result(self, key):
        video_feature = self.get_video_feature(key)
        # video_mask = self.get_video_mask(video_feature)
        answer, answer_type = self.get_answer(key)
        if str(answer) in self.ans2idx:
            answer = self.ans2idx[answer]
        else:
            # unknown token, check later
            answer = 1
        question = self.get_question(key)
        # question_mask = self.get_question_mask(question)
        answer_type = float(int(answer_type))
        debug_sent = self.data_df.loc[key, 'question']

        ret = {
            'ids': key,
            'video_features': video_feature,
            'question_words': question,
            # 'question_words_right': batch_question_right,
            # 'video_mask': batch_video_mask,
            # 'question_mask': batch_question_mask,
            'answer': answer,
            'answer_type': answer_type,
            'debug_sent': debug_sent
        }
        return ret

    def get_Count_question(self, key):
        '''
        Return question string for given key.
        '''
        question = self.data_df.loc[key, 'question']
        return self.convert_sentence_to_matrix(question, eos=False)

    def get_Count_question_mask(self, sentence):
        sent_length = len(sentence)
        return data_util.fill_mask(self.max_length, sent_length, zero_location='RIGHT')

    def get_Count_answer(self, key):
        return self.data_df.loc[key, 'answer']

    def get_Count_result(self, key):
        video_feature = self.get_video_feature(key)
        # video_mask = self.get_video_mask(video_feature)
        answer = max(self.get_Count_answer(key), 1)
        question = self.get_Count_question(key)
        # question_mask = self.get_Count_question_mask(question)
        # # Left align
        # batch_question[k, :len(question)] = question
        # # Right align
        # batch_question_right[k, -len(question):] = question
        # batch_question_mask[k] = question_mask
        debug_sent = self.data_df.loc[key, 'question']

        ret = {
            'ids': key,
            'video_features': video_feature,
            'question_words': question,
            # 'question_words_right': batch_question_right,
            # 'video_mask': batch_video_mask,
            # 'question_mask': batch_question_mask,
            'answer': answer,
            'debug_sent': debug_sent
        }
        return ret

    def get_Trans_dict(self, key):
        a1 = self.data_df.loc[key, 'a1'].strip()
        a2 = self.data_df.loc[key, 'a2'].strip()
        a3 = self.data_df.loc[key, 'a3'].strip()
        a4 = self.data_df.loc[key, 'a4'].strip()
        a5 = self.data_df.loc[key, 'a5'].strip()
        question = self.data_df.loc[key, 'question'].strip()
        row_index = self.data_df.loc[key, 'row_index']
        # as list of sentence strings
        candidates = [a1, a2, a3, a4, a5]
        answer = self.data_df.loc[key, 'answer']

        candidates_to_indices = [self.convert_sentence_to_matrix(question + ' ' + x)
                                 for x in candidates]
        return {
            'answer': answer,
            'candidates': candidates_to_indices,
            'raw_sentences': candidates,
            'row_indices': row_index,
            'question': question
        }

    def get_Trans_matrix(self, candidates, is_left=True):
        candidates_matrix = np.zeros([5, self.max_length], dtype=np.uint32)
        for k in range(5):
            sentence = candidates[k]
            if is_left:
                candidates_matrix[k, :len(sentence)] = sentence
            else:
                candidates_matrix[k, -len(sentence):] = sentence
        return candidates_matrix

    def get_Trans_mask(self, candidates):
        mask_matrix = np.zeros([5, self.max_length], dtype=np.uint32)
        for k in range(5):
            mask_matrix[k] = data_util.fill_mask(self.max_length,
                                                 len(candidates[k]),
                                                 zero_location='RIGHT')
        return mask_matrix

    def get_Trans_result(self, key):
        MC_dict = self.get_Trans_dict(key)
        candidates = MC_dict['candidates']
        raw_sentences = MC_dict['raw_sentences']
        answer = int(MC_dict['answer'])
        question = MC_dict['question']
        video_feature = self.get_video_feature(key)
        candidates_matrix = self.get_Trans_matrix(candidates)
        candidates_matrix_right = self.get_Trans_matrix(candidates, is_left=False)
        # video_mask = self.get_video_mask(video_feature)
        candidates_mask = self.get_Trans_mask(candidates)
        row_indices = MC_dict['row_indices']
        debug_sent = self.data_df.loc[key, 'a' + str(int(answer + 1))]

        ret = {
            'ids': key,
            'video_features': video_feature,
            'question_words': candidates,
            'candidates_right': candidates_matrix_right,
            'answer': answer,
            'raw_sentences': raw_sentences,
            # 'video_mask': batch_video_mask,
            'candidates_mask': candidates_mask,
            'debug_sent': debug_sent,
            'row_indices': row_indices,
            'question': question,
        }


        return ret

    def get_Action_result(self, key):
        return self.get_Trans_result(key)

    def split_dataset(self, ratio=0.1):
        data_split = DatasetTGIF(dataset_name=self.dataset_name,
                                 max_length=self.max_length,
                                 max_n_videos=self.max_n_videos,
                                 data_type=self.data_type,
                                 dataframe_dir=self.dataframe_dir,
                                 vocab_dir=self.vocabulary_dir)

        data_split.ids = self.ids[-int(ratio*len(self.ids)):]
        self.ids = self.ids[:-int(ratio*len(self.ids))]
        return data_split

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self.ids)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self.word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self.ans2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self.word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self.ans2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self.idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self.idx2ans

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self.mode == 'test':
            idx = self.test_key_translate()
            time.sleep(5)
        if self.data_type == 'FrameQA':
            return self.get_FrameQA_result(idx)
        elif self.data_type == 'Count':
            return self.get_Count_result(idx)
        elif self.data_type == 'Trans':
            return self.get_Trans_result(idx)
        elif self.data_type == 'Action':
            return self.get_Action_result(idx)
        else:
            raise Exception('data_type error in next_batch')

