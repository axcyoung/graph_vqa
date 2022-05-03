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

import pandas as pd
import data_util
import hickle as hkl
import pickle as pkl

# from IPython import embed
from transformer import Constants
from transformer.Constants import CLIP_SIZE_Q,CLIP_SIZE_R,CLIP_SIZE_O,Q_TYPE,Q_IDS,IDS2TYPE

def paired_collate_fn(rets):
    tmp1 = [ret['video_features'][0] for ret in rets]
    tmp2 = [ret['video_features'][1] for ret in rets]
    tmp3 = [ret['video_features'][2] for ret in rets]
    tmp4 = [ret['question_words'] for ret in rets]
    tmp5 = [ret['answer'] for ret in rets]

    q_type = [IDS2TYPE[ret['question_words'][0]] for ret in rets]

    #n = 3  # clip_size
    max_length = 20
    res_features = collate_fn_fea(tmp1, CLIP_SIZE_R)
    obj_features = collate_fn_fea(tmp2, CLIP_SIZE_O)
    obj_st = collate_fn_obj(tmp3, CLIP_SIZE_O)
    questions = collate_fn(tmp4, CLIP_SIZE_Q, max_length)
    answer = torch.LongTensor(tmp5)

    return (*res_features, *obj_features, *obj_st, *questions, answer,q_type)


def collate_fn(insts, n, _max_length):
    ''' Pad the instance to the max seq length in batch '''
    max_len = (_max_length // n + 1) * n
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


def collate_fn_fea(insts, n):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)
    max_len = (max_len // n + 1) * n
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
TGIF_DATA_DIR = os.path.normpath(os.path.join(__PATH__, '../dataset'))
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
                 max_length=20,
                 use_moredata=False,
                 max_n_videos=None,
                 data_type=None,
                 dataframe_dir=None,
                 vocab_dir=None,
                 vgg_dir=None
                 ):
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
        self.vgg_dir=vgg_dir

        if max_n_videos is not None:
            self.data_df = self.data_df[:max_n_videos]
        self.ids = list(self.data_df.index)
        print('ids len', len(self.ids))
        print('dataset_name',dataset_name)
        # if dataset_name == 'train':
            # random.shuffle(self.ids)

        # self.feat_h5 = self.read_tgif_from_hdf5()
    # def __del__(self):
        # self.feat_h5.close()

    def __len__(self):
        if self.max_n_videos is not None:
            if self.max_n_videos <= len(self.data_df):
                return self.max_n_videos
        return len(self.data_df)

    # def read_tgif_from_hdf5(self, gif_name):
    #     res_feature_dir = os.path.join(TGIF_DATA_DIR, 'res_feature')
    #     obj_feature_dir = os.path.join(TGIF_DATA_DIR, 'obj_feature')
    #     file_name = gif_name + '.h5'
    #     res_file = os.path.join(res_feature_dir, file_name)
    #     obj_file = os.path.join(obj_feature_dir, file_name)
    #     return {"res_feature": h5py.File(res_file, 'r'), "obj_feature": h5py.File(obj_file, 'r')}

    def read_df_from_csvfile(self):
        assert self.data_type in ['FrameQA', 'Count', 'Trans', 'Action'], 'Should choose data type '

        if self.data_type == 'FrameQA':
            self.train_data_path = os.path.join(self.dataframe_dir, 'train_qa.json')
            self.test_data_path = os.path.join(self.dataframe_dir, 'test_qa.json')
            self.answerset_path = os.path.join(self.dataframe_dir, 'answer_set.txt')
            #self.total_q = pd.DataFrame().from_csv(os.path.join(self.dataframe_dir,'Total_frameqa_question.csv'), sep='\t')
        elif self.data_type == 'Count':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_count_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_count_question.csv')
            self.total_q = pd.DataFrame().from_csv(os.path.join(self.dataframe_dir,'Total_count_question.csv'), sep='\t')
        elif self.data_type == 'Trans':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_transition_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_transition_question.csv')
            self.total_q = pd.DataFrame().from_csv(os.path.join(self.dataframe_dir,'Total_transition_question.csv'), sep='\t')
        elif self.data_type == 'Action':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_action_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_action_question.csv')
            self.total_q = pd.DataFrame().from_csv(os.path.join(self.dataframe_dir,'Total_action_question.csv'), sep='\t')

        assert_exists(self.train_data_path)
        assert_exists(self.test_data_path)

        if self.dataset_name == 'train':
            data_df = pd.read_json(self.train_data_path)
        elif self.dataset_name == 'test':
            data_df = pd.read_json(self.test_data_path)
        return data_df

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
        #get answers
        if self.dataset_name == 'train':
            answer_freq = self.data_df['answer'].value_counts()
            answer_freq = pd.DataFrame(answer_freq.iloc[0:1000])
            answer_freq.to_csv(self.answerset_path, columns=[], header=False)

        #get ans_to_index and index_to_ans
        self.answerset = pd.read_csv(self.answerset_path, header=None)[0]
        self.ans2idx = {}
        self.idx2ans = {}
        for idx, w in enumerate(self.answerset):
            self.ans2idx[w] = idx
            self.idx2ans[idx] = w
        pkl.dump(self.ans2idx, open(os.path.join(self.vocabulary_dir, 'ans_to_index_%s.pkl'%self.data_type), 'wb'))
        pkl.dump(self.idx2ans, open(os.path.join(self.vocabulary_dir, 'index_to_ans_%s.pkl'%self.data_type), 'wb'))

        #
        word_counts = dict()
        train_qa = self.data_df
        train_qa = train_qa[train_qa['answer'].isin(self.answerset)]

        questions = train_qa['question'].values
        for q in questions:
            words = q.rstrip('?').split()
            for word in words:
                if len(word) >= 2:
                    word_counts[word] = word_counts.get(word, 0) + 1
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
        qa_data_df = pd.DataFrame().from_csv(os.path.join(self.dataframe_dir, TYPE_TO_CSV[self.data_type]), sep='\t')

        all_sents = []
        for row in qa_data_df.iterrows():
            all_sents.extend(self.get_captions(row))
        # self.data_type
        return all_sents

    def get_captions(self, row):
        if self.data_type == 'FrameQA':
            columns = ['question', 'answer']
        elif self.data_type == 'Count':
            columns = ['question']
        elif self.data_type == 'Trans':
            columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']
        elif self.data_type == 'Action':
            columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']

        sents = [row[1][col] for col in columns if not pd.isnull(row[1][col])]
        return sents

    def load_video_feature(self, vid):
        obj = h5py.File(os.path.join(self.dataframe_dir, 'obj_feature', 'vid'+str(vid)+'.h5'), 'r')
        if not hasattr(self, 'vgg'):
            self.vgg = h5py.File(os.path.join(self.vgg_dir, 'video_feature_20.h5'), 'r')['vgg']
        #vgg starts from 0
        vgg_feature = self.vgg[vid-1]

        obj_feature = np.zeros(1024)
        obj_st = np.zeros(5)
        tmp = np.zeros(5)
        num_frames = len(obj.keys())
        if num_frames <= 30:
            sample_list = range(num_frames)
        elif num_frames <= 60:
            sample_list = range(0, num_frames, 2)
        elif num_frames <= 90:
            sample_list = range(0, num_frames, 3)
        elif num_frames <= 150:
            sample_list = range(0, num_frames, 5)
        elif num_frames <= 240:
            sample_list = range(0, num_frames, 8)
        else:
            sample_list = range(0, num_frames, 20)

        for i in sample_list:
            i = i + 1
            frame = 'frame'+str(i)
            rois = np.array(obj[frame]['rois'])
            class_ids = np.array(obj[frame]['class_ids'])
            scores = np.array(obj[frame]['scores'])
            features = np.array(obj[frame]['features'])
            n = 0
            # select 3 object per frame
            for idx, score in enumerate(scores.tolist()):
                if score > 0.9:
                    n += 1
                    tmp[0] = (rois[idx][1] + rois[idx][3]) / 2
                    tmp[1] = (rois[idx][0] + rois[idx][2]) / 2
                    tmp[2] = abs(rois[idx][1] - rois[idx][3])
                    tmp[3] = abs(rois[idx][0] - rois[idx][2])
                    tmp[4] = i
                    obj_st = np.vstack((obj_st, tmp))
                    obj_feature = np.vstack((obj_feature, features[idx]))
                if n > 2:
                    break

        if len(obj_st.shape) > 1 and obj_st.shape[0] > 2:
            obj_st = obj_st[1:]
            obj_feature = obj_feature[1:]
        else:
            obj_feature = np.ones([2, 1024])*0.01
            obj_st = np.ones([2, 5])
        return [vgg_feature, obj_feature, obj_st]

    # def get_video_feature_dimension(self):
    #     if self.image_feature_net == 'resnet':
    #         assert self.layer.lower() in ['fc1000', 'pool5', 'res5c']
    #         if self.layer.lower() == 'res5c':
    #             return (self.max_length, 7, 7, 2048)
    #         elif self.layer.lower() == 'pool5':
    #             return (self.max_length, 1, 1, 2048)
    #     elif self.image_feature_net.lower() == 'c3d':
    #         if self.layer.lower() == 'fc6':
    #             return (self.max_length, 1, 1, 4096)
    #         elif self.layer.lower() == 'conv5b':
    #             return (self.max_length, 7, 7, 1024)
    #     elif self.image_feature_net.lower() == 'concat':
    #         assert self.layer.lower() in ['fc', 'conv']
    #         if self.layer.lower() == 'fc':
    #             return (self.max_length, 1, 1, 4096+2048)
    #         elif self.layer.lower() == 'conv':
    #             return (self.max_length, 7, 7, 1024+2048)

    def get_video_feature(self, vid):
        video_feature = self.load_video_feature(vid)
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
        '''question = self.data_df.loc[key, ['question', 'description']].values
        if len(list(question.shape)) > 1:
            question = question[0]
        question = question[0]'''
        question = self.data_df['question'][key]
        return self.convert_sentence_to_matrix(question, eos=False)

    def get_question_mask(self, sentence):
        sent_length = len(sentence)
        return data_util.fill_mask(self.max_length, sent_length, zero_location='LEFT')

    def get_answer(self, key):
        '''answer = [self.data_df['answer'][key],self.data_df['type'][key]]

        anstype = answer[1]
        answer = answer[0]

        return answer, anstype'''
        return self.data_df['answer'][key]

    def get_FrameQA_result(self, key):
        vid = self.data_df['video_id'][key]
        video_feature = self.get_video_feature(vid)
        # video_mask = self.get_video_mask(video_feature)
        answer= self.get_answer(key)
        if str(answer) in self.ans2idx:
            answer = self.ans2idx[answer]
        else:
            # unknown token, check later
            answer = 1
        question = self.get_question(key)
        # question_mask = self.get_question_mask(question)
        #answer_type = float(int(answer_type))
        debug_sent = self.data_df['question'][key]

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
            'max_length': self.max_length
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

    def get_question_type_index(self):
        question_types = ['what', 'who', 'how', 'when', 'where']
        q_ids = [self.word2idx[question_type]  for question_type in question_types ]
        id2q = {}
        _idx = 0
        for q_id in q_ids:
            id2q[q_id] = _idx
            _idx +=1
        return id2q, question_types

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
