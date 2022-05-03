'''
This script handling the training process.
'''

import argparse
import math
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import DatasetTGIF, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from tensorboardX import SummaryWriter
import pickle
from transformer.Constants import CLIP_SIZE_Q,CLIP_SIZE_R,CLIP_SIZE_O,Q_TYPE,Q_IDS,IDS2TYPE

writer = SummaryWriter()

def cal_mc_performance(pred, gold):

    tmp = gold.unsqueeze(1).expand(-1, 5)
    pred = pred.squeeze_().view(-1,5)
    s_correct = pred.gather(dim=1, index=tmp)
    loss = torch.max(torch.zeros_like(pred), 1+pred-s_correct)
    loss = loss.sum(dim=1) - 1
    loss = loss.mean()

    pred = pred.max(1)[1]
    n_correct = pred.eq(gold)
    n_correct = n_correct.sum()

    return loss, n_correct

def cal_ct_performance(pred, gold):

    pred.squeeze_()
    #print(pred)
    #print(gold)
    loss = pred - gold.type_as(pred)
    loss = loss * loss
    loss = loss.mean()

    pred = pred.round().clamp(1, 10)
    out_loss = pred - gold.type_as(pred)
    out_loss = out_loss * out_loss
    out_loss = out_loss.mean()

    pred = pred.type_as(gold)
    n_correct = pred.eq(gold)
    n_correct = n_correct.sum()

    return loss, n_correct, out_loss

def cal_ct_performance_fm(pred, gold):

    loss_fun = torch.nn.CrossEntropyLoss()
    loss = loss_fun(pred, gold)

    softmax = torch.nn.Softmax(dim=1)
    pred = softmax(pred)
    pred = torch.argmax(pred, dim=1)

    out_loss = pred - gold
    out_loss = out_loss * out_loss
    out_loss = out_loss.type_as(loss).mean()

    n_correct = pred.eq(gold)
    n_correct = n_correct.sum()

    return loss, n_correct, out_loss

def cal_fm_performance(pred, gold):
    '''if len(gold.size())==1:
        pred = pred.view(1,-1)'''

    loss_fun = torch.nn.CrossEntropyLoss()
    loss = loss_fun(pred, gold)

    softmax = torch.nn.Softmax(dim=1)
    pred = softmax(pred)
    pred = torch.argmax(pred, dim=1)
    n_correct = pred.eq(gold)
    n_correct_for_type = n_correct
    n_correct = n_correct.sum()

    return loss, n_correct, n_correct_for_type


def cal_correct_for_type(q_type, n_correct_for_type):
    acc_for_type = [0.0 for i in range(5)]
    total_for_type = [0 for i in range(5)]
    for i in range(len(q_type)):
        cur_q_type = q_type[i]
        acc_for_type[cur_q_type] += float(n_correct_for_type[i])
        total_for_type[cur_q_type] += 1
    return acc_for_type, total_for_type

# def cal_loss(pred, gold, smoothing):
#     ''' Calculate cross entropy loss, apply label smoothing if needed. '''
#
#     gold = gold.contiguous().view(-1)
#
#     if smoothing:
#         eps = 0.1
#         n_class = pred.size(1)
#
#         one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
#         one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
#         log_prb = F.log_softmax(pred, dim=1)
#
#         non_pad_mask = gold.ne(Constants.PAD)
#         loss = -(one_hot * log_prb).sum(dim=1)
#         loss = loss.masked_select(non_pad_mask).sum()  # average later
#     else:
#         loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')
#
#     return loss

def mc_data_expand(data):
    s = data.size()
    if len(s) == 3:
        data = data.unsqueeze_(1).expand(-1, 5, -1, -1).contiguous().view(-1, s[1], s[2])
    elif len(s) == 2:
        data = data.unsqueeze_(1).expand(-1, 5, -1).contiguous().view(-1, s[1])
    else:
        data = data.unsqueeze_(1).expand(-1, 5)
        #data = data.view(-1)
    return data


def train_epoch(model, training_data, optimizer, device, epoch_i, d_type):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_sample_total = 0
    n_sample_correct = 0
    n_sample_total_q_type = [0 for i in range(5)]
    n_sample_correct_q_type = [0.0 for i in range(5)]
    ct = 0


    for batch in tqdm(
            training_data, desc='  - (Training)   '):
        # prepare data
        res_feature, res_feature_pos, obj_feature, obj_feature_pos, \
        obj_st, obj_st_pos, questions, questions_pos, answer, q_type = batch
        if d_type == "Trans" or d_type == "Action":
            s = questions.size()
            questions = questions.view(-1, s[2])
            questions_pos = questions_pos.view(-1, s[2])
            res_feature = mc_data_expand(res_feature)
            res_feature_pos = mc_data_expand(res_feature_pos)
            obj_feature = mc_data_expand(obj_feature)
            obj_st = mc_data_expand(obj_st)
            obj_st_pos = mc_data_expand(obj_st_pos)
            #answer = mc_data_expand(answer)

        res_feature = res_feature.to(device)
        res_feature_pos = res_feature_pos.to(device)
        obj_feature = obj_feature.to(device)
        obj_feature_pos = obj_feature_pos.to(device)
        obj_st = obj_st.to(device)
        obj_st_pos = obj_st_pos.to(device)
        questions = questions.to(device)
        questions_pos = questions_pos.to(device)
        answer = answer.to(device)
        gold = answer
        # print(res_feature.size())
        # print(res_feature_pos.size())
        # print(obj_feature.size())
        # print(obj_st.size())
        # print(obj_st_pos.size())

        # forward
        optimizer.zero_grad()
        pred = model(res_feature, res_feature_pos, obj_feature, obj_feature_pos,
                     obj_st, obj_st_pos, questions, questions_pos, answer)

        # backward
        if d_type == "Trans" or d_type == "Action":
            loss, n_correct = cal_mc_performance(pred, gold)
        elif d_type == "Count":
            loss, n_correct, out_loss = cal_ct_performance_fm(pred, gold)
        else:
            loss, n_correct, n_correct_for_type = cal_fm_performance(pred, gold)

        n_correct_for_type = n_correct_for_type.cpu().numpy()
        cur_acc_for_type, cur_total_for_type = cal_correct_for_type(q_type, n_correct_for_type)
        n_sample_correct_q_type = [n_sample_correct_q_type[i]+cur_acc_for_type[i] for i in range(5)]
        n_sample_total_q_type = [n_sample_total_q_type[i]+cur_total_for_type[i] for i in range(5)]
        '''for i in range(gold.size()[0]):
            n_sample_total_q_type[q_type[i]] += 1'''

        loss.backward()

        # update parameters
        optimizer.step_and_update_lr(epoch_i)

        # note keeping
        if d_type == "Count":
            loss_i = out_loss.item()
        else:
            loss_i = loss.item()
        n_correct_i = n_correct.item()
        #print(loss_i)
        #print(n_correct_i)
        total_loss += loss_i

        n_sample_total += gold.size()[0]
        n_sample_correct += n_correct_i
        acc_per_batch = n_correct_i / gold.size()[0]

        #print("batch loss : %f , acc : %f" % (loss.item(), acc_per_batch))
        if ct % 100 == 0:
            writer.add_scalar('train/batch_loss', loss_i, epoch_i*len(training_data)+ct)
            writer.add_scalar('train/batch_acc', acc_per_batch, epoch_i*len(training_data)+ct)
        ct = ct + 1

    loss_per_epoch = total_loss / ct
    accuracy = n_sample_correct / n_sample_total
    return loss_per_epoch, accuracy, n_sample_correct_q_type,n_sample_total_q_type

def eval_epoch(model, validation_data, device, d_type):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_sample_total = 0
    n_sample_correct = 0
    n_sample_total_q_type = [0 for i in range(5)]
    n_sample_correct_q_type = [0.0 for i in range(5)]
    ct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, desc='  - (Validation) '):
            # prepare data
            #src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            res_feature, res_feature_pos, obj_feature, obj_feature_pos, \
            obj_st, obj_st_pos, questions, questions_pos, answer, q_type = batch

            if d_type == "Trans" or d_type == "Action":
                s = questions.size()
                questions = questions.view(-1, s[2])
                questions_pos = questions_pos.view(-1, s[2])
                res_feature = mc_data_expand(res_feature)
                res_feature_pos = mc_data_expand(res_feature_pos)
                obj_feature = mc_data_expand(obj_feature)
                obj_st = mc_data_expand(obj_st)
                obj_st_pos = mc_data_expand(obj_st_pos)
                # answer = mc_data_expand(answer)

            res_feature = res_feature.to(device)
            res_feature_pos = res_feature_pos.to(device)
            obj_feature = obj_feature.to(device)
            obj_feature_pos = obj_feature_pos.to(device)
            obj_st = obj_st.to(device)
            obj_st_pos = obj_st_pos.to(device)
            questions = questions.to(device)
            questions_pos = questions_pos.to(device)
            answer = answer.to(device)
            gold = answer

            # forward
            #pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            pred = model(res_feature, res_feature_pos, obj_feature, obj_feature_pos,
                         obj_st, obj_st_pos, questions, questions_pos, answer)

            if d_type == "Trans" or d_type == "Action":
                loss, n_correct = cal_mc_performance(pred, gold)
            elif d_type == "Count":
                loss, n_correct, out_loss = cal_ct_performance_fm(pred, gold)
            else:
                loss, n_correct, n_correct_for_type = cal_fm_performance(pred, gold)
            n_correct_for_type = n_correct_for_type.cpu().numpy()
            cur_acc_for_type, cur_total_for_type = cal_correct_for_type(q_type, n_correct_for_type)
            n_sample_correct_q_type = [n_sample_correct_q_type[i]+cur_acc_for_type[i] for i in range(5)]
            n_sample_total_q_type = [n_sample_total_q_type[i]+cur_total_for_type[i] for i in range(5)]
            '''for i in range(gold.size()[0]):
                n_sample_total_q_type[q_type[i]] += 1'''

            # note keeping
            if d_type == "Count":
                loss_i = out_loss.item()
            else:
                loss_i = loss.item()
            n_correct_i = n_correct.item()
            total_loss += loss_i

            n_sample_total += gold.size()[0]
            n_sample_correct += n_correct_i




            #acc_per_batch = n_correct / gold.size()[0]
            #print("batch loss : %f , acc : %f" % (loss.item(), acc_per_batch))
            ct = ct + 1

    loss_per_epoch = total_loss / ct
    accuracy = n_sample_correct / n_sample_total
    return loss_per_epoch, accuracy, n_sample_correct_q_type,n_sample_total_q_type

def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    #log_train_file = None
    #log_valid_file = None

    # if opt.log:
    #     cur_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    #     log_train_file = opt.log + opt.data_type + '_' + cur_time + '_train.log'
    #     log_valid_file = opt.log + opt.data_type + '_' + cur_time + '_valid.log'
    #
    #     print('[Info] Training performance will be written to file: {} and {}'.format(
    #         log_train_file, log_valid_file))
    #
    #     with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
    #         log_tf.write('epoch,loss,ppl,accuracy\n')
    #         log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    train_time = []

    model.to(device)

    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_accu, n_sample_correct_q_type,n_sample_total_q_type = train_epoch(
            model, training_data, optimizer, device, epoch_i, d_type=opt.data_type)
        elapse = (time.time() - start) / 60
        train_time.append(elapse)
        print('  - (Training)   loss: {loss: 8.4f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(loss=train_loss, accu=100*train_accu, elapse=elapse))
        acc_q_type = [n_sample_correct_q_type[i]/n_sample_total_q_type[i] if n_sample_total_q_type[i]!=0 else 0.0 for i in range(5)]
        print('acc_q_type',acc_q_type)
        print('n_sample_correct_q_type',n_sample_correct_q_type)
        print('n_sample_total_q_type',n_sample_total_q_type)
        writer.add_scalar('train/epoch_loss', train_loss, epoch_i)
        writer.add_scalar('train/epoch_acc', train_accu, epoch_i)

        start = time.time()
        valid_loss, valid_accu, n_sample_correct_q_type,n_sample_total_q_type = eval_epoch(
            model, validation_data, device, d_type=opt.data_type)
        print('  - (Validation) loss: {loss: 8.4f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(loss=valid_loss, accu=100*valid_accu, elapse=(time.time()-start)/60))
        acc_q_type = [n_sample_correct_q_type[i]/n_sample_total_q_type[i] if n_sample_total_q_type[i]!=0 else 0.0  for i in range(5)]
        print('acc_q_type',acc_q_type)
        print('n_sample_correct_q_type',n_sample_correct_q_type)
        print('n_sample_total_q_type',n_sample_total_q_type)
        
        writer.add_scalar('valid/epoch_loss', valid_loss, epoch_i)
        writer.add_scalar('valid/epoch_acc', valid_accu, epoch_i)

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + opt.data_type + '/' + opt.data_type + '_epoch_{epoch:d}_accu_{accu:3.3f}.chkpt'\
                    .format(epoch=epoch_i, accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                if valid_accu >= max(valid_accus):
                    model_name = opt.save_model + opt.data_type + '/' + opt.data_type + '_best_2.chkpt'
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The best checkpoint has been updated.')
                # if epoch_i >= opt.epoch-5:
                #     model_name = opt.save_model + opt.data_type + '_epoch_{epoch:d}_accu_{accu:3.3f}.chkpt'\
                #         .format(epoch=epoch_i, accu=100 * valid_accu)
                #     torch.save(checkpoint, model_name)

    cur_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    with open('./log2/train_time_'+opt.data_type+'_'+cur_time+'.pkl', 'wb') as f:
        pickle.dump(train_time, f)

        # if log_train_file and log_valid_file:
        #     with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
        #         log_tf.write('{epoch},{loss: 8.4f},{accu:3.3f}\n'.format(
        #             epoch=epoch_i, loss=train_loss, accu=100*train_accu))
        #         log_vf.write('{epoch},{loss: 8.4f},{accu:3.3f}\n'.format(
        #             epoch=epoch_i, loss=valid_loss, accu=100*valid_accu))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    # parser.add_argument('-data', required=True)
    parser.add_argument('-data_type', type=str, choices=['FrameQA', 'Count', 'Trans', 'Action'], default='FrameQA')
    parser.add_argument('-epoch', type=int, default=40)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-clip_size_q', type=int, default=3)
    parser.add_argument('-clip_size_r', type=int, default=3)
    parser.add_argument('-clip_size_o', type=int, default=3)

    # parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-learning_rate', type=float, default=0.002)
    parser.add_argument('-d_model', type=int, default=1024)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    # parser.add_argument('-d_k', type=int, default=64)
    # parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head_q', type=int, default=6)
    parser.add_argument('-n_head_v', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-n_warmup_steps', type=int, default=12000)

    parser.add_argument('-dropout', type=float, default=0.1)
    # parser.add_argument('-embs_share_weight', action='store_true')
    # parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default="./log2/")
    parser.add_argument('-save_model', default="./checkpoint/")
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    dataset_is_msvd = True
    if dataset_is_msvd:
        parser.add_argument('-vgg_dir', type=str, default='/home1/gumao/MM19/VideoQA-master/data/msvd_qa')
        parser.add_argument('-dataframe_dir', type=str, default='/home1/gumao/MM19/MSVD-QA')
        parser.add_argument('-vocab_dir', type=str, default='/home1/gumao/MM19/MSVD-QA/vocabulary')
    else:
        parser.add_argument('-vgg_dir', type=str, default='/home1/gumao/MM19/VideoQA-master/data/msrvtt_qa')
        parser.add_argument('-dataframe_dir', type=str, default='/home1/gumao/MM19/MSRVTT-QA')
        parser.add_argument('-vocab_dir', type=str, default='/home1/gumao/MM19/MSRVTT-QA/vocabulary')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = 300
    #opt.d_res_vec = 2048
    opt.d_res_vec = 4096
    opt.d_obj_vec = 1024
    opt.max_video_seq = 600
    opt.d_hid = 200
    opt.df_dir = opt.dataframe_dir
    opt.vc_dir = opt.vocab_dir

    opt.clip_size_q = CLIP_SIZE_Q
    opt.clip_size_r = CLIP_SIZE_R
    opt.clip_size_o = CLIP_SIZE_O

    #========= Loading Dataset =========#
    # data = torch.load(opt.data)
    print('start prepare_dataloaders')
    training_data, validation_data = prepare_dataloaders(opt)
    print('prepare over')

    opt.max_token_seq_len = (training_data.dataset.max_length // opt.clip_size_q + 1) * opt.clip_size_q
    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size
    opt.word_matrix = torch.FloatTensor(training_data.dataset.word_matrix)

    #========= Preparing Model =========#
    # if opt.embs_share_weight:
    #     assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
    #         'The src/tgt word2idx table are different but asked to share word embedding.'

    #print(opt)
    print('building model')
    device = torch.device('cuda:0' if opt.cuda else 'cpu')
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.word_matrix,
        opt.max_token_seq_len,
        opt.max_video_seq,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_res_vec=opt.d_res_vec,
        d_obj_vec=opt.d_obj_vec,
        d_inner=opt.d_inner_hid,
        d_hid=opt.d_hid,
        n_layers=opt.n_layers,
        n_head_q=opt.n_head_q,
        n_head_v=opt.n_head_v,
        data_type=opt.data_type,
        dropout=opt.dropout).to(device)
    print('build over')

    # optimizer = ScheduledOptim(
    #     optim.Adam(
    #         filter(lambda x: x.requires_grad, transformer.parameters()),
    #         betas=(0.9, 0.98), eps=1e-09),
    #     opt.d_model, opt.n_warmup_steps)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-08),
        opt.learning_rate, opt.n_warmup_steps)

    question_type_ids, question_types = training_data.dataset.get_question_type_index()

    train(transformer, training_data, validation_data, optimizer, device, opt)
    writer.close()


def prepare_dataloaders(opt):
    # ========= Preparing DataLoader =========#
    train_dataset = DatasetTGIF(
            dataset_name='train',
            data_type=opt.data_type,
            dataframe_dir=opt.df_dir,
            vocab_dir=opt.vc_dir,
            vgg_dir=opt.vgg_dir)

    train_dataset.load_word_vocabulary()

    #print('get_question_type_index\n',train_dataset.get_question_type_index())

    test_dataset = DatasetTGIF(
            dataset_name='test',
            data_type=opt.data_type,
            dataframe_dir=opt.df_dir,
            vocab_dir=opt.vc_dir,
            vgg_dir=opt.vgg_dir)

    test_dataset.share_word_vocabulary_from(train_dataset)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=4,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn,
            shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
            test_dataset,
            num_workers=4,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn)

    return train_loader, valid_loader


if __name__ == '__main__':
    main()
