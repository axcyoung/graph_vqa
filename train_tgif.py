import argparse
import math
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants_tgif as Constants
from dataset_tgif import DatasetTGIF, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from tensorboardX import SummaryWriter
import pickle
from transformer.Constants_tgif import CLIP_SIZE_Q,CLIP_SIZE_R,CLIP_SIZE_O, DATAFRAME_DIR, MAX_F_NUM, MAX_Q_LEN, MAX_O_F_NUM
from setproctitle import setproctitle
import os
import matplotlib.pyplot as plt
import seaborn as sns


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

    loss_fun = torch.nn.CrossEntropyLoss()
    loss = loss_fun(pred, gold)

    softmax = torch.nn.Softmax(dim=1)
    pred = softmax(pred)
    pred = torch.argmax(pred, dim=1)
    n_correct = pred.eq(gold)
    n_correct = n_correct.sum()

    return loss, n_correct

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
    if len(s) ==4:
        data = data.unsqueeze_(1).expand(-1, 5, -1, -1,-1).contiguous().view(-1, s[1], s[2], s[3])
    elif len(s) == 3:
        data = data.unsqueeze_(1).expand(-1, 5, -1, -1).contiguous().view(-1, s[1], s[2])
    elif len(s) == 2:
        data = data.unsqueeze_(1).expand(-1, 5, -1).contiguous().view(-1, s[1])
    else:
        data = data.unsqueeze_(1).expand(-1, 5)
        #data = data.view(-1)
    return data


def train_epoch(model, training_data, optimizer, device, epoch_i, d_type, writer, check=0, validation_data=None, opt=None, max_valid_acc=None):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_sample_total = 0
    n_sample_correct = 0
    ct = 0

    for batch in tqdm(
            training_data, desc='  - (Training)   '):

        # prepare data
        res_feature, res_feature_pos, obj_feature, obj_st, obj_num, trajectory, \
                 questions, questions_pos, answer = batch

        if ct>check and check>0:
            break
        if d_type == "Trans" or d_type == "Action":
            s = questions.size()
            questions = questions.view(-1, s[2])
            questions_pos = questions_pos.view(-1, s[2])
            res_feature = mc_data_expand(res_feature)
            res_feature_pos = mc_data_expand(res_feature_pos)
            obj_feature = mc_data_expand(obj_feature)
            obj_st = mc_data_expand(obj_st)
            obj_num = mc_data_expand(obj_num)
            trajectory = mc_data_expand(trajectory)
            #answer = mc_data_expand(answer)

        res_feature = res_feature.to(device)
        res_feature_pos = res_feature_pos.to(device)
        obj_feature = obj_feature.to(device)
        obj_st = obj_st.to(device)
        obj_num = obj_num.to(device)
        trajectory = trajectory.to(device)
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
        pred = model(res_feature, res_feature_pos, obj_feature, obj_st, obj_num, trajectory,  questions, questions_pos, answer)

        #print(pred)
        #print(gold)
        # backward
        if d_type == "Trans" or d_type == "Action":
            loss, n_correct = cal_mc_performance(pred, gold)
        elif d_type == "Count":
            loss, n_correct, out_loss = cal_ct_performance(pred, gold)
        else:
            loss, n_correct = cal_fm_performance(pred, gold)

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
        '''if ct!=0 and ct%8==0:
            tqdm.write('current loss:{}, current_acc:{}'.format(loss_i, n_correct_i))'''

        n_sample_total += gold.size()[0]
        n_sample_correct += n_correct_i
        acc_per_batch = n_correct_i / gold.size()[0]
        #print("batch loss : %f , acc : %f" % (loss.item(), acc_per_batch))
        if ct % 100 == 0:
            writer.add_scalar('train/batch_loss', loss_i, epoch_i*len(training_data)+ct)
            writer.add_scalar('train/batch_acc', acc_per_batch, epoch_i*len(training_data)+ct)


        if opt.valid_steps!=-1 and ct%opt.valid_steps==0 and validation_data is not None:
            
            start = time.time()
            valid_loss, valid_accu = eval_epoch(model, validation_data, device, d_type, writer)
            print('  - (Validation) loss: {loss: 8.4f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(loss=valid_loss, accu=100*valid_accu, elapse=(time.time()-start)/60))
        
            model_state_dict = model.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'settings': opt,
                'epoch': epoch_i}

            if opt.save_model:
                if opt.save_mode == 'all':
                    model_name = opt.save_model + opt.data_type + '/' + 'Q' + str(CLIP_SIZE_Q) + 'R' + str(CLIP_SIZE_R) + 'O' + str(CLIP_SIZE_O) + '_' \
                    + opt.data_type + '_epoch_{epoch:d}_accu_{accu:3.3f}.chkpt'\
                        .format(epoch=epoch_i, accu=100*valid_accu)
                    torch.save(checkpoint, model_name)
                elif opt.save_mode == 'best':
                    if valid_accu >= max_valid_acc:
                        max_valid_acc = valid_accu
                        model_name = opt.save_model + opt.data_type + '/' + 'Q' + str(CLIP_SIZE_Q) + 'R' + str(CLIP_SIZE_R) + 'O' + str(CLIP_SIZE_O) + '_'\
                        + opt.data_type + '_best_2.chkpt'
                        torch.save(checkpoint, model_name)
                        print('    - [Info] The best checkpoint has been updated.')
            model.train()

        ct = ct + 1

    loss_per_epoch = total_loss / ct
    accuracy = n_sample_correct / n_sample_total
    return loss_per_epoch, accuracy, max_valid_acc

def eval_epoch(model, validation_data, device, d_type, writer, check=0):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_sample_total = 0
    n_sample_correct = 0
    ct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, desc='  - (Validation) '):

            # prepare data
            #src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            res_feature, res_feature_pos, obj_feature, obj_st, obj_num, trajectory, \
                 questions, questions_pos, answer = batch

            if ct>check and check>0:
                break
            if d_type == "Trans" or d_type == "Action":
                s = questions.size()
                questions = questions.view(-1, s[2])
                questions_pos = questions_pos.view(-1, s[2])
                res_feature = mc_data_expand(res_feature)
                res_feature_pos = mc_data_expand(res_feature_pos)
                obj_feature = mc_data_expand(obj_feature)
                obj_st = mc_data_expand(obj_st)
                obj_num = mc_data_expand(obj_num)
                trajectory = mc_data_expand(trajectory)
                # answer = mc_data_expand(answer)

            res_feature = res_feature.to(device)
            res_feature_pos = res_feature_pos.to(device)
            obj_feature = obj_feature.to(device)
            obj_st = obj_st.to(device)
            obj_num = obj_num.to(device)
            trajectory = trajectory.to(device)
            questions = questions.to(device)
            questions_pos = questions_pos.to(device)
            answer = answer.to(device)
            gold = answer

            # forward
            #pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            pred = model(res_feature, res_feature_pos, obj_feature, obj_st, obj_num, trajectory, questions, questions_pos, answer)

            if d_type == "Trans" or d_type == "Action":
                loss, n_correct = cal_mc_performance(pred, gold)
            elif d_type == "Count":
                loss, n_correct, out_loss = cal_ct_performance(pred, gold)
            else:
                loss, n_correct = cal_fm_performance(pred, gold)

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
    return loss_per_epoch, accuracy

def train(model, training_data, validation_data, optimizer, device, opt, writer):
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
    max_valid_acc = 0.0
    train_time = []

    model.to(device)

    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_accu, max_valid_acc = train_epoch(
            model, training_data, optimizer, device, epoch_i, d_type=opt.data_type, writer=writer, validation_data=validation_data, opt=opt, max_valid_acc=max_valid_acc)
        valid_accus.append(max_valid_acc)
        elapse = (time.time() - start) / 60
        train_time.append(elapse)
        print('  - (Training)   loss: {loss: 8.4f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(loss=train_loss, accu=100*train_accu, elapse=elapse))
        writer.add_scalar('train/epoch_loss', train_loss, epoch_i)
        writer.add_scalar('train/epoch_acc', train_accu, epoch_i)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, d_type=opt.data_type, writer=writer)
        print('  - (Validation) loss: {loss: 8.4f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(loss=valid_loss, accu=100*valid_accu, elapse=(time.time()-start)/60))
        
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
                model_name = opt.save_model + opt.data_type + '/' + 'Q' + str(CLIP_SIZE_Q) + 'R' + str(CLIP_SIZE_R) + 'O' + str(CLIP_SIZE_O) + '_' \
                + opt.data_type + '_epoch_{epoch:d}_accu_{accu:3.3f}.chkpt'\
                    .format(epoch=epoch_i, accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                if valid_accu >= max(valid_accus):
                    max_valid_acc = valid_accu
                    model_name = opt.save_model + opt.data_type + '/' + 'Q' + str(CLIP_SIZE_Q) + 'R' + str(CLIP_SIZE_R) + 'O' + str(CLIP_SIZE_O) + '_'\
                    + opt.data_type + '_best_2.chkpt'
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The best checkpoint has been updated.')


    cur_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    with open('./log/train_time_'+opt.data_type+'_'+cur_time+'.pkl', 'wb') as f:
        pickle.dump(train_time, f)



def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    # Training Settings
    parser.add_argument('-data_type', type=str, choices=['FrameQA', 'Count', 'Trans', 'Action'], default='FrameQA')
    parser.add_argument('-epoch', type=int, default=40)
    parser.add_argument('-warmup_epoches', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=6)
    parser.add_argument('-learning_rate', type=float, default=0.002)
    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-mode', type=str, default='train')

    # CLIPS
    parser.add_argument('-clip_size_q', type=int, default=3)
    parser.add_argument('-clip_size_r', type=int, default=3)
    parser.add_argument('-clip_size_o', type=int, default=3)

    # Dims
    parser.add_argument('-d_word_vec', type=int, default=300)
    parser.add_argument('-d_res_vec', type=int, default=512)
    parser.add_argument('-d_obj_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=1024)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_hid', type=int, default=200)

    # Heads
    parser.add_argument('-n_head_q', type=int, default=6)
    parser.add_argument('-n_head_v', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=2)

    # SAVING ISSUES
    parser.add_argument('-log', default="./log_tensorboard/")
    parser.add_argument('-save_model', default="./checkpoint/")
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-checkpoint_PATH', type=str, default='')
    parser.add_argument('-video_name', type=str, default='')
    parser.add_argument('-vocab_dir', type=str, default='./vocabulary/')
    parser.add_argument('-valid_steps', type=int, default=-1)


    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Other opt settings
    opt.max_video_seq = MAX_F_NUM
    opt.df_dir = DATAFRAME_DIR
    opt.vc_dir = opt.vocab_dir

    opt.clip_size_q = CLIP_SIZE_Q
    opt.clip_size_r = CLIP_SIZE_R
    opt.clip_size_o = CLIP_SIZE_O

    # Prepare Files
    _name = opt.data_type + '_lr' + str(opt.learning_rate)
    setproctitle(_name)

    log_name = os.path.join(opt.log, opt.data_type, _name, time.strftime("%Y-%m-%d_%H-%M", time.localtime()))
    print('tensorboardX dir',log_name)
    if not os.path.exists(log_name):
        os.makedirs(log_name)
    writer = SummaryWriter(log_name)

    #========= Loading Dataset =========#
    train_dataset, training_data, validation_data = prepare_dataloaders(opt)
    #train_dataset, training_data, validation_data = prepare_tv_dataloaders(opt)

    opt.max_token_seq_len = ((train_dataset.max_length-1) // CLIP_SIZE_Q + 1) * CLIP_SIZE_Q
    print('[Checking]: max_token_seq_len {}, CLIP_SIZE_Q {}, CLIP_SIZE_O {}, CLIP_SIZE_R {}'.format(opt.max_token_seq_len, CLIP_SIZE_Q, CLIP_SIZE_O, CLIP_SIZE_R))
    opt.src_vocab_size = train_dataset.src_vocab_size
    opt.tgt_vocab_size = train_dataset.tgt_vocab_size
    opt.word_matrix = torch.FloatTensor(train_dataset.word_matrix)

    #========= Preparing Model =========#
    # if opt.embs_share_weight:
    #     assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
    #         'The src/tgt word2idx table are different but asked to share word embedding.'

    #print(opt)

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
        dropout=opt.dropout,
        device=device).to(device)



    steps_per_epoch = len(training_data)
    n_warmup_steps = steps_per_epoch*opt.warmup_epoches
    print('steps_per_epoch', steps_per_epoch)
    print('n_warmup_steps', n_warmup_steps)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-08),
        opt.learning_rate, n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt, writer)
    writer.close()

def prepare_tv_dataloaders(opt, val_ratio=0.1):
    # ========= Preparing Training&Validation DataLoader ========= #
    full_dataset = DatasetTGIF(
            dataset_name='train',
            data_type=opt.data_type,
            dataframe_dir=opt.df_dir,
            vocab_dir=opt.vc_dir,
            mode=opt.mode,
            video_name=opt.video_name)

    full_dataset.load_word_vocabulary()

    val_size = int(val_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])
    print(
        'Dataset lengths train/val %d/%d' %
        (len(full_dataset), len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=10,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn,
            shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers= 10,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn)

    return full_dataset, train_loader, valid_loader

def prepare_dataloaders(opt):
    # ========= Preparing DataLoader =========#
    train_dataset = DatasetTGIF(
            dataset_name='train',
            data_type=opt.data_type,
            dataframe_dir=opt.df_dir,
            vocab_dir=opt.vc_dir,
            mode=opt.mode,
            video_name=opt.video_name)

    train_dataset.load_word_vocabulary()

    test_dataset = DatasetTGIF(
            dataset_name='test',
            data_type=opt.data_type,
            dataframe_dir=opt.df_dir,
            vocab_dir=opt.vc_dir,
            mode=opt.mode,
            video_name=opt.video_name)

    test_dataset.share_word_vocabulary_from(train_dataset)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=10,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn,
            shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
            test_dataset,
            num_workers= 1 if opt.mode=='test' else 10,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn)

    return train_dataset, train_loader, valid_loader


if __name__ == '__main__':
    main()

