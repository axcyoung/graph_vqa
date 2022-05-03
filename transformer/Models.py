''' nsformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer_a, EncoderLayer_b, EncoderLayer_c, DecoderLayer
from transformer.Graph_video import Graph
from transformer.Constants_tgif import CLIP_SIZE_Q,CLIP_SIZE_R,CLIP_SIZE_O


__author__ = "Yu-Hsiang Huang"

def isnan(inputs, name):
    if torch.any(torch.isnan(inputs)):
        print('isnan!!!!!!!in ',name)

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze_(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze_(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class RNNEncoder(nn.Module):
    def __init__(self, in_dim, rnn_dim, bidirectional=True):
        super(RNNEncoder, self).__init__()

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder = nn.LSTM(in_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=0.18)


    def forward(self, seq, seq_len):
        """
        Args:
            seq: [Float Tensor] (batch_size, max_seq_len, in_dim)
            seq_len: [Tensor] (batch_size)
        return:
            seq representation [Tensor] (batch_size, rnn_dim)
        """
        seq = nn.utils.rnn.pack_padded_sequence(seq, seq_len, batch_first=True,
                                                  enforce_sorted=False)
        self.encoder.flatten_parameters()
        _, (encode_seq, _) = self.encoder(seq)
        if self.bidirectional:
            encode_seq = torch.cat([encode_seq[0], encode_seq[1]], -1)
        encode_seq = self.dropout(encode_seq)

        return encode_seq

class OutputLayers(nn.Module):
    def __init__(self, d_word_vec, d_res_vec, d_obj_vec, module_dim=512):
        super(OutputLayers, self).__init__()

        #self.question_proj = nn.Linear(module_dim, module_dim)

        self.Linear = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(d_word_vec+d_res_vec+d_obj_vec, module_dim),
                                        nn.ELU(),
                                        nn.LayerNorm(module_dim),
                                        nn.Dropout(0.15)
                                        )

    def forward(self, q_fea, r_fea, o_fea):
        #question_embedding = self.question_proj(question_embedding)
        out = torch.cat([q_fea, r_fea, o_fea], -1)
        out = self.Linear(out)

        return out



class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_qseq, len_max_vseq, word_matrix, d_word_vec, d_res_vec, d_obj_vec,
            n_layers, n_head_q, n_head_v,
            d_model, d_inner, d_hid, dropout=0.1,device=0):

        super().__init__()

        self.device=device

        n_q_position = len_max_qseq + 1
        n_v_position = len_max_vseq + 1
        n_o_position = len_max_vseq*5 +1 

        # Dims
        self.d_word_vec = d_word_vec
        self.d_res_vec = d_res_vec
        self.d_obj_vec = d_obj_vec
        self.d_hid = d_hid
        self.n_head_v = n_head_v

        # Q
        self.src_word_emb = nn.Embedding.from_pretrained(word_matrix)
        # R
        self.res_in_proj = nn.Linear(2048, d_res_vec)
        nn.init.xavier_normal_(self.res_in_proj.weight)


        self.fc1 = nn.Linear(len_max_qseq, 1)
        nn.init.xavier_normal_(self.fc1.weight)
        self.relu = torch.nn.ReLU()

        # Position Embed
        self.position_enc_q = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_q_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.position_enc_r = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_v_position, d_res_vec, padding_idx=0),
            freeze=True)

        self.position_enc_o = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_o_position, d_obj_vec, padding_idx=0),
            freeze=True)

        # Spa Matrix
        tmp = np.array([np.power(10000, 2 * (hid_j // 2) / self.d_hid) for hid_j in range(self.d_hid)])
        vec = np.ones([1, 1, 1, 1, 200]) * tmp
        self.position_enc_t = torch.FloatTensor(vec).cuda()
        self.fc_spa_loc = nn.Linear(4*d_hid, d_obj_vec)
        nn.init.xavier_normal_(self.fc_spa_loc.weight)
        self.spa_loc_layernorm = nn.LayerNorm(self.d_obj_vec)

        # Multi-Head
        d_k = int(d_word_vec / n_head_q)
        d_v = d_k
        self.enc_layer_q = EncoderLayer_a(d_word_vec, d_res_vec, d_obj_vec, d_inner, n_head_q, d_k, d_v, dropout=dropout)

        d_k = int(d_res_vec / n_head_v)
        d_v = d_k
        self.enc_layer_r = EncoderLayer_b(d_res_vec, d_inner, n_head_v, d_k, d_v, dropout=dropout, clip_size_sr=CLIP_SIZE_R, clip_size_tg=CLIP_SIZE_Q)
        self.layer_stack_r = nn.ModuleList([
            EncoderLayer_b(d_res_vec, d_inner, n_head_v, d_k, d_v, dropout=dropout, clip_size_sr=CLIP_SIZE_Q, clip_size_tg=CLIP_SIZE_Q)
            for _ in range(n_layers-1)])

        d_k = int(d_obj_vec / n_head_v)
        d_v = d_k
        #self.enc_layer_o = EncoderLayer_c(d_obj_vec, d_inner, n_head_v, d_k, d_v, dropout=dropout)
        self.enc_layer_o = EncoderLayer_c(d_obj_vec, d_inner, n_head_v, d_k, d_v, dropout=dropout, clip_size_sr=CLIP_SIZE_O, clip_size_tg=CLIP_SIZE_Q)
        self.layer_stack_o = nn.ModuleList([
            EncoderLayer_c(d_obj_vec, d_inner, n_head_v, d_k, d_v, dropout=dropout, clip_size_sr=CLIP_SIZE_Q, clip_size_tg=CLIP_SIZE_Q)
            for _ in range(n_layers-1)])

        self.graph_video = Graph(self.d_obj_vec, device=device)
        self.tool_eye = torch.eye(5).to(self.device).float()

        # Aggregate
        self.q_rnn = RNNEncoder(in_dim=d_word_vec, rnn_dim=d_word_vec)
        self.r_rnn = RNNEncoder(in_dim=d_res_vec, rnn_dim=d_res_vec)
        self.o_rnn = RNNEncoder(in_dim=d_obj_vec, rnn_dim=d_obj_vec)

        self.cat_out_layer = OutputLayers(d_word_vec, d_res_vec, d_obj_vec)

    def spatial_vec_comp(self, obj_st, eps=1e-8):

        N = obj_st.size()[2]
        obj_st_reshape = obj_st.view(-1, N, 5)
        mask = obj_st_reshape.sum(-1)!=0 #[B*F, N]
        mask_info = (mask[:,None,:]*mask[:,:,None]).float()
        
        obj_st_reshape = obj_st_reshape.float()
        # [B*F, N, 5] --> [B*F, N]
        x = (obj_st_reshape[:,:,0]+obj_st_reshape[:,:,2])/2
        y = (obj_st_reshape[:,:,1]+obj_st_reshape[:,:,3])/2
        w = (obj_st_reshape[:,:,2]-obj_st_reshape[:,:,0]).abs()
        h = (obj_st_reshape[:,:,3]-obj_st_reshape[:,:,1]).abs()
        
        X_mn = (  ((x[:,None,:]-x[:,:,None]).abs()+eps) / (w[:,:,None]+eps) ).log()*mask_info
        Y_mn = (  ((y[:,None,:]-y[:,:,None]).abs()+eps) / (h[:,:,None]+eps) ).log()*mask_info
        W_mn = (  (w[:,None,:]+eps)/(w[:,:,None]+eps)  ).log()*mask_info
        H_mn = (  (h[:,None,:]+eps)/(h[:,:,None]+eps)  ).log()*mask_info

        p_enc = torch.cat([X_mn.unsqueeze(-1), Y_mn.unsqueeze(-1), W_mn.unsqueeze(-1), H_mn.unsqueeze(-1)], dim=-1)[:,:,:,:,None]/self.position_enc_t

        p_enc[:,:,:,:,0::2].sin_()  # dim 2i
        p_enc[:,:,:,:,1::2].cos_()  # dim 2i+1
        st_feature = p_enc.reshape(-1, N, N, 4*self.d_hid)  # dim 4*d_hid
        st_weight = self.relu(self.fc_spa_loc(st_feature))*mask_info[:,:,:,None]*(1-self.tool_eye[None,:,:,None]) # [B*F, N, N, D]
        
        '''print('obj_st', obj_st.size(), obj_st[0,0])
        print('mask_info', mask_info[0])
        print('x', x[0])
        print('y', y[0])
        print('w', w[0])
        print('h', h[0])
        print('X_mn', X_mn[0])
        print('Y_mn', Y_mn[0])
        print('W_mn', W_mn[0])
        print('H_mn', H_mn[0])
        print('st_weight',st_weight.size(), st_weight[0,:3,:3,:6])'''

        return st_weight


    def forward(self, res_feature, res_feature_pos, obj_feature_input, obj_st, obj_num, trajectory, questions, questions_pos, return_attns=False):

        #print('res_feature',res_feature.size())
        #print('obj_feature_input',obj_feature_input.size())
        #print('questions',questions.size())

        enc_slf_attn_list_q = []
        enc_slf_attn_list_r = []
        enc_slf_attn_list_o = []
        r_q_attn_list = []
        o_q_attn_list = []

        # Obj Graph
        edge_fea = self.spatial_vec_comp(obj_st)
        obj_feature , obj_st_pos = self.graph_video(obj_feature_input, obj_num, trajectory, edge_fea)

        # Res proj
        res_feature_proj = self.res_in_proj(res_feature)


        # -- Prepare masks
        slf_attn_mask_q = get_attn_key_pad_mask(seq_k=questions, seq_q=questions)
        non_pad_mask_q = get_non_pad_mask(questions)
        q_q_attn_mask = get_attn_key_pad_mask(seq_k=questions, seq_q=questions)

        slf_attn_mask_r = get_attn_key_pad_mask(seq_k=res_feature_pos, seq_q=res_feature_pos)
        non_pad_mask_r = get_non_pad_mask(res_feature_pos)
        r_q_attn_mask = get_attn_key_pad_mask(seq_k=res_feature_pos, seq_q=questions)

        slf_attn_mask_o = get_attn_key_pad_mask(seq_k=obj_st_pos, seq_q=obj_st_pos)
        non_pad_mask_o = get_non_pad_mask(obj_st_pos)
        o_q_attn_mask = get_attn_key_pad_mask(seq_k=obj_st_pos, seq_q=questions)

        # -- Forward
        enc_output_q = self.src_word_emb(questions) + self.position_enc_q(questions_pos)
        enc_output_r = res_feature_proj + self.position_enc_r(res_feature_pos)
        enc_output_o = obj_feature + self.position_enc_o(obj_st_pos)

        '''print('questions_pos', questions_pos.size(), questions_pos[0])
        print('res_feature_pos', res_feature_pos.size(), res_feature_pos[0])
        print('obj_st_pos', obj_st_pos.size(), obj_st_pos[0])

        print('obj_feature', obj_feature.size())
        print('self.position_enc_o(obj_st_pos)', self.position_enc_o(obj_st_pos).size())
        print('enc_output_o 0', enc_output_o.size())'''

        encode_q, enc_output_qr, enc_output_qo, enc_slf_attn_q = self.enc_layer_q(
            enc_output_q,
            non_pad_mask=non_pad_mask_q,
            slf_attn_mask=slf_attn_mask_q)
        if return_attns:
            enc_slf_attn_list_q += [enc_slf_attn_q]

        enc_output_r, enc_slf_attn_r, r_q_attn = self.enc_layer_r(
            enc_output_r, enc_output_qr,
            non_pad_mask1=non_pad_mask_r,
            non_pad_mask2=non_pad_mask_q,
            slf_attn_mask=slf_attn_mask_r,
            dec_enc_attn_mask=r_q_attn_mask)

        if return_attns:
            enc_slf_attn_list_r += [enc_slf_attn_r]
            r_q_attn_list += [r_q_attn]

        for enc_layer in self.layer_stack_r:
            enc_output_r, enc_slf_attn_r, r_q_attn = enc_layer(
                enc_output_r, enc_output_qr,
                non_pad_mask1=non_pad_mask_q,
                non_pad_mask2=non_pad_mask_q,
                slf_attn_mask=slf_attn_mask_q,
                dec_enc_attn_mask=q_q_attn_mask)

            if return_attns:
                enc_slf_attn_list_r += [enc_slf_attn_r]
                r_q_attn_list += [r_q_attn]

        

        enc_output_o, enc_slf_attn_o, o_q_attn = self.enc_layer_o(
            enc_output_o, enc_output_qo,
            non_pad_mask1=non_pad_mask_o,
            non_pad_mask2=non_pad_mask_q,
            slf_attn_mask=slf_attn_mask_o,
            dec_enc_attn_mask=o_q_attn_mask)

        if return_attns:
            #enc_slf_attn_list_o += [enc_slf_attn_o]
            o_q_attn_list += [o_q_attn]

        for enc_layer in self.layer_stack_o:
            enc_output_o, enc_slf_attn_o, o_q_attn = enc_layer(
                enc_output_o, enc_output_qo,
                non_pad_mask1=non_pad_mask_q,
                non_pad_mask2=non_pad_mask_q,
                slf_attn_mask=slf_attn_mask_q,
                dec_enc_attn_mask=q_q_attn_mask)

            if return_attns:
                #enc_slf_attn_list_o += [enc_slf_attn_o]
                o_q_attn_list += [o_q_attn]

        
        # Aggregate&Fusion
        q_len_tmp = questions.ne(Constants.PAD).type(torch.long)
        q_len = q_len_tmp.sum(-1)
        
        agg_q = self.q_rnn(encode_q, q_len)
        agg_r = self.r_rnn(enc_output_r, q_len)   
        agg_o = self.o_rnn(enc_output_o, q_len)        

        enc_output = self.cat_out_layer(agg_q, agg_r, agg_o)

        '''#print('questions', questions[:2])
        print('q_len', q_len.size(),'\n', q_len[:2])
        print('encode_q', encode_q.size(),'\n', encode_q[0, q_len[0]-1:q_len[0]+1:,:6])
        print('enc_output_r', enc_output_r.size(),'\n', enc_output_r[0, q_len[0]-1:q_len[0]+1:,:6])
        print('agg_q', agg_q.size(),'\n', agg_q[0,:6])
        print('agg_o', agg_o.size(),'\n', agg_o[0,:6])'''



        '''enc_output = torch.cat((enc_output_r, enc_output_o), -1)
        enc_output = enc_output.permute(0, 2, 1).contiguous()
        enc_output = self.fc1(enc_output)
        enc_output = enc_output.squeeze()'''

        if return_attns:
            print('r_q_attn_list[-1]',r_q_attn_list[-1])
            print('r_q_attn_list[-1]',r_q_attn_list[-1].size())
            return enc_output, r_q_attn_list[-1], o_q_attn_list[-1]
        return enc_output, None

# class Decoder(nn.Module):
#     ''' A decoder model with self attention mechanism. '''
#
#     def __init__(
#             self,
#             n_tgt_vocab, len_max_seq, word_matrix, d_word_vec,
#             n_layers, n_head, d_k, d_v,
#             d_model, d_inner, dropout=0.1):
#
#         super().__init__()
#         n_position = len_max_seq + 1
#
#         self.tgt_word_emb = nn.Embedding.from_pretrained(word_matrix)
#
#         self.position_enc = nn.Embedding.from_pretrained(
#             get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
#             freeze=True)
#
#         self.layer_stack = nn.ModuleList([
#             DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
#             for _ in range(n_layers)])
#
#     def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
#
#         dec_slf_attn_list, dec_enc_attn_list = [], []
#
#         # -- Prepare masks
#         non_pad_mask = get_non_pad_mask(tgt_seq)
#
#         slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
#         slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
#         slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
#
#         dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)
#
#         # -- Forward
#         dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
#
#         for dec_layer in self.layer_stack:
#             dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
#                 dec_output, enc_output,
#                 non_pad_mask=non_pad_mask,
#                 slf_attn_mask=slf_attn_mask,
#                 dec_enc_attn_mask=dec_enc_attn_mask)
#
#             if return_attns:
#                 dec_slf_attn_list += [dec_slf_attn]
#                 dec_enc_attn_list += [dec_enc_attn]
#
#         if return_attns:
#             return dec_output, dec_slf_attn_list, dec_enc_attn_list
#         return dec_output,


class Decoder_mc(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, d_res_vec, d_obj_vec, dropout=0.1):

        super().__init__()

        self.fc = nn.Linear(512, 1)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, enc_output, return_attns=False):

        dec_output = self.fc(enc_output)

        return dec_output

class Decoder_count(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, d_res_vec, d_obj_vec, dropout=0.1):

        super().__init__()

        self.fc = nn.Linear(512, 1)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, enc_output, return_attns=False):

        dec_output = self.fc(enc_output)
        # print(dec_output)
        # dec_output = dec_output * 10
        # dec_output = dec_output.round()
        # dec_output = dec_output.clamp(1, 10)

        return dec_output

class Decoder_frameqa(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, n_tgt_vocab, d_res_vec, d_obj_vec, dropout=0.1):

        super().__init__()

        self.fc = nn.Linear(512, n_tgt_vocab)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, enc_output, return_attns=False):

        dec_output = self.fc(enc_output)
        return dec_output

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, word_matrix, len_max_qseq, len_max_vseq,
            d_word_vec=300, d_res_vec=2048, d_obj_vec=1024, d_model=1024, d_inner=2048,
            n_layers=6, n_head_q=6, n_head_v=8, d_hid=200, data_type='FrameQA', dropout=0.1, device=0):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_qseq=len_max_qseq, len_max_vseq=len_max_vseq, word_matrix=word_matrix,
            d_word_vec=d_word_vec, d_res_vec=d_res_vec, d_obj_vec=d_obj_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head_q=n_head_q, n_head_v=n_head_v, d_hid=d_hid,
            dropout=dropout,device=device)

        if data_type == "Trans" or data_type == "Action":
            self.decoder = Decoder_mc(d_res_vec, d_obj_vec, dropout)
        elif data_type == "Count":
            self.decoder = Decoder_count(d_res_vec, d_obj_vec, dropout)
        else:
            self.decoder = Decoder_frameqa(n_tgt_vocab, d_res_vec, d_obj_vec, dropout)

        #self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        #nn.init.xavier_normal_(self.tgt_word_prj.weight)

        # assert d_model == d_word_vec, \
        # 'To facilitate the residual connections, \
        #  the dimensions of all module outputs shall be the same.'
        #
        # if tgt_emb_prj_weight_sharing:
        #     # Share the weight matrix between target word embedding & the final logit dense layer
        #     self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
        #     self.x_logit_scale = (d_model ** -0.5)
        # else:
        #     self.x_logit_scale = 1.
        #
        # if emb_src_tgt_weight_sharing:
        #     # Share the weight matrix between source & target word embeddings
        #     assert n_src_vocab == n_tgt_vocab, \
        #     "To share word embedding table, the vocabulary size of src/tgt shall be the same."
        #     self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, res_feature, res_feature_pos, obj_feature, obj_st, obj_num, trajectory, questions, questions_pos, answer, return_attns=False):

        # tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(res_feature, res_feature_pos, obj_feature, obj_st, obj_num, trajectory, questions, questions_pos, return_attns)
        dec_output = self.decoder(enc_output)
        #seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        #return seq_logit.view(-1, seq_logit.size(2))
        return dec_output

