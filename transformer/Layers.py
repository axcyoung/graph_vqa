''' yers '''
import torch.nn as nn
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward, InteracionUnit
import torch
from transformer.Constants_tgif import CLIP_SIZE_Q,CLIP_SIZE_R,CLIP_SIZE_O

__author__ = "Yu-Hsiang Huang"




class EncoderLayer_a(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_res_vec, d_obj_vec, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer_a, self).__init__()
        self.d_res_vec = d_res_vec
        self.d_obj_vec = d_obj_vec
        self.fc_r = nn.Linear(d_model, d_res_vec)
        self.fc_o = nn.Linear(d_model, d_obj_vec)
        self.slf_attn = InteracionUnit(
            n_head, d_model, d_k, d_v, dropout=dropout, clip_size_q=CLIP_SIZE_Q, clip_size_k=CLIP_SIZE_Q)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        encode_q = enc_output

        enc_output = self.pos_ffn(enc_output)
        enc_output_r = self.fc_r(enc_output)
        enc_output_o = self.fc_o(enc_output)
        enc_output_r *= non_pad_mask
        enc_output_o *= non_pad_mask

        return encode_q, enc_output_r, enc_output_o, enc_slf_attn


class EncoderLayer_b(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, clip_size_sr=None, clip_size_tg=None):
        super(EncoderLayer_b, self).__init__()
        self.slf_attn = InteracionUnit(n_head, d_model, d_k, d_v, dropout=dropout, clip_size_q=clip_size_sr, clip_size_k=clip_size_sr)
        self.enc_attn = InteracionUnit(n_head, d_model, d_k, d_v, dropout=dropout, clip_size_q=clip_size_tg, clip_size_k=clip_size_sr)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, tgt_output, non_pad_mask1=None, non_pad_mask2=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask1

        enc_output, enc_tgt_attn = self.enc_attn(
            tgt_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        enc_output *= non_pad_mask2

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask2

        return enc_output, enc_slf_attn, enc_tgt_attn

class EncoderLayer_c(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, clip_size_sr=None, clip_size_tg=None):
        super(EncoderLayer_c, self).__init__()
        #self.slf_attn = InteracionUnit(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = InteracionUnit(n_head, d_model, d_k, d_v, dropout=dropout, clip_size_q=clip_size_tg, clip_size_k=clip_size_sr)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, tgt_output, non_pad_mask1=None, non_pad_mask2=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        enc_slf_attn = None
        #enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = enc_input * non_pad_mask1

        enc_output, enc_tgt_attn = self.enc_attn(
            tgt_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        enc_output *= non_pad_mask2

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask2

        return enc_output, enc_slf_attn, enc_tgt_attn

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn

