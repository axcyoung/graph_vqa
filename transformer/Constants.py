
PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

CLIP_SIZE_Q = 4
CLIP_SIZE_R = 3
CLIP_SIZE_O = 2


#CUDA_VISIBLE_DEVICES=2 python train_origin.py -data_type Trans -learning_rate 0.012 -n_warmup_steps 180000 -batch_size 2

#CUDA_VISIBLE_DEVICES=0 python train_origin.py -data_type FrameQA -learning_rate 0.01 -n_warmup_steps 12000 -batch_size 6 -dropout 0.1

#CUDA_VISIBLE_DEVICES=3 python train_origin.py -data_type Action -learning_rate 0.04 -n_warmup_steps 70000 -batch_size 4 -n_layers 2 -dropout 0.15

#CUDA_VISIBLE_DEVICES=0 python train_origin.py -data_type Count -learning_rate 0.002 -n_warmup_steps 10000 -batch_size 6 -n_layers 2 
