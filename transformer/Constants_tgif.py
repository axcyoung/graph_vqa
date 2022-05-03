
PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

CLIP_SIZE_Q = 1
CLIP_SIZE_R = 1
CLIP_SIZE_O = 1

TGIF_DATA_DIR = '/home1/gumao/MM19/cvpr19/dataset'
VIDEO_DATA_DIR = "/home1/gumao/from_other_service/server1/data/TGIF-QA/gif_imgs"
DATAFRAME_DIR = '/home1/gumao/MM19/cvpr19/dataset/tgif-qa-master/dataset'
 
MAX_F_NUM = 60
MAX_Q_LEN = 40
MAX_O_F_NUM = 24

#CUDA_VISIBLE_DEVICES=1 python train_tgif.py -data_type Trans -learning_rate 0.012 -n_warmup_steps 180000 -batch_size 2

#CUDA_VISIBLE_DEVICES=0 python train_tgif.py -data_type FrameQA -learning_rate 0.01 -n_warmup_steps 12000 -batch_size 6 -dropout 0.1

#CUDA_VISIBLE_DEVICES=3 python train_tgif.py -data_type Action -learning_rate 0.04 -n_warmup_steps 70000 -batch_size 4 -n_layers 2 -dropout 0.15

#CUDA_VISIBLE_DEVICES=0 python train_tgif.py -data_type Count -learning_rate 0.002 -n_warmup_steps 10000 -batch_size 6 -n_layers 2 
