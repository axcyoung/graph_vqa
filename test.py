import torch
import numpy as np
from dataset import DatasetTGIF, paired_collate_fn
a = DatasetTGIF(dataset_name='train',data_type='FrameQA',dataframe_dir='/home1/jinweike/projects/cvpr19/dataset/tgif-qa-master/dataset',vocab_dir='./vocabulary')
a.load_word_vocabulary()
b = torch.utils.data.DataLoader(a,num_workers=2,batch_size=6,collate_fn=paired_collate_fn)
d_hid = 200
fc = torch.nn.Linear(5*d_hid, 1)
relu = torch.nn.ReLU()

for data in b:
    res_feature,res_feature_pos,obj_feature,obj_feature_pos,obj_st,obj_st_pos,questions,questions_pos,answer = data
    print(res_feature.shape)
    print(res_feature_pos.shape)
    print(obj_feature.shape)
    print(obj_feature_pos.shape)
    print(obj_st.shape)
    print(obj_st_pos.shape)
    #print(obj_st)
    # print(obj_st_pos)
    # print(questions)
    # print(questions_pos)
    # print([b.dataset.idx2word[i] for i in questions.tolist()[0]])
    # print(answer)
    l = obj_st.shape[1]
    o1 = obj_st.clone().unsqueeze(1).expand(-1, l, -1, -1)
    o2 = obj_st.clone().unsqueeze(2).expand(-1, -1, l, -1)
    o3 = obj_st.clone().unsqueeze(2).expand(-1, -1, l, -1)
    o2[:,:,:,2:] = torch.zeros([o2.shape[0],o2.shape[1],o2.shape[2],3])
    out = o1 - o2
    out = torch.abs(out) +0.01
    o3[:,:,:,:2] = o3[:,:,:,2:4]
    o3 = o3 + 0.01
    out = out / o3
    out = torch.log(out)
    non_pad_mask = obj_st_pos.ne(0).type(torch.float)
    m1 = non_pad_mask.unsqueeze(-1)
    m2 = non_pad_mask.unsqueeze(-2)
    m3 = m1*m2
    m = m3.unsqueeze(-1)
    out = out*m

    size = out.size()
    tmp = np.array([np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)])
    vec = np.ones([size[0], size[1], size[2], size[3], 200]) * tmp
    vec_t = torch.Tensor(vec)
    p_enc = out.unsqueeze(-1) / vec_t
    p_enc[:, :, :, :, 0::2] = torch.sin(p_enc[:, :, :, :, 0::2])  # dim 2i
    p_enc[:, :, :, :, 1::2] = torch.cos(p_enc[:, :, :, :, 1::2])  # dim 2i+1
    st_feature = p_enc.reshape(size[0], size[1], size[2], -1)  # dim 5*d_hid
    st_weight = relu(fc(st_feature))
    st_weight = st_weight.squeeze()
    st_weight = st_weight*m3
    print(st_weight)
    print(st_weight.shape)

    break
#print(len(b.dataset.idx2word))
#print(len(b.dataset.word_matrix))
