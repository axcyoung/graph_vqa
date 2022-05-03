import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import math

def isnan(inputs, name):
    if torch.any(torch.isnan(inputs)):
        print('isnan!!!!!!!in ',name)


def xprint(s, x, print_opt=False):
    if print_opt:
         print(s,x)


class Graph(nn.Module):
    ''' calculate graph '''

    def __init__(self, hid_dim, object_num=5, device=0, graph_dropout=0.2):
        super().__init__()
        self.device=device
        self.loop_num = 2
        self.N = object_num
        self.mrcnn_dim = 1024
        self.hid_dim = hid_dim
        #define fcs
        self.hid_fc = nn.Linear(self.mrcnn_dim, self.hid_dim)
        nn.init.xavier_normal_(self.hid_fc.weight)
        self.receiver_fc = nn.Linear(self.hid_dim, self.hid_dim)
        nn.init.xavier_normal_(self.receiver_fc.weight)
        self.sender_fc = nn.Linear(self.hid_dim, self.hid_dim)
        nn.init.xavier_normal_(self.sender_fc.weight)
        self.v_fc = nn.Linear(3*self.hid_dim,1)
        nn.init.xavier_normal_(self.v_fc.weight)
        self.st_fc = nn.Linear(self.hid_dim,self.hid_dim)
        nn.init.xavier_normal_(self.st_fc.weight)

        self.edge_fc = nn.Linear(self.hid_dim,self.hid_dim)
        nn.init.xavier_normal_(self.edge_fc.weight)


        self.dropout = nn.Dropout(graph_dropout)

        #layernorm
        self.layer_norm_sender = nn.LayerNorm(self.hid_dim)
        self.layer_norm_receiver = nn.LayerNorm(self.hid_dim)
        self.layer_norm_edge = nn.LayerNorm(self.hid_dim)

        self.layer_norm_inter_message = nn.LayerNorm(self.hid_dim)
        self.layer_norm_st_message = nn.LayerNorm(self.hid_dim)

        self.layer_norm_out = nn.LayerNorm(self.hid_dim)

        # tool
        self.tool_arange = torch.arange(self.N).to(self.device)#[N]
        self.tool_eye = torch.eye(self.N).to(self.device)#[N,N]

    def forward(self, video_feature, mask, trajectory, edge_fea, att_threshold=0.1):
        '''
            video_feature: float, [B, F, N, D]
            mask: int, [B, F]
            trajectory: float, [B,2,F,N]
        '''

        forward_trajectory = trajectory[:,0,:,:].contiguous()
        backward_trajectory = trajectory[:,1,:,:].contiguous()

        #get information in the internal env of the same frame.
        B = video_feature.size()[0]
        F = video_feature.size()[1]
        mask_reshape = mask.view(-1)
        video_feature = video_feature.view(-1, self.N, self.mrcnn_dim)#[B*F,N,D]
        node_feature = self.hid_fc(video_feature)#[-1,N,D]
        

        # reshape tool
        tool_torch = self.tool_arange[None,:].clone().expand(B*F,-1)#[B*F,N]

        xprint('mask_reshape', mask_reshape[0])
        mask_info = self.tool_arange[None,:] < mask_reshape[:,None] #[-1, N]
        mask_info_tile = mask_info[:,None,:]*mask_info[:,:,None] * (1-self.tool_eye[None,:,:].byte())

        #produce forward_traj info.
        forward_trajectory = forward_trajectory.view(-1, self.N).long()
        Ad = tool_torch[:,None,:] == forward_trajectory[:,:,None]#[-1,N,N]
        Ad = Ad.float()

        #produce backward_traj info.
        backward_trajectory = backward_trajectory.view(-1, self.N).long()
        bw_Ad = tool_torch[:,None,:] == backward_trajectory[:,:,None]#[-1,N,N]
        bw_Ad = bw_Ad.float()

        for i in range(self.loop_num):
            sender_feature = self.sender_fc(node_feature)
            sender_feature = sender_feature.clone().unsqueeze(1).expand(-1,self.N,-1,-1).contiguous()#[-1,N,N,D]
            sender_feature = self.layer_norm_sender(sender_feature)

            receiver_feature = self.receiver_fc(node_feature)
            receiver_feature = receiver_feature.clone().unsqueeze(2).expand(-1,-1,self.N,-1).contiguous()#[-1,N,N,D]
            receiver_feature = self.layer_norm_receiver(receiver_feature)

            edge_fea = self.layer_norm_edge(self.edge_fc(edge_fea))
            edge_fea *= (mask_info_tile[:,:,:,None].float())

            gamma = self.v_fc(torch.cat((receiver_feature, sender_feature, edge_fea),-1)).squeeze(-1)#[-1,N,N]
            xprint('idx', i)
            #xprint('gamma1', gamma[0])
            gamma = torch.exp(torch.nn.functional.leaky_relu(gamma))
            xprint('gamma2', gamma[0])
            gamma_mask = torch.ones_like(mask_info_tile).float()*(1e-5)
            gamma = torch.where(mask_info_tile, gamma, gamma_mask)#[-1,N,N]
            
            att = gamma/(torch.sum(gamma,-1)[:,:,None].clone().expand(-1,-1,self.N))#[-1,N,N]
            att_mask_info = mask_info_tile
            att_mask = torch.zeros_like(mask_info_tile).float()
            att = torch.where(att_mask_info, att, att_mask)

            xprint('att:', att[0])
            #filter att lower than threshold
            att *= (att>att_threshold).float()
            xprint('att filter:', att[0])

            #calculate internal messange
            message_graph = att[:,:,:,None]*sender_feature
            message_inter = torch.sum(message_graph, 2)#[-1,N,D]
            xprint('message_inter1', message_inter[0,:,:6])
            message_inter = self.layer_norm_inter_message(message_inter)
            message_inter *= mask_info[:,:,None].float()
            xprint('message_inter2', message_inter[0,:,:6])

            xprint('edge_fea', edge_fea[0,:,:,:6])
            edge_fea = self.layer_norm_edge(message_graph+edge_fea)
            xprint('update edge_fea', edge_fea[0,:,:,:6])

            #calculate st message (fw)
            # Right Move
            remake_node_feature = node_feature.view(-1,F,self.N,self.hid_dim)[:,:-1,:,:]
            tmp_node_feature = remake_node_feature[:,0,:,:].unsqueeze(1)
            remake_node_feature = torch.cat((tmp_node_feature, remake_node_feature),1).view(-1,self.N,self.hid_dim)#[-1,N,D]
            xprint('Ad', Ad[1])
            xprint('remake_node_feature', remake_node_feature[1,:,:6])
            # BMM
            st_message = torch.bmm(Ad,remake_node_feature)#[-1,N,D]
            xprint('st_message1', st_message[1,:,:6])
            st_message = self.layer_norm_st_message(st_message)
            st_message *= mask_info[:,:,None].float()
            xprint('st_message2', st_message[1,:,:6])

            #calculate st message (bw)
            # Left Move
            bw_remake_node_feature = node_feature.view(-1,F,self.N,self.hid_dim)[:,1:,:,:]
            tmp_node_feature = bw_remake_node_feature[:,-1,:,:].unsqueeze(1)
            bw_remake_node_feature = torch.cat((bw_remake_node_feature, tmp_node_feature),1).view(-1,self.N,self.hid_dim)#[-1,N,D]
            xprint('bw_Ad', bw_Ad[0])
            xprint('bw_remake_node_feature', bw_remake_node_feature[0,:,:6])
            # BMM
            bw_st_message = torch.bmm(bw_Ad,bw_remake_node_feature)#[-1,N,D]
            xprint('bw_st_message1', bw_st_message[0,:,:6])
            bw_st_message = self.layer_norm_st_message(bw_st_message)
            bw_st_message *= mask_info[:,:,None].float()
            xprint('bw_st_message2', bw_st_message[0,:,:6])

            xprint('torch.nn.functional.tanh(message_inter)', torch.nn.functional.tanh(message_inter)[0,:,:6])
            xprint('torch.nn.functional.tanh(st_message)', torch.nn.functional.tanh(st_message+bw_st_message)[1,:,:6])

            update_fea = torch.nn.functional.tanh(message_inter) + torch.nn.functional.tanh(st_message+bw_st_message)
            update_fea = self.dropout(update_fea)
            node_feature =self.layer_norm_out(node_feature+update_fea)


        graph_mask = tool_torch < mask_reshape[:,None]#[B*F,N]

        graph_feature = node_feature.view(-1,F*self.N,self.hid_dim)
        graph_mask = graph_mask.view(-1,F*self.N).long()


        return graph_feature, graph_mask 





