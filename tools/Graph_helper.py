import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def Euclidean_distance(l1, l2):
    return math.sqrt(np.sum((l1-l2)*(l1-l2)))

def iou_and_area(loc1, loc2, delta=1e-6, print_op=False):
    left_top_x1, left_top_y1, right_bottom_x1, right_bottom_y1 = loc1[0], loc1[1], loc1[2], loc1[3]
    left_top_x2, left_top_y2, right_bottom_x2, right_bottom_y2 = loc2[0], loc2[1], loc2[2], loc2[3]
    area1 = (left_top_x1 - right_bottom_x1) * (left_top_y1 - right_bottom_y1)
    area2 = (left_top_x2 - right_bottom_x2) * (left_top_y2 - right_bottom_y2)
    
    area_compare = math.exp(- (abs(min(area1,area2) *1.0 /(max(area1,area2)+delta)-1)))
    #calculate overlap area of the rectangle
    if left_top_x1 >= right_bottom_x2 or right_bottom_x1 <= left_top_x2 or \
                left_top_y1 >= right_bottom_y2 or right_bottom_y1 <= left_top_y2:
        overlap_area = 0.0
        if print_op:
        	print('[Not Overlap] area1:{}, area2:{}'.format(area1,area2))
    else:
        delta_x = min(abs(left_top_x1 - right_bottom_x2), abs(right_bottom_x1 - left_top_x2))
        delta_y = min(abs(left_top_y1 - right_bottom_y2), abs(right_bottom_y1 - left_top_y2))
        overlap_area = delta_x*delta_y
        if print_op:
            print('[Overlap] area1:{},area2:{},delta_x:{},delta_y:{},overlap_area:{}, total_area:{}'.format(area1, area2, delta_x, delta_y, overlap_area, area1 + area2 - overlap_area))
    IoU = overlap_area*1.0/(area1 + area2 - overlap_area+delta)
    if print_op:
        print('IoU', IoU, 'area_compare', area_compare)
    return IoU, area_compare

def get_video_trajectory(input_video_feature, input_video_location, mask):
    
    forward_traj = _get_video_trajectory(input_video_feature, input_video_location, mask)

    frame_num = 0
    for i in range(len(mask)):
    	if mask[i]!=0:
    		frame_num=i+1
    

    _backward_traj = _get_video_trajectory(input_video_feature[:frame_num][::-1], input_video_location[:frame_num][::-1], mask[:frame_num][::-1])
    

    feature_shape = input_video_feature.shape
    F, N, D = feature_shape[0], feature_shape[1], feature_shape[-1]
    tool_trajectory = np.ones((F-frame_num, N))*-1
    backward_traj = np.concatenate((_backward_traj[::-1], tool_trajectory), axis=0)

    '''print('F', input_video_feature.shape[0])
    print('mask', mask)
    print('frame_num', frame_num)
    print('_backward_traj', _backward_traj)
    print('_backward_traj', _backward_traj.shape)
    print('backward_traj', backward_traj)
    print('backward_traj', backward_traj.shape)
    print('forward_traj', forward_traj)'''

    return [forward_traj, backward_traj]


def _get_video_trajectory(input_video_feature, input_video_location, mask, delta=1e-6, filter_threshold=0.6):
    '''
        input:

        input_video_feature: float, [F, N, D]
        input_video_location: float, [F, N, 5]
        mask: [F]

        return:

        trajectory: float [F, N]

    '''

    feature_shape = input_video_feature.shape
    F, N, D = feature_shape[0], feature_shape[1], feature_shape[-1]
    first_frame_feature = input_video_feature[0]#[N, D]
    trajectory = np.ones((F, N))*-1
    trajectory[0] = np.arange(N)
    score_zeros = np.zeros((N,N))


    non_pad_nums = [sum(np.sum(x, -1)!=0) for x in input_video_feature]
    non_pad_num_in_frame0 = sum(np.sum(first_frame_feature, -1)!=0)
    traj_id = np.ones((F,N), dtype=np.int)*-1
    traj_id[0] = [x if x<non_pad_num_in_frame0 else -1 for x in range(N)]

    '''print('non_pad_num_in_frame0', non_pad_num_in_frame0)
    print('non_pad_nums', non_pad_nums)
    print('traj_id[0]', traj_id[0])
    print('traj_id[1]', traj_id[1])'''

    for frame_idx in range(1,F): 
        #print('frame_idx',frame_idx, 'non_pad_num', non_pad_nums[frame_idx])       
        #mask
        mask_cond_sender = np.array([1.0 if i<mask[0] else 0.0 for i in range(N)])
        mask_cond_receiver = np.array([1.0 if i<mask[frame_idx] else 0.0 for i in range(N)])
        mask_score = mask_cond_sender[:,None]*mask_cond_receiver[None,:]

        score = np.zeros((N,N))
        app_matrix, iou_matrix, area_matrix = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))
        for sender_idx in range(non_pad_num_in_frame0):

            sender_feature = first_frame_feature[sender_idx]
            sender_loc = input_video_location[0][sender_idx]

            for receiver_idx in range(non_pad_nums[frame_idx]):

                receiver_feature = input_video_feature[frame_idx][receiver_idx]
                app_matrix[sender_idx][receiver_idx]=Euclidean_distance(sender_feature, receiver_feature)
                iou_matrix[sender_idx][receiver_idx], area_matrix[sender_idx][receiver_idx] = \
                        iou_and_area(sender_loc, input_video_location[frame_idx][receiver_idx])

        iou_matrix = iou_matrix*mask_score
        area_matrix = area_matrix*mask_score
        app_matrix = np.exp(-(app_matrix/(np.amax(app_matrix)+delta)))
        app_matrix = app_matrix*mask_score
        score = (app_matrix+iou_matrix+area_matrix)/3 #[N,N]
        
        # get traj id according to max score
        tmp_cur_traj_id = np.argmax(score, axis=0)
        traj_id[frame_idx] = [tmp_cur_traj_id[i] if i<non_pad_nums[frame_idx] else -1 for i in range(N)]
        cur_frame_max_score = np.max(score, axis=0)
        tmp_score_for_compare = np.zeros(N)
        # filter and match
        for i in range(non_pad_nums[frame_idx]):
            tmp_max_score = cur_frame_max_score[i]
            if tmp_max_score<filter_threshold:
                trajectory[frame_idx][i] = -1
                traj_id[frame_idx][i] = -1
                continue
            # search its origin in frame_idx-1
            for j in range(non_pad_nums[frame_idx-1]):
                if traj_id[frame_idx-1][j]!=-1 and traj_id[frame_idx-1][j] == traj_id[frame_idx][i]:
                    trajectory[frame_idx][i] = j

        '''print('app_matrix',app_matrix)
        print('iou_matrix',iou_matrix)
        print('area_matrix',area_matrix)
        print('score',score)
        print('traj_id', traj_id[frame_idx])
        print('trajectory[frame_idx]',trajectory[frame_idx])'''

    # zeros the first frame
    trajectory[0] = np.ones(N)*-2
    return trajectory



if __name__ == '__main__':
    loc1 = [0,0, 2,4]
    loc2 = [-1,-1, 1,2]
    print('loc1', loc1, '\nloc2',loc2)
    iou_and_area(loc1, loc2, print_op=True)





