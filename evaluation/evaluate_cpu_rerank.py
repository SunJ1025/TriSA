import scipy.io
import torch
import numpy as np
import time
from re_ranking import re_ranking


# 根据待查集以及查询集的特征以及标签进行评估
def evaluate(score, ql, gl):
    index = np.argsort(score)  # from small to large

    # 正确的索引
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    # 垃圾索引
    junk_index = np.argwhere(gl == -1)
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i] != 0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


# 读取保存的结果文件
result = scipy.io.loadmat('./weights/sallite_2_drone.mat') # drone_2_sallite sallite_2_drone
query_feature = torch.FloatTensor(result['query_f'])  # query_feature 是所有查询集特征的集合
query_label = result['query_label'][0]

print(result['query_f'].shape)
print(result['gallery_f'].shape)

gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]


test_name = result['name']
query_name = result['query_name']
gallery_name = result['gallery_name']

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0

print('calculate initial distance')
q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
q_q_dist = np.dot(query_feature, np.transpose(query_feature))
g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))

since = time.time()
re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)
time_elapsed = time.time() - since
print('Reranking complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(re_rank[i, :], query_label[i], gallery_label)
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    
CMC = CMC.float()
CMC = CMC/len(query_label)  # average CMC

print('Recall@1: %.2f  Recall@5: %.2f  Recall@10: %.2f  Recall@top1: %.2f  AP: %.2f'\
      % (CMC[0]*100, CMC[4]*100, CMC[9]*100, CMC[round(len(gallery_label)*0.01)]*100, ap/len(query_label)*100))


