### choose n classes.
No_class = 2

lbp_n_points = 8
lbp_n_radius = 2
lbp_Method = 'nri_uniform'
global_color = []
color_map_file_s = 'color_map_s1.pkl'
color_map_file_l = 'color_map_s1.pkl'
feature_file_s = 'feature_s1.pkl'

import os
import numpy as np
import sys
import imageio
from skimage.feature import local_binary_pattern
from sklearn.cluster import MiniBatchKMeans
from skimage.transform import resize
from sklearn.feature_extraction.text import TfidfTransformer
import pickle



def picture_process_train_word(pic_temp):
    block_hist = []
    with open(color_map_file_l, 'rb') as file:  
        Color_Map = pickle.load(file)
    pic_matrix = imageio.imread(pic_temp)
    width, height, dim = pic_matrix.shape
    if width < 64:
         return
    if height < 64:
         return
    num_blocks = int((width -32)/32) *int((height -32)/32)
    while num_blocks > 10:
        print(num_blocks)
        for i in range(int((width -32)/32)):
            for j in range(int((height -32)/32)):
                Block = pic_matrix[i*32:i*32+64,j*32:j*32+64,:]
                Block_gray = 0.2989 * Block[:,:,0] + 0.5870 * Block[:,:,1] + 0.1140 * Block[:,:,2]
                lbp = local_binary_pattern(Block_gray, lbp_n_points, lbp_n_radius, lbp_Method).reshape(1,-1).astype(int)
                lbp_his = np.zeros((1,59))
                color_his = np.zeros((1,30))
                for k in range(59):
                    lbp_his[0,k] = list(lbp[0]).count(k)
                for m in range(64):
                    for n in range(64):
                        ans = Color_Map.predict([[Block[m,n,0],Block[m,n,1],Block[m,n,2]]])
                        color_his[0,ans[0]] = color_his[0,ans[0]] + 1
                his = np.append(lbp_his,color_his)
                block_hist.append(his)
        if width >= (64 *1.25):
            width = width /1.25
        if height >= (64*1.25):
            height = height/1.25
        pic_matrix = resize(pic_matrix ,(int(width),int(height),3))
        width, height, dim = pic_matrix.shape
        num_blocks = int((width -32)/32) *int((height -32)/32)
        print(width,height,num_blocks)
        print(int((width -32)/32))
        print(int((height -32)/32))
    return block_hist


def Train_C_map_read_pic(pic_temp):
    print(pic_temp)
    pic_matrix = imageio.imread(pic_temp)
    width, height, dimm = pic_matrix.shape
    for i in range(int(width)):
        for j in range(int(height)):
            global_color.append([pic_matrix[i,j,0],pic_matrix[i,j,1],pic_matrix[i,j,2]])

def Train_C_map_cal():
    sc = MiniBatchKMeans(20).fit(global_color)
    center = sc.cluster_centers_
    with open(color_map_file_s, 'wb') as file:  
        pickle.dump(sc, file)

def extract():
    dir_path = '256_ObjectCategories'
    files_in_path = [os.path.join(dir_path,x) for x in os.listdir(dir_path)]
    dir_in_path = [x for x in files_in_path if os.path.isdir(x)]
    dir = [os.path.split(x)[1] for x in dir_in_path]
    np.random.seed()
    visited = []
    pic_temp = []
    pic_all = []
    if No_class > len(dir):
        print("too many classed!")
        return 
    for i in range(No_class):
        dir_id = int(np.floor(np.random.random() * len(dir) ))
        while dir_id in visited:
            dir_id = int(np.floor(np.random.random() * len(dir) ))
        visited.append(dir_id)
    for i in range(No_class):
        pic_temp = pic_temp + [os.path.join(dir_in_path[visited[i]], x) for x in os.listdir(dir_in_path[visited[i]])]

        
######## used for train color map ####uncomment this part to train color map may take quite a long time!!!!
    # for i in range(256):
    #     pic_all = pic_all + [os.path.join(dir_in_path[i], x) for x in os.listdir(dir_in_path[i])]
    # for j in range(len(pic_all)):
    #     Train_C_map_read_pic(pic_all[j])
    # Train_C_map_cal()
########used for trainging test!!!
    # for j in range(len(pic_temp)):
    #     Train_C_map_read_pic(pic_temp[j])
    # Train_C_map_cal()

####### used for tarin  picture to obtain feature
    # block_hist_all = []
    # for j in range(len(pic_all)):
    #     block_hist_all = block_hist_all + (picture_process_train_word(pic_all[j]))
    # feature = MiniBatchKMeans(1000).fit(block_hist_all)
    # with open(feature_file_s, 'wb') as file:  
    #     pickle.dump(feature, file)
    with open(feature_file_s, 'rb') as file:  
        feature_learnt = pickle.load(file)
    block_hist_cur = []
    for j in range(1):
        block_hist_cur = block_hist_cur + (picture_process_train_word(pic_temp[j]))
        result = feature_learnt.predict(block_hist_cur)

### caluculate p.
    # tfidf = transformer.fit_transform(result)


####### used for calculte the cluster imformation
    # for j in range(len(pic_temp)):
    #     Train_C_map_read_pic(pic_temp[j])




#main 
extract()