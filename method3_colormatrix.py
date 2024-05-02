import cv2
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm


def getFolderFilesName(folder_path):
  files = os.listdir(folder_path)
  file_names = []
  for file in files:
      file_name, file_ext = os.path.splitext(file)
      if file_name.isdigit():
        file_names.append(int(file_name))
  file_names.sort()
  return file_names

download_path = './'
path_query_num=getFolderFilesName(download_path+"query_4186")
path_gallery_num=getFolderFilesName(download_path+"gallery_4186")

path_save = os.path.join(download_path, 'cm')
path_query = os.path.join(download_path, 'query_4186')
path_query_txt = os.path.join(download_path, 'query_txt_4186')
path_gallery = os.path.join(download_path, 'gallery_4186')
num_query = 20
num_gallery = 5000

name_query   = [ path_query +"/"+str(n)+".jpg" for n in path_query_num]
name_gallery = [ path_gallery +"/"+str(n)+".jpg" for n in path_gallery_num]
name_query_txt = [ path_query_txt +"/"+str(n)+".txt" for n in path_query_num]

def box_crop(i):
  query_img_txt_path = name_query_txt[i]
  with open(query_img_txt_path, 'r') as f:
    x, y, w, h = map(int, f.readline().strip().split())
    return x, y, w, h
  
def color_distance(hist1, hist2):
    h_diff = (hist1[0] - hist2[0])
    s_diff = (hist1[1] - hist2[1]) 
    v_diff = (hist1[2] - hist2[2]) 

    h_std_diff = (hist1[3] - hist2[3])
    s_std_diff = (hist1[4] - hist2[4])
    v_std_diff = (hist1[5] - hist2[5])

    h_thirdMoment_diff = (hist1[6] - hist2[6]) 
    s_thirdMoment_diff = (hist1[7] - hist2[7])
    v_thirdMoment_diff = (hist1[8] - hist2[8]) 

    return np.sqrt((h_diff**2 + s_diff**2 + v_diff**2 + h_std_diff**2 + s_std_diff**2 + v_std_diff**2 + h_thirdMoment_diff**2 + s_thirdMoment_diff**2 + v_thirdMoment_diff**2))
    
def color_matrix(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    std_h = np.std(h)
    std_s = np.std(s)
    std_v = np.std(v)
    skew_h = np.mean(np.abs(h - mean_h) ** 3)
    skew_s = np.mean(np.abs(s - mean_s) ** 3)
    skew_v = np.mean(np.abs(v - mean_v) ** 3)
    third_moment_h = skew_h ** (1. / 3)
    third_moment_s = skew_s ** (1. / 3)
    third_moment_v = skew_v ** (1. / 3)
    return [mean_h, mean_s, mean_v, std_h, std_s, std_v, third_moment_h, third_moment_s, third_moment_v]

window_size = 300
overlap = 100

query_hist_list = []
gallery_hist_list = []
for i in tqdm(range(len(name_query)), desc='Processing query images'):
    x, y, w, h = box_crop(i)
    query_img = cv2.imread(name_query[i])[y:y+h, x:x+w]
    query_hist = color_matrix(query_img)
    query_hist_list.append(query_hist)

for j in tqdm(range(len(name_gallery)), desc='Processing gallery images'):
    gallery_img = cv2.imread(name_gallery[j])
    gallery_hist_list.append([])
    for y in range(0, gallery_img.shape[0] - window_size + 1, overlap):
        for x in range(0, gallery_img.shape[1] - window_size + 1, overlap):
            window = gallery_img[y:y+window_size, x:x+window_size]
            gallery_hist = color_matrix(window)
            gallery_hist_list[-1].append(gallery_hist)

distances = [{} for i in range(20)]
for i, qh in tqdm(enumerate(query_hist_list)):
    for j, gh in enumerate(gallery_hist_list):
        minH = 10000000000
        for win in gh:
            distance = color_distance(qh, win)
            minH = min(distance, minH)
        distances[i][j] = minH

with open(path_save+'/rankList_top10.txt', 'w') as f:
    for i in range(num_query):
      sorted_matches = sorted(distances[i].items(), key=lambda x:x[1], reverse=False)
      f.write('Q{}: '.format(i+1))
      cnt=0
      for j in sorted_matches:
        f.write('{} '.format(path_gallery_num[j[0]]))
        cnt+=1
        if cnt>=10:
              break
      f.write('\n')