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

path_save = os.path.join(download_path, 'ch')
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
  
window_size = 300
overlap = 100
distances = [{} for i in range(20)]
hist_size = [180, 256]
hist_range = [0, 180, 0, 256]
alpha = 1.25 
beta = 50 

for i in range(20):
    query_name = os.path.splitext(os.path.basename(name_query[i]))[0]
    x, y, w, h = box_crop(i)
    query_img = cv2.imread(name_query[i])[y:y+h, x:x+w]
    hsv = cv2.cvtColor(query_img, cv2.COLOR_BGR2HSV)
    hsv = cv2.convertScaleAbs(hsv, alpha=alpha, beta=beta)
    
    query_hist = cv2.calcHist([hsv], [0, 1], None, hist_size, hist_range)
    cv2.normalize(query_hist, query_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    for j in tqdm(range(num_gallery)):
        gallery_img = cv2.imread(name_gallery[j])
        hsv = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2HSV)
        hsv = cv2.convertScaleAbs(hsv, alpha=alpha, beta=beta)
        minH = 10000000000
        for y in range(0, hsv.shape[0] - window_size + 1, overlap):
            for x in range(0, hsv.shape[1] - window_size + 1, overlap):
                window = hsv[y:y+window_size, x:x+window_size]
                hist = cv2.calcHist([window], [0, 1], None, hist_size, hist_range)
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                distance = cv2.compareHist(query_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
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