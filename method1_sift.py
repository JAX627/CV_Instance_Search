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

path_save = os.path.join(download_path, 'sift')
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
  
#######################################################################################################################
query_kps = [None for _ in range(num_query)]
query_descs = [None for _ in range(len(name_query))]
sift = cv2.xfeatures2d.SIFT_create()

for i in range(len(name_query)):
  query_name = os.path.splitext(os.path.basename(name_query[i]))[0]

  x, y, w, h = box_crop(i)
  query_img = cv2.imread(name_query[i])[y:y+h, x:x+w]
  query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
  query_img_gray = cv2.resize(query_img_gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
  query_kp, query_desc = sift.detectAndCompute(query_img_gray, None)

  query_descs[i]=np.float32(query_desc)

gallery_descs = [None for _ in range(num_gallery)]
sift = cv2.xfeatures2d.SIFT_create()

for j in tqdm(range(num_gallery), desc='Processing gallery images'):
    gallery_img = cv2.imread(name_gallery[j])
    gallery_img_gray = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2GRAY)
    gallery_img_gray = cv2.resize(gallery_img_gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gallery_kp, gallery_desc = sift.detectAndCompute(gallery_img_gray, None) 
    gallery_descs[j] = np.float32(gallery_desc)

matches_dict = {}
for i in range(num_query):
    query_desc = query_descs[i]
    matches_dict[i]={}
    mii=1000
    for j in tqdm(range(num_gallery), desc='Processing gallery images for query {}'.format(i)):
        gallery_desc = gallery_descs[j]

        flann = cv2.FlannBasedMatcher()
        matches = flann.knnMatch(query_desc, gallery_desc, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance and m.distance>0.01*n.distance:
                mii=min(m.distance,mii)
                good_matches.append(m)
        matches_dict[i][j] = len(good_matches)

with open(path_save+'/rankList_top10.txt', 'w') as f:
    for i in range(num_query):
      sorted_matches = sorted(matches_dict[i].items(), key=lambda x:x[1], reverse=True)
      f.write('Q{}: '.format(i+1))
      cnt=0
      for j in sorted_matches:
        f.write('{} '.format(path_gallery_num[j[0]]))
        cnt+=1
        if cnt>=10:
              break
      f.write('\n')