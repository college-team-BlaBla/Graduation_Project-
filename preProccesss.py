from PIL import Image
import numpy as np
import download
import argparse
import requests
import cv2

import pathlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model_size = "s"
n_px = 32
n_gpu = 1
n_sub_batch = 1
def normalize_img(img):
  return img/127.5 - 1

def squared_euclidean_distance_np(a,b):
  b = b.T
  a2 = np.sum(np.square(a),axis=1)
  b2 = np.sum(np.square(b),axis=0)
  ab = np.matmul(a,b)
  d = a2[:,None] - 2*ab + b2[None,:]
  return d

def color_quantize_np(x, clusters):
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance_np(x, clusters)
    return np.argmin(d,axis=1)


#get images

im = Image.open(requests.get('https://cdn.discordapp.com/attachments/844823862888759326/846904119414882324/MTTARANAKI-AGORAjpg.png', stream=True).raw)

# Resize original images to n_px by n_px


dim = (n_px, n_px)

x = np.zeros((n_gpu * n_sub_batch, n_px, n_px, 3), dtype=np.uint8)

#may use for loop
img_np = cv2.imread('1.png')  # reads an image in the BGR format
img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # BGR -> RGB
H, W, C = img_np.shape
D = min(H, W)
img_np = img_np[:D, :D, :C]  # get square piece of image
x[0] = cv2.resize(img_np, dim, interpolation=cv2.INTER_AREA)  # resize to n_px by n_px

f, axarr = plt.subplots(1,4,dpi=64)

i = 0
for img in x:
  print('hello')
  axarr[i].axis('off')
  axarr[i].imshow(img)
  i += 1

color_cluster_path = "clusters/s/kmeans_centers.npy"
clusters = np.load(color_cluster_path)  # get color clusters
x_norm = normalize_img(x)  # normalize pixels values to -1 to +1

samples = color_quantize_np(x_norm, clusters).reshape(x_norm.shape[:-1])  # map pixels to closest color cluster

n_px_crop = 12
primers = samples.reshape(-1, n_px * n_px)[:, :n_px_crop * n_px]  # crop top n_px_crop rows

#visualize samples and crops with Image-GPT color palette. Should look similar to original resized images


samples_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px, n_px, 3]).astype(np.uint8) for s in samples] # convert color clusters back to pixels
primers_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px_crop,n_px, 3]).astype(np.uint8) for s in primers] # convert color clusters back to pixels

f, axarr2 = plt.subplots(1,4,dpi=64)
i = 0
for img in samples_img:
    axarr2[i].axis('off')
    axarr2[i].imshow(img)
    i += 1

f, axarr = plt.subplots(1,4,dpi=64)
i = 0
for img in primers_img:
    axarr[i].axis('off')
    axarr[i].imshow(img)
    i += 1
