import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
import random
import argparse
import sys
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE

#train =60000,test = 10000 1-784-1
#id 0 onwards

train_data_read = pd.read_csv("train.csv")
train_y = train_data_read.iloc[:,785].values
train_X = train_data_read.iloc[:,1:785].values
train_data_read = pd.read_csv("test.csv")
test_y = train_data_read.iloc[:,785].values
test_X = train_data_read.iloc[:,1:785].values
train_X=(train_X>=127).astype(int)
test_X=(test_X>=127).astype(int)

train_X_img = train_X.reshape(-1,28,28)

train_X_len=60000
m=784
n=256
epochs=2
k=1
etta=0.1

W = np.random.randn(n,m)/30
b = np.zeros(m)
c = np.zeros(n)

V = train_X

def sigmoid(x):
    return 1/(1+np.exp(-x))

arr = np.zeros((64,784))
counttt = 0
count = 1
errors = []
for epoch in range(epochs):
  total_loss=0
  temp_err=0
  for indx, vd in enumerate(V):
    V_tilda = vd
    if count%1000==0:
      errors.append(total_loss)
      total_loss=0
    for i in range(k):
      H = (sigmoid(np.dot(W,V_tilda)+c)>np.random.rand(W.shape[0])).astype(int)
      V_tilda_int = sigmoid(np.dot(W.T,H)+b)
      V_tilda = ((V_tilda_int)>np.random.rand((W.T).shape[0])).astype(int)
    temp=np.sum((V_tilda_int-vd)**2)
    temp_err=temp/784
    total_loss+=temp_err
    count+=1
    W += etta*(np.outer(sigmoid(np.dot(W,vd)+c),vd) + np.outer(sigmoid(np.dot(W,V_tilda)+c),V_tilda))
    W = (W-np.mean(W))/np.std(W)
    b += etta*(vd-V_tilda)
    c += etta*(sigmoid(np.dot(W,vd)+c) - sigmoid(np.dot(W,V_tilda)+c))
    if (epoch*indx+indx+1)%((epochs*train_X_len)//64) == 0:
      H = (sigmoid(np.dot(W,train_X[15])+c)>np.random.rand(W.shape[0])).astype(int)
      r_img = (sigmoid(np.dot(W.T,H)+b)>np.random.rand((W.T).shape[0])).astype(int)
      arr[counttt] = r_img
      counttt +=1

img_arr = arr.reshape(-1,28,28)

V_test = test_X.T
H = sigmoid((np.dot(W,V_test)+c.reshape(n,1)))

H_embedded = TSNE(n_components=2).fit_transform(H.T)

def plotCode(x, colors):

    palette = np.array(sns.color_palette("hls", 10))

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('tight')
    txts = []
    for i in range(10):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
# plotCode(H_embedded, test_y)

fig, ax = plt.subplots(8, 8, sharex='col', sharey='row', figsize = (9,9))
for i in range(8):
    for j in range(8):
        ax[i, j].imshow(img_arr[i*8+j])
