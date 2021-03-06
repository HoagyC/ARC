# Stolen from the Tiles and Symmetry kernel

import numpy as np
from numpy.lib.stride_tricks import as_strided

import pandas as pd

import os
import json
from pathlib import Path
from tqdm import tqdm
import inspect

import matplotlib.pyplot as plt
from matplotlib import colors
from skimage import measure

import warnings
warnings.filterwarnings("ignore")

def bbox(a):
    try:
        r = np.any(a, axis=1)
        c = np.any(a, axis=0)
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    except:
        return 0,a.shape[0],0,a.shape[1]

def cmask(t_in):
    cmin = 999
    cm = 0
    for c in range(10):
        t = t_in.copy().astype('int8')
        t[t==c],t[t>0],t[t<0]=-1,0,1
        b = bbox(t)
        a = (b[1]-b[0])*(b[3]-b[2])
        s = (t[b[0]:b[1],b[2]:b[3]]).sum()
        if a>2 and a<cmin and s==a:
            cmin=a
            cm=c
    return cm

def mask_rect(a):
    r,c = a.shape
    m = a.copy().astype('uint8')
    for i in range(r-1):
        for j in range(c-1):
            if m[i,j]==m[i+1,j]==m[i,j+1]==m[i+1,j+1]>=1:m[i,j]=2
            if m[i,j]==m[i+1,j]==1 and m[i,j-1]==2:m[i,j]=2
            if m[i,j]==m[i,j+1]==1 and m[i-1,j]==2:m[i,j]=2
            if m[i,j]==1 and m[i-1,j]==m[i,j-1]==2:m[i,j]=2
    m[m==1]=0
    return (m==2)

def crop_min(t_in):
    try:
        b = np.bincount(t_in.flatten(),minlength=10)
        c = int(np.where(b==np.min(b[np.nonzero(b)]))[0])
        coords = np.argwhere(t_in==c)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return t_in[x_min:x_max+1, y_min:y_max+1]
    except:
        return t_in

def call_pred_train(t_in, t_out, pred_func):
    # collects a dict of the various features
    #
    feat = {}
    feat['s_out'] = t_out.shape
    if t_out.shape==t_in.shape:
        diff = in_out_diff(t_in,t_out)
        feat['diff'] = diff
        feat['cm'] = t_in[diff!=0].max()
    else:
        feat['diff'] = (t_in.shape[0]-t_out.shape[0],t_in.shape[1]-t_out.shape[1])
        feat['cm'] = cmask(t_in)
    feat['sym'] = check_symmetric(t_out)
    args = inspect.getargspec(pred_func).args
    if len(args)==1:
        return pred_func(t_in)
    elif len(args)==2:
        t_pred = pred_func(t_in,feat[args[1]])
    elif len(args)==3:
        t_pred = pred_func(t_in,feat[args[1]],feat[args[2]])
    feat['sizeok'] = len(t_out)==len(t_pred)
    t_pred = np.resize(t_pred,t_out.shape)
    acc = (t_pred==t_out).sum()/t_out.size
    return t_pred, feat, acc

def call_pred_test(t_in, pred_func, feat):
    args = inspect.getargspec(pred_func).args
    if len(args)==1:
        return pred_func(t_in)
    elif len(args)==2:
        t_pred = pred_func(t_in,feat[args[1]])
    elif len(args)==3:
        t_pred = pred_func(t_in,feat[args[1]],feat[args[2]])
    return t_pred


# checks the percentage accuracy of the predicted
def check_p(task, pred_func):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(3, n, figsize=(4*n,12), dpi=50)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fnum = 0
    t_acc = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        t_pred, feat, acc = call_pred_train(t_in, t_out, pred_func)
        plot_one(axs[0,fnum],t_in,f'train-{i} input')
        plot_one(axs[1,fnum],t_out,f'train-{i} output')
        plot_one(axs[2,fnum],t_pred,f'train-{i} pred')
        t_acc+=acc
        fnum += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        t_pred = call_pred_test(t_in, pred_func, feat)        
        plot_one(axs[0,fnum],t_in,f'test-{i} input')
        plot_one(axs[1,fnum],t_out,f'test-{i} output')
        plot_one(axs[2,fnum],t_pred,f'test-{i} pred')
        t_pred = np.resize(t_pred,t_out.shape)
        if len(t_out)==1:
            acc = int(t_pred==t_out)
        else:
            acc = (t_pred==t_out).sum()/t_out.size
        t_acc += acc
        fnum += 1
    plt.show()
    return t_acc/fnum

# checks the accuracy without any of the printing
def check_p_bare(task, pred_func):
    n = len(task["train"]) + len(task["test"])
    fnum = 0
    t_acc = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        t_pred, feat, acc = call_pred_train(t_in, t_out, pred_func)
        t_acc+=acc
        fnum += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        t_pred = call_pred_test(t_in, pred_func, feat)
        t_pred = np.resize(t_pred,t_out.shape)
        if len(t_out)==1:
            acc = int(t_pred==t_out)
        else:
            acc = (t_pred==t_out).sum()/t_out.size
        t_acc += acc
        fnum += 1
    return t_acc/fnum

# checks the accuracy without any of the printing
def test_p_bare(task, pred_func):
    n = len(task["train"]) + len(task["test"])
    fnum = 0
    t_acc = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        t_pred, feat, acc = call_pred_train(t_in, t_out, pred_func)
        t_acc+=acc
        fnum += 1
    t_preds = []
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        t_preds.append(call_pred_test(t_in, pred_func, feat))
    return t_preds

def get_tile(img ,mask):
    try:
        m,n = img.shape
        a = img.copy().astype('int8')
        a[mask] = -1
        r=c=0
        for x in range(n):
            if np.count_nonzero(a[0:m,x]<0):continue
            for r in range(2,m):
                if 2*r<m and (a[0:r,x]==a[r:2*r,x]).all():break
            if r<m:break
            else: r=0
        for y in range(m):
            if np.count_nonzero(a[y,0:n]<0):continue
            for c in range(2,n):
                if 2*c<n and (a[y,0:c]==a[y,c:2*c]).all():break
            if c<n:break
            else: c=0
        if c>0:
            for x in range(n-c):
                if np.count_nonzero(a[:,x]<0)==0:
                    a[:,x+c]=a[:,x]
                elif np.count_nonzero(a[:,x+c]<0)==0:
                    a[:,x]=a[:,x+c]
        if r>0:
            for y in range(m-r):
                if np.count_nonzero(a[y,:]<0)==0:
                    a[y+r,:]=a[y,:]
                elif np.count_nonzero(a[y+r,:]<0)==0:
                    a[y,:]=a[y+r,:]
        return a[r:2*r,c:2*c]
    except:
        return a[0:1,0:1]


def patch_image(t_in,s_out,cm=0):
    try:
        t = t_in.copy()
        ty,tx=t.shape
        if cm>0:
            m = mask_rect(t==cm)
        else:
            m = (t==cm)   
        tile = get_tile(t ,m)
        if tile.size>2 and s_out==t.shape:
            rt = np.tile(tile,(1+ty//tile.shape[0],1+tx//tile.shape[1]))[0:ty,0:tx]
            if (rt[~m]==t[~m]).all():
                return rt
        for i in range(6):
            m = (t==cm)
            t -= cm
            if tx==ty:
                a = np.maximum(t,t.T)
                if (a[~m]==t[~m]).all():t=a.copy()
                a = np.maximum(t,np.flip(t).T)
                if (a[~m]==t[~m]).all():t=a.copy()
            a = np.maximum(t,np.flipud(t))
            if (a[~m]==t[~m]).all():t=a.copy()
            a = np.maximum(t,np.fliplr(t))
            if (a[~m]==t[~m]).all():t=a.copy()
            t += cm
            m = (t==cm)
            lms = measure.label(m.astype('uint8'))
            for l in range(1,lms.max()+1):
                lm = np.argwhere(lms==l)
                lm = np.argwhere(lms==l)
                x_min = max(0,lm[:,1].min()-1)
                x_max = min(lm[:,1].max()+2,t.shape[0])
                y_min = max(0,lm[:,0].min()-1)
                y_max = min(lm[:,0].max()+2,t.shape[1])
                gap = t[y_min:y_max,x_min:x_max]
                sy,sx=gap.shape
                if i==1:
                    sy//=2
                    y_max=y_min+sx
                gap = t[y_min:y_max,x_min:x_max]
                sy,sx=gap.shape
                allst = as_strided(t, shape=(ty,tx,sy,sx),strides=2*t.strides)    
                allst = allst.reshape(-1,sy,sx)
                allst = np.array([a for a in allst if np.count_nonzero(a==cm)==0])
                gm = (gap!=cm)
                for a in allst:
                    if sx==sy:
                        fpd = a.T
                        fad = np.flip(a).T
                        if i==1:gm[sy-1,0]=gm[0,sx-1]=False
                        if (fpd[gm]==gap[gm]).all():
                            gm = (gap!=cm)
                            np.putmask(gap,~gm,fpd)
                            t[y_min:y_max,x_min:x_max] = gap
                            break
                        if i==1:gm[0,0]=gm[sy-1,sx-1]=False
                        if (fad[gm]==gap[gm]).all():
                            gm = (gap!=cm)
                            np.putmask(gap,~gm,fad)
                            t[y_min:y_max,x_min:x_max] = gap
                            break 
                    fud = np.flipud(a)
                    flr = np.fliplr(a)
                    if i==1:gm[sy-1,0]=gm[0,sx-1]=gm[0,0]=gm[sy-1,sx-1]=False
                    if (a[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,a)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
                    elif (fud[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,fud)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
                    elif (flr[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,flr)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
        if s_out==t.shape:
            return t
        else:
            m = (t_in==cm)
            return np.resize(t[m],crop_min(m).shape)
    except:
        return np.resize(t_in, s_out)
# from: https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines

cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

def plot_one(ax, input_matrix, title_text):
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title_text)

def plot_task(task):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """    
    num_train = len(task['train'])
    num_test = len(task['test'])
    num_tot = num_train + num_test
    fig, axs = plt.subplots(2, num_tot, figsize=(3*num_tot,3*2))
    for i in range(num_train):
        plot_one(axs[0,i],task['train'][i]['input'],'train input')
        plot_one(axs[1,i],task['train'][i]['output'],'train output')
    i+=1
    for j in range(num_test):
        plot_one(axs[0,i+j],task['test'][j]['input'],'test input')
        plot_one(axs[1,i+j],task['test'][j]['output'],'test output')  
    plt.tight_layout()
    plt.show()
def in_out_diff(t_in, t_out):
    x_in, y_in = t_in.shape
    x_out, y_out = t_out.shape
    diff = np.zeros((max(x_in, x_out), max(y_in, y_out)))
    diff[:x_in, :y_in] -= t_in
    diff[:x_out, :y_out] += t_out
    return diff


def check_symmetric(a):
    try:
        sym = 1
        if np.array_equal(a, a.T):
            sym *= 2 #Check main diagonal symmetric (top left to bottom right)
        if np.array_equal(a, np.flip(a).T):
            sym *= 3 #Check antidiagonal symmetric (top right to bottom left)
        if np.array_equal(a, np.flipud(a)):
            sym *= 5 # Check horizontal symmetric of array
        if np.array_equal(a, np.fliplr(a)):
            sym *= 7 # Check vertical symmetric of array
        return sym
    except:
        return 0
