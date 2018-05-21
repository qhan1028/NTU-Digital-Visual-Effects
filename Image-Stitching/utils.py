#
#   Image Stitching: Utilities
#   Written by Qhan
#   2018.5.20
#

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pandas as pd
from scipy.spatial.distance import cdist


#
#   Cylinder Warping
#

def cylinder_warping(images, savedir='cw/', focals=[]):
    if not osp.exists(savedir): os.mkdir(savedir)
    if len(focals) == 0: focals = np.ones(len(images)) * 6000.
    
    n, h, w = images.shape[:3]
    res = np.zeros([n, h, w, 3])

    # Use inverse warping interpolation
    x_origin = np.floor(w / 2)
    y_origin = np.floor(h / 2)
    x_arange = np.arange(w)
    y_arange = np.arange(h)
    x_prime, y_prime = np.meshgrid(x_arange, y_arange)
    x_prime = x_prime - x_origin
    y_prime = y_prime - y_origin

    for i in range(n):
        img = images[i].flatten()
        s = focals[i]

        x = s * np.tan(x_prime / s)
        y = np.sqrt(x*x + s*s) / s * y_prime
        x += x_origin
        y += y_origin

        idx = np.ones([h, w])
        floor_x = np.floor(x).astype('int32')
        idx[floor_x < 0] = 0; idx[floor_x > w-1] = 0
        floor_x[floor_x < 0] = 0; floor_x[floor_x > w-1] = w-1;

        ceil_x = np.ceil(x).astype('int32')
        idx[ceil_x < 0] = 0; idx[ceil_x > w-1] = 0
        ceil_x[ceil_x < 0] = 0; ceil_x[ceil_x > w-1] = w-1;

        floor_y = np.floor(y).astype('int32')
        idx[floor_y < 0] = 0; idx[floor_y > h-1] = 0
        floor_y[floor_y < 0] = 0; floor_y[floor_y > h-1] = h-1;

        ceil_y = np.ceil(y).astype('int32')
        idx[ceil_y < 0] = 0; idx[ceil_y > h-1] = 0
        ceil_y[ceil_y < 0] = 0; ceil_y[ceil_y > h-1] = h-1;

        xt = ceil_x - x
        yt = ceil_y - y
        for c in range(3):
            left_up = img[c :: 3][floor_y*w + floor_x]
            right_up = img[c :: 3][floor_y*w + ceil_x]
            left_down = img[c :: 3][ceil_y*w + floor_x]
            right_down = img[c :: 3][ceil_y*w + ceil_x]
            t1 = left_up*xt + right_up*(1-xt)
            t2 = left_down*xt + right_down*(1-xt)

            res[i][:,:,c] = t1*yt + t2*(1-yt)

        res[i][idx == 0] = [0, 0, 0]
        
        cv2.imwrite(savedir + 'warp{}.png'.format(i), res[i])
        print('\rcylinder warping %d' % (i), end=' ')
    
    print()    
    return res.astype(np.uint8)


#
#   Harris Corner Detector
#

def calculate_R(gray, ksize=9, S=3, k=0.04):
    K = (ksize, ksize)

    gray_blur = cv2.GaussianBlur(gray, K, S)
    Iy, Ix = np.gradient(gray_blur)

    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    Sx2 = cv2.GaussianBlur(Ix2, K, S)
    Sy2 = cv2.GaussianBlur(Iy2, K, S)
    Sxy = cv2.GaussianBlur(Ixy, K, S)

    detM = (Sx2 * Sy2) - (Sxy ** 2)
    traceM = Sx2 + Sy2

    R = detM - k * (traceM ** 2)
    
    print('R:', np.min(R), np.max(R))
    return R, Ix, Iy, Ix2, Iy2

def find_local_max_R(R, rthres=0.1):
    kernels = []
    for y in range(3):
        for x in range(3):
            if x == 1 and y == 1: continue
            k = np.zeros((3, 3), dtype=np.float32)
            k[1, 1] = 1
            k[y, x] = -1
            kernels.append(k)

    localMax = np.ones(R.shape, dtype=np.uint8)
    localMax[R <= np.max(R) * rthres] = 0

    for k in kernels:
        d = np.sign(cv2.filter2D(R, -1, k))
        d[d < 0] = 0
        localMax &= np.uint8(d)

    print('found corners:', np.sum(localMax))
    feature_points = np.where(localMax > 0)
    
    return feature_points[1], feature_points[0]


#
#   Image Descriptors
#
    
def get_orientations(Ix, Iy, Ix2, Iy2, bins=8):
    M = (Ix2 + Iy2) ** (1/2)

    theta = np.arctan(Iy / (Ix + 1e-8)) * (180 / np.pi)
    theta[Ix < 0] += 180
    theta = (theta + 360) % 360

    bin_size = 360. / bins
    theta_bins = (theta + (bin_size / 2)) // int(bin_size) % bins # divide to 8 bins

    ori_1hot = np.zeros((bins,) + Ix.shape)
    for b in range(bins):
        ori_1hot[b][theta_bins == b] = 1
        ori_1hot[b] *= M
        ori_1hot[b] = cv2.GaussianBlur(ori_1hot[b], (9, 9), 0)

    ori = np.argmax(ori_1hot, axis=0)
    
    return ori, ori_1hot, theta, theta_bins, M


def get_descriptors(fpx, fpy, ori, theta):
    
    bins, h, w = ori.shape

    def get_sub_vector(fy, fx, oy, ox, ori):
        sv = []
        for b in range(bins):
            sv.append(np.sum(ori[b][fy:fy+oy, fx:fx+ox]))
            
        sv_n1 = [x / (np.sum(sv) + 1e-8) for x in sv]
        sv_clip = [x if x < 0.2 else 0.2 for x in sv_n1]
        sv_n2 = [x / (np.sum(sv_clip) + 1e-8) for x in sv_clip]
        
        return sv_n2
    
    def get_vector(y, x):
        # +angle in cv2 is counter-clockwise.
        # +y is down in image coordinates.
        M = cv2.getRotationMatrix2D((12, 12), theta[y, x], 1)
        if y-12 < 0 or x-12 < 0: return 0
        ori_rotated = [cv2.warpAffine(t[y-12:y+12, x-12:x+12], M, (24, 24)) for t in ori]
        
        vector = []
        subpatch_offsets = [4, 8, 12, 16]
        for fy in subpatch_offsets:
            for fx in subpatch_offsets:
                vector += get_sub_vector(fy, fx, 4, 4, ori_rotated)
        return vector

    descriptors = []
    for y, x in zip(fpy, fpx):
        vector = get_vector(y, x)
        if np.sum(vector) > 0:
            descriptors.append({'y': y, 'x': x, 'vector': vector})
    
    print('descriptors:', len(descriptors))    
    return descriptors


#
#   Feature Matching
#

def find_matches(des1, des2, thres=0.8):
    df1 = pd.DataFrame(des1)
    df2 = pd.DataFrame(des2)
    v1 = df1.loc[:]['vector'].tolist()
    v2 = df2.loc[:]['vector'].tolist()

    distances = cdist(v1, v2)
    sorted_index = np.argsort(distances, axis=1)
    
    matches = []
    for i, si in enumerate(sorted_index):
        first = distances[i, si[0]]
        second = distances[i, si[1]]
        if first / second < thres:
            matches.append([i, si[0]])
    
    print('found matches:', len(matches))
    return matches
    

#
#   RANSAC Algorithm
#

def ransac(matches, des1, des2, n=4):
    matches = np.array(matches)
    m1, m2 = matches[:, 0], matches[:, 1]
    
    df1 = pd.DataFrame(des1)
    df2 = pd.DataFrame(des2)
    P1 = np.array(df1.loc[m1][['x', 'y']])
    P2 = np.array(df2.loc[m2][['x', 'y']])

    E, Dxy = [], []
    for k in range(1000):
        samples = np.random.randint(0, len(P1), n)
        dxy = np.mean(P1[samples] - P2[samples], axis=0).astype(np.int)
        diff_xy = np.abs(P1 - (P2 + dxy))
        err = np.sum( np.sign(np.sum(diff_xy, axis=1)) )
        E.append(err); Dxy.append(dxy)

    Ei = np.argsort(E)
    best_dxy = Dxy[Ei[0]]
    
    print('ransac, best dxy: {}, error: {:.4f}'.format(best_dxy, E[Ei[0]]))
    return best_dxy


def get_amap(size1, size2, dxy):
    (h1, w1), (h2, w2), (dx, dy) = size1, size2, dxy
    
    sx = dx if dx > 0 else w2
    sy = dy if dy > 0 else h2
    ex = w1 if dx > 0 else -dx
    ey = h1 if dy > 0 else -dy
    nx, ny = abs(sx - ex), abs(sy - ey)
    
    xlin = np.linspace(1, 0, nx) if dx > 0 else np.linspace(0, 1, nx)
    ylin = np.linspace(1, 0, ny) if dy > 0 else np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(xlin, ylin)
    
    #bw = (xv + yv) / 2
    bw = xv
    bw = np.stack([bw, bw, bw], axis=2)
    
    if sx > ex: sx, ex = ex, sx
    if sy > ey: sy, ey = ey, sy
    
    return (bw, 1 - bw), (sx, sy), (ex, ey)

def init_blend(h, w, c):
    m = np.zeros((h, w, c), dtype=np.float32)
    mw = np.ones((h, w, c), dtype=np.float32)
    return m, np.copy(m), mw, np.copy(mw)

def blend(im1, im2, dxy, dwh):
    h1, w1, c = im1.shape
    h2, w2, c = im2.shape
    (a1, a2), (sx, sy), (ex, ey) = get_amap((h1, w1), (h2, w2), dxy)
    (dx, dy) = dxy
    
    w, h = w1 + abs(dwh[0]), h1 + abs(dwh[1])
    m1, m2, mw1, mw2 = init_blend(h, w, c)
    
    x1, y1 = -min(0, dx), -min(0, dy)
    x2, y2 = max(0, dx), max(0, dy)
    
    #print('im1:', im1.shape)
    #print('im2:', im2.shape)
    #print('dxy:', dxy)
    #print('dwh:', dwh)
    #print('h, w:', h, w)
    #print('x1, y1:', x1, y1)
    #print('x2, y2:', x2, y2)
    
    m1[y1:y1+h1, x1:x1+w1] += im1
    m2[y2:y2+h2, x2:x2+w2] += im2

    mw1[sy:ey, sx:ex] *= a1
    mw1[m2 == 0] = 1
    mw1[m1 == 0] = 0
    
    mw2[sy:ey, sx:ex] *= a2
    mw2[m1 == 0] = 1
    mw2[m2 == 0] = 0    
    
    merged = mw1 * m1 + mw2 * m2
    
    print('blend:', merged.shape)
    return merged.astype(np.uint8)


#
#   RANSAC Algorithn with Rotations
#

def ransac_with_rotation(matches, des1, des2, rgb1, rgb2):
    
    def get_angle(dx, dy):
        angle = np.arctan(dy / (dx + 1e-8)) * (180 / np.pi)
        if dx < 0: angle += 180
        angle = (angle + 360) % 360
        return angle
    
    matches = np.array(matches)
    m1, m2 = matches[:, 0], matches[:, 1]
    
    df1, df2 = pd.DataFrame(des1), pd.DataFrame(des2)
    X1, Y1 = np.array(df1.loc[m1]['x']), np.array(df1.loc[m1]['y'])
    X2, Y2 = np.array(df2.loc[m2]['x']), np.array(df2.loc[m2]['y'])
    
    P1 = np.array([X1, Y1]).T.reshape(1, -1, 2)
    P2 = np.array([X2, Y2]).T.reshape(1, -1, 2)
    
    E, content = [], []

    for i in range(1000):
        sampled_matches = np.random.randint(0, len(matches), 2)

        m1, m2 = sampled_matches
        
        m1_x1, m1_y1, m1_x2, m1_y2 = X1[m1], Y1[m1], X2[m1], Y2[m1]
        m2_x1, m2_y1, m2_x2, m2_y2 = X1[m2], Y1[m2], X2[m2], Y2[m2]
        
        dx1, dy1 = m2_x1 - m1_x1, m2_y1 - m1_y1
        dx2, dy2 = m2_x2 - m1_x2, m2_y2 - m1_y2
        
        a1 = get_angle(dx1, dy1)
        a2 = get_angle(dx2, dy2)
        
        da = a1 - a2 # im2 rotate to im1
        R = cv2.getRotationMatrix2D((m1_x2, m1_y2), -da, 1) # cv2 rotate is counter-clockwise, y down +
        
        P2_r = cv2.transform(P2, R)
        
        dx, dy = m1_x1 - m1_x2, m1_y1 - m1_y2
        P2_r[:, :, 0] += dx
        P2_r[:, :, 1] += dy
        
        e = np.abs(P2_r - P1)
        e = np.sum(np.sign(e[e > 1]))
        E.append(e)
        content.append([dx, dy, -da, m1_x2, m1_y2])
        
    E = np.array(E)
    Ei = np.argsort(E)
    bm = Ei[:5]
    
    print('Best matches:', bm, '\nError:', E[bm])
    return np.mean(np.array(content)[bm], axis=0)


def blend_with_rotation(im1, im2, dxy, da, p2): 
    (dx, dy), (x2, y2) = int(dxy), int(p2)
    
    h1, w1, c = im1.shape
    h2, w2, c = im2.shape    
    (a1, a2), (sx, sy), (ex, ey) = get_amap((h1, w1), (h2, w2), (dx, dy))
    
    h, w = h1 + abs(dy), w1 + abs(dx)
    m1, m2, mw1, mw2 = init_blend(h, w, c)
    
    # rotate im2
    R = cv2.getRotationMatrix2D((x2, y2), da, 1)
    im2 = cv2.warpAffine(im2, R, (w2, h2))
    
    x1, y1 = -min(0, dx), -min(0, dy)
    x2, y2 = max(0, dx), max(0, dy)
    
    m1[y1:y1+h1, x1:x1+w1] += im1
    m2[y2:y2+h2, x2:x2+w2] += im2

    mw1[sy:ey, sx:ex] *= a1
    mw1[m2 == 0] = 1
    mw1[m1 == 0] = 0
    mw2[sy:ey, sx:ex] *= a2
    mw2[m1 == 0] = 1
    mw2[m2 == 0] = 0    
    
    merged = mw1 * m1 + mw2 * m2
    
    return merged.astype(np.uint8)