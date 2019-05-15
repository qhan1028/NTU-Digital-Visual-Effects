#
#   Image Stitching: Utilities
#   Written by Qhan
#   First Version: 2018.5.20
#   Last Update: 2018.5.22
#

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pandas as pd
from scipy.spatial.distance import cdist


########################
#                      #
#   Cylinder Warping   #
#                      #
########################

def cylinder_warping(images, focals=[], save_cw=False, savedir='cw/'):
    if save_cw and not osp.exists(savedir): os.mkdir(savedir)
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
        
        if save_cw: cv2.imwrite(savedir + 'warp{}.png'.format(i), res[i])
        print('\rcylinder warping %d' % (i), end=' ')
    
    print()    
    return res.astype(np.uint8)


#################################################
#                                               #
#   Feature Detection: Harris Corner Detector   #
#                                               #
#################################################

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

def find_local_max_R(R, rthres=0.5):
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


###########################
#                         #
#   Feature Descriptors   #
#                         #
###########################

def get_orientations(Ix, Iy, Ix2, Iy2, bins=8, ksize=9):
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
        ori_1hot[b] = cv2.GaussianBlur(ori_1hot[b], (ksize, ksize), 0)

    ori = np.argmax(ori_1hot, axis=0)
    
    return ori, ori_1hot, theta, theta_bins, M

def get_descriptors(fpx, fpy, ori, theta, half_width):
    
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

    descriptors_left = []
    descriptors_right = []
    for y, x in zip(fpy, fpx):
        vector = get_vector(y, x)

        if np.sum(vector) > 0:
            if x <= half_width:
                descriptors_left.append({'y': y, 'x': x, 'vector': vector})

            else:
                descriptors_right.append({'y': y, 'x': x, 'vector': vector})
    
    print('descriptors: (left: %d, right: %d)' % (len(descriptors_left), len(descriptors_right)))
    return descriptors_left, descriptors_right


########################
#                      #
#   Feature Matching   #
#                      #
########################

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


########################################
#                                      #
#   Image Matching: RANSAC Algorithm   #
#                                      #
########################################

def ransac(matches, des1, des2, n=10, K=1000):
    matches = np.array(matches)
    m1, m2 = matches[:, 0], matches[:, 1]
    
    df1 = pd.DataFrame(des1)
    df2 = pd.DataFrame(des2)
    P1 = np.array(df1.loc[m1][['x', 'y']])
    P2 = np.array(df2.loc[m2][['x', 'y']])

    E, Dxy = [], []
    for k in range(K):
        samples = np.random.randint(0, len(P1), n)
        dxy = np.mean(P1[samples] - P2[samples], axis=0).astype(np.int)
        diff_xy = np.abs(P1 - (P2 + dxy))
        err = np.sum( np.sign(np.sum(diff_xy, axis=1)) )
        E += [err]; Dxy += [dxy]

    Ei = np.argsort(E)
    best_dxy = np.round(Dxy[Ei[0]]).astype(int)
    
    print('ransac, best dxy: {}, error: {}'.format(best_dxy, E[Ei[0]]))
    return best_dxy


##############################
#                            #
#   Image Blending: Linear   #
#                            #
##############################

def init_panorama(Dxy_sum, im_shape):
    print('init panorama.')
    Dx, Dy = Dxy_sum[:, 0], Dxy_sum[:, 1]
    dx_max, dx_min = np.max(Dx), np.min(Dx)
    dy_max, dy_min = np.max(Dy), np.min(Dy)
    
    ox = -dx_min if dx_min < 0 else 0
    oy = -dy_min if dy_min < 0 else 0
    
    h, w, c = im_shape
    W = (ox + dx_max + w) if dx_max > 0 else w + ox 
    H = (oy + dy_max + h) if dy_max > 0 else h + oy
    
    pano = np.zeros((H, W, c)).astype(np.float32)
    
    return pano, ox, oy

def find_union_inter(pano, im, x, y, w, h):
    map_pano = np.sign(pano)
    
    map_add = np.zeros(pano.shape)
    map_add[y:y+h, x:x+w][im > 0] = 1
    
    union = np.sign(map_pano + map_add)
    inter = map_add + map_pano - union
    
    return union, inter, map_pano, map_add

def find_range(array):
    sum_x = np.sum(array, axis=0)
    sum_y = np.sum(array, axis=1)
    
    index_x = np.where(sum_x > 0)[0]
    sx = index_x[0]
    ex = index_x[-1] + 1 # slicing
    index_y = np.where(sum_y > 0)[0]
    sy = index_y[0]
    ey = index_y[-1] + 1 # slicing
    
    return sx, ex, sy, ey

def find_amap(inter, signx):
    sx, ex, sy, ey = find_range(inter)
    
    xlen, ylen = ex-sx, ey-sy
    amap = np.zeros((ylen, xlen))
    amap += np.linspace(0, 1, xlen) if signx >= 0 else np.linspace(1, 0, xlen)
    amap = np.stack([amap, amap, amap], axis=2)
    
    return amap, sx, ex, sy, ey

def _blend_linear(pano, im, x, y, signx):
    h, w, c = im.shape
    if np.sum(pano) == 0:
        pano[y:y+h, x:x+w] = im.astype(np.float32)
    else:
        # find intersection, union (OK)
        union, inter, w_pano, w_add = find_union_inter(pano, im, x, y, w, h)
        
        # find intersection amap array with linear weights
        amap, sx, ex, sy, ey = find_amap(inter, signx)
        
        # construct added image
        add = np.zeros(pano.shape).astype(np.float32)
        add[y:y+h, x:x+w] = im.astype(np.float32)
        
        # get weights, blend
        w_pano[sy:ey, sx:ex] *= (1. - amap)
        w_pano[add == 0] = 1
        w_pano[pano == 0] = 0
        w_add[sy:ey, sx:ex] *= amap
        w_add[pano == 0] = 1
        w_add[add == 0] = 0
        
        # blend
        pano = w_pano * pano + w_add * add
        
    return pano

def blend_linear(images, Dxy):
    h, w, c = images[0].shape
    Dxy_sum = [np.zeros(2).astype(int)]
    for dxy in Dxy:
        Dxy_sum.append(Dxy_sum[-1] + dxy)
    
    Dxy_sum = np.array(Dxy_sum)
    pano, ox, oy = init_panorama(Dxy_sum, (h, w, c))
    Dxy = [np.zeros(2)] + Dxy # dx sign
    
    for i, (im, dxy_sum, dxy) in enumerate(zip(images, Dxy_sum, Dxy)):
        dx_sum, dy_sum = dxy_sum
        x, y = ox + dx_sum, oy + dy_sum
        print(i, 'x = %4d, y = %4d' % (x, y))
        pano = _blend_linear(pano, im, x, y, dxy[0])
        
    return pano.astype(np.uint8)


#########################
#                       #
#   Bundle Adjustment   #
#                       #
#########################

def bundle_adjust(pano):
    print('find corners:')
    pano_gray = cv2.cvtColor(pano, cv2.COLOR_RGB2GRAY)
    h, w = pano_gray.shape
    sx, ex, sy, ey = find_range(pano_gray)
    
    lc = pano_gray[:, sx] # left column
    ly = np.where(lc > 0)[0]
    upper_left = [sx, ly[0]]
    bottom_left = [sx, ly[-1]]
    
    ex -= 1
    rc = pano_gray[:, ex] # right column
    ry = np.where(rc > 0)[0]
    upper_right = [ex, ry[0]]
    bottom_right = [ex, ry[-1]]
    
    corner1 = np.float32([upper_left, upper_right, bottom_left, bottom_right])
    corner2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    print('corner1:\n', corner1, '\ncorner2:\n', corner2)
    
    print('warp perspective.')
    M = cv2.getPerspectiveTransform(corner1, corner2)
    pano_adjust = cv2.warpPerspective(pano, M, (w, h))
    
    return pano_adjust
