#
#   Image Stitching: Other Utilities
#   Written by Qhan
#   2018.5.22
#

import cv2
import numpy as np
import pandas as pd


#
#   Blend Directly
#

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


def blend(images, Dxy):
    h, w, c = images[0].shape
    Dxy_sum = [np.zeros(2).astype(int)]
    for dxy in Dxy:
        Dxy_sum.append(Dxy_sum[-1] + dxy)
    
    Dxy_sum = np.array(Dxy_sum)
    pano, ox, oy = init_panorama(Dxy_sum, (h, w, c))
    pano_w = np.copy(pano) + 1e-8
    
    for i, (im, dxy_sum) in enumerate(zip(images, Dxy_sum)):
        dx_sum, dy_sum = dxy_sum
        x, y = ox + dx_sum, oy + dy_sum
        pano[y:y+h, x:x+w] += im
        pano_w[y:y+h, x:x+w][im > 0] += 1
        
    pano /= pano_w
    return pano.astype(np.uint8)


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
