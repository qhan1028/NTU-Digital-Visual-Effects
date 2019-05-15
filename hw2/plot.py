#
#   Image Stitching: Visualize
#   Written by Qhan
#   First Version: 2018.5.20
#   Last Update: 2018.5.22
#

import cv2
import matplotlib.pyplot as plt
import numpy as np


#########################################
#                                       #
#   Visualize: Harris Corner Detector   #
#                                       #
#########################################

def plot_features(im, R, fpx, fpy, Ix, Iy, arrow_size=1.0, i=0):
    h, w, c = im.shape
    
    feature_points = np.copy(im)
    for x, y in zip(fpx, fpy):
        cv2.circle(feature_points, (x, y), radius=1, color=[255, 0, 0], thickness=1, lineType=1) 
        
    feature_gradients = np.copy(im)
    for x, y in zip(fpx, fpy):
        ex, ey = int(x + Ix[y, x] / arrow_size), int(y + Iy[y, x] / arrow_size)
        ex, ey = np.clip(ex, 0, w), np.clip(ey, 0, h)
        cv2.arrowedLine(feature_gradients, (x, y), (ex, ey), (255, 255, 0), 1)
        
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0, 0].imshow(im); ax[0, 0].set_title('Original')
    ax[0, 1].imshow(np.log(R + 1e-4), cmap='jet'); ax[0, 1].set_title('R')
    ax[1, 0].imshow(feature_points); ax[1, 0].set_title('Feature Points')
    ax[1, 1].imshow(feature_gradients); ax[1, 1].set_title('Gradients')
    plt.savefig('features-%d.png' % i)
    # plt.show()


########################################################
#                                                      #
#   Visualize: Orientations, Local Image Descriptors   #
#                                                      #
########################################################

def plot_orientations(magnitude, theta, theta_bins, orientations, bins=8, i=0):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(magnitude, cmap='gray'); ax[0, 0].set_title('Magnitude')
    ax[0, 1].imshow(theta, cmap='jet'); ax[0, 1].set_title(r'$\theta$')
    ax[1, 0].imshow(theta_bins, cmap='jet'); ax[1, 0].set_title(r'$\theta$ bins: ' + str(bins))
    ax[1, 1].imshow(orientations, cmap='jet'); ax[1, 1].set_title('Local Weighted Orientations')
    plt.savefig('orientations-%d.png' % i)
    # plt.show()
    

###################################
#                                 #
#   Visualize: Feature Matching   #
#                                 #
###################################

def plot_matches(im1, im2, des1, des2, matches, i=0):
    h1, w1, c = im1.shape
    h2, w2, c = im2.shape
    vis = np.zeros([max(h1, h2), w1 + w2, 3], dtype=np.uint8) + 255
    vis[:h1, :w1] = im1
    vis[:h2, w1:] = im2

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(vis)

    for (m1, m2) in matches:
        x1, y1 = des1[m1]['x'], des1[m1]['y']
        x2, y2 = des2[m2]['x'], des2[m2]['y']
        ax.plot([x1, w1 + x2], [y1, y2], marker='o', linewidth=1, markersize=4)
    
    plt.savefig('matches-%d.png' % i)
    #plt.show()