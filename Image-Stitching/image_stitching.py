#
#   Image Stitching
#   Written by Qhan
#   First Version: 2018.5.20
#   Last Update: 2018.5.22
#

import argparse
import os
import os.path as osp
import cv2
from utils import *


################
#              #
#   Data I/O   #
#              #
################

def parse_txt(path):
    with open(path, 'r') as f:
        data = [line for line in map(str, f)]
        focals = []
        for i in range(1, len(data)-1):
            if data[i-1] is '\n' and data[i+1] is '\n':
                focals += [float(data[i])]
    return np.array(focals)

def read_data(dirname, fix_h=480):
    
    def file_ext(filename):
        name, ext = osp.splitext(osp.basename(filename))
        return ext
    
    if dirname[-1] != '/': dirname += '/'
    
    rgbs, focals = [], []
    for filename in os.listdir(dirname):
        if filename == 'pano.jpg': continue
            
        if file_ext(filename) in ['.png', '.jpg', '.gif', '.JPG']:
            bgr = cv2.imread(dirname + filename)
            h, w, c = bgr.shape
            if h > fix_h:
                new_w = int(w * (fix_h / h))
                bgr = cv2.resize(bgr, (new_w, fix_h))
            rgbs += [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)]
            
        if file_ext(filename) in ['.txt']:
            focals = parse_txt(dirname + filename)
            
    return np.array(rgbs), focals

def save_image(rgb, path):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    

####################################
#                                  #
#   Image Stitching Process Flow   #
#                                  #
####################################

def process_images(images, focals, use_cw=True, save_cw=False, bins=8, mthres=0.8, rthres=0.5, ransac_n=4):
    if use_cw:
        print('\n[Cylinder Warping]')
        print('focal length:\n', focals)
        images = cylinder_warping(images, focals, save_cw)

    print('\n[Feature Detection, Descriptors]')
    Des = []
    for i in range(len(images)):
        print(i)
        im_gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
        R, Ix, Iy, Ix2, Iy2 = calculate_R(im_gray)
        fpx, fpy = find_local_max_R(R, rthres)
        ori, ori_1hot, theta, theta_bins, M = get_orientations(Ix, Iy, Ix2, Iy2, bins)
        Des += [get_descriptors(fpx, fpy, ori_1hot, theta)]
        
    print('\n[Feature Matching]')
    Dxy = []
    for i in range(len(images)-1):
        print(i)
        matches = find_matches(Des[i], Des[i+1], mthres)
        Dxy += [ransac(matches, Des[i], Des[i+1], n=ransac_n)]

    print('\n[Blending]')
    pano = blend_linear(images, Dxy)
    
    print('\n[Bundle Adjustment]')
    pano_adjust = bundle_adjust(pano)
    
    return pano_adjust


#####################
#                   #
#   Example Usage   #
#                   #
#####################

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Image Stitching')
    parser.add_argument('input_dir', type=str, help='Input directory of images and focal length.')
    parser.add_argument('-m', type=float, nargs='?', default=0.8, help='Matching threshold.')
    parser.add_argument('-r', type=float, nargs='?', default=0.5, help='R threshold.')
    parser.add_argument('-b', type=int, nargs='?', default=8, help='Orientation bins.')
    parser.add_argument('-n', type=int, nargs='?', default=4, help='RANSAC samples.')
    parser.add_argument('--no-cw', action='store_false', dest='cw', default=True, help='Not using cylinder warping.')
    parser.add_argument('--save-cw', action='store_true', dest='save_cw', default=False, help='Save cylinder warping.')
    args = vars(parser.parse_args())
    
    # Example Usage
    dirname = args['input_dir']
    if dirname[-1] != '/': dirname += '/'
    
    images, focals = read_data(dirname)
    panorama = process_images(images,
                              focals,
                              use_cw=args['cw'],
                              save_cw=args['save_cw'],
                              bins=args['b'],
                              mthres=args['m'],
                              rthres=args['r'],
                              ransac_n=args['n'])
    save_image(panorama, dirname[:-1] + '_pano.jpg')