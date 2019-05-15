#
#   Image Stitching
#   Written by Qhan
#   First Version: 2018.5.20
#   Last Update: 2018.5.22
#

import argparse
import cv2
import os
import os.path as osp
from plot import *
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
    filenames = list(os.listdir(dirname))
    filenames.sort()
    print(filenames)
    for filename in filenames:
        if filename == 'pano.jpg': continue
            
        if file_ext(filename) in ['.png', '.jpg', '.gif', '.JPG']:
            bgr = cv2.imread(dirname + filename)
            h, w, c = bgr.shape
            if h > fix_h:
                new_w = int(w * (fix_h / h))
                bgr = cv2.resize(bgr, (new_w, fix_h), cv2.INTER_LINEAR)
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

def process_images(images, focals, 
                   use_cw=True, save_cw=False, 
                   r_ksize=9, r_sigma=3, r_k=0.04, r_thres=0.5,
                   o_bins=8, o_ksize=9,
                   m_thres=0.8, m_n=4, m_K=1000, 
                   visualize=False, right_to_left=False):
    if use_cw:
        print('\n[Cylinder Warping]')
        print('focal length:\n', focals)
        images = cylinder_warping(images, focals, save_cw)

    height, width, channels = images[0].shape

    print('\n[Feature Detection, Descriptors]')
    Des_left, Des_right = [], []
    for i in range(len(images)):
        print(i)
        im_gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
        R, Ix, Iy, Ix2, Iy2 = calculate_R(im_gray, r_ksize, r_sigma, r_k)
        fpx, fpy = find_local_max_R(R, r_thres)
        ori, ori_1hot, theta, theta_bins, M = get_orientations(Ix, Iy, Ix2, Iy2, o_bins, o_ksize)
        des_left, des_right = get_descriptors(fpx, fpy, ori_1hot, theta, width / 2)
        Des_left += [des_left]
        Des_right += [des_right]

        if visualize:
            plot_features(images[i], R, fpx, fpy, Ix, Iy, i=i)
        
    print('\n[Feature Matching]')
    Dxy = []
    for i in range(len(images)-1):
        print(i)

        if right_to_left:
            matches = find_matches(Des_left[i], Des_right[i+1], m_thres)
            Dxy += [ransac(matches, Des_left[i], Des_right[i+1], m_n, m_K)]

            if visualize:
                plot_matches(images[i], images[i+1], Des_left[i], Des_right[i+1], matches, i)

        else:
            matches = find_matches(Des_right[i], Des_left[i+1], m_thres)
            Dxy += [ransac(matches, Des_right[i], Des_left[i+1], m_n, m_K)]
            
            if visualize:
                plot_matches(images[i], images[i+1], Des_right[i], Des_left[i+1], matches, i)

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
    parser.add_argument('--fix-h', type=int, nargs='?', default=480, help='Resize input image height.')
    parser.add_argument('--no-cw', action='store_false', dest='cw', default=True, help='Not using cylinder warping.')
    parser.add_argument('--save-cw', action='store_true', dest='save_cw', default=False, help='Save cylinder warping.')
    parser.add_argument('--r-ksize', type=int, nargs='?', default=9, help='Gaussian blur kernel size.')
    parser.add_argument('-s', type=float, nargs='?', default=3., help='Gaussian blur sigma.')
    parser.add_argument('-k', type=float, nargs='?', default=0.04, help='R coefficient k.')
    parser.add_argument('-r', type=float, nargs='?', default=0.5, help='R threshold.')
    parser.add_argument('-b', type=int, nargs='?', default=8, help='Orientation bins.')
    parser.add_argument('--o-ksize', type=int, nargs='?', default=9, help='Orientation blur kernel size.')
    parser.add_argument('-m', type=float, nargs='?', default=0.8, help='Matching threshold.')
    parser.add_argument('-n', type=int, nargs='?', default=10, help='RANSAC samples.')
    parser.add_argument('-K', type=int, nargs='?', default=1000, help='RANSAC iterations.')
    args = vars(parser.parse_args())
    
    # Example Usage
    dirname = args['input_dir']
    if dirname[-1] != '/': dirname += '/'
    
    images, focals = read_data(dirname, fix_h=args['fix_h'])
    panorama = process_images(images, focals,
                              use_cw=args['cw'], save_cw=args['save_cw'],
                              r_ksize=args['r_ksize'], r_sigma=args['s'], r_k=args['k'], r_thres=args['r'],
                              o_bins=args['b'], o_ksize=args['o_ksize'],
                              m_thres=args['m'], m_n=args['n'], m_K=args['K'])
    save_image(panorama, dirname[:-1] + '-pano.jpg')