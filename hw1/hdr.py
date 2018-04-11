#
#   High Dynamic Range Imaging
#   Written by Qhan
#   2018.4.11
#

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path as osp
import os
from argparse import ArgumentParser

from utils import ImageAlignment, DebevecMethod, ToneMapping
from timer import Timer


t = Timer()

class HDR(ImageAlignment, DebevecMethod, ToneMapping):
    
    def __init__(self, savedir='res'):
        self.bgr_string = ['b', 'g', 'r']
        self.weight_type = 2
        self.savedir = savedir
        if not osp.exists(savedir): os.mkdir(savedir)
        print('[Init] savedir:', savedir)
        
    def read_images(self, dirname='images'):
        self.images = []
        self.images_rgb = []

        for filename in os.listdir(dirname):
            self.images += [cv2.imread(dirname + '/' + filename)]
            self.images_rgb += [cv2.cvtColor(self.images[-1], cv2.COLOR_BGR2RGB)]

        self.height, self.width, self.channel = self.images[0].shape
        print('[Read] image shape:', self.images[0].shape)
        
        self.P = len(self.images)
        print('[Read] # images =', self.P)

    def display_images(self):
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        for p in range(self.P):
            row = int(p / 4)
            col = int(p % 4)
            axes[row, col].imshow(self.images_rgb[p])
        plt.show()
        
    def read_shutter_times(self, dirname='images'):
        # read txt file
        self.shutter_times = np.array([1/90., 1/60., 1/45., 1/30., 
                                       1/20., 1/15., 1/10.,  1/8., 
                                        1/6.,  1/4.,  1/3.,  1/2., 
                                       1/1.5,    1.,   1.5,    2.], dtype=np.float32)
        self.log_st = np.log(self.shutter_times).astype(np.float32)
        
    def sample_points(self, w_points, h_points):
        xp = np.random.randint(0, self.width, w_points)
        yp = np.random.randint(0, self.height, h_points)

        self.N = len(xp) * len(yp) # number of selected pixels
        print('[Sample Points] # samples per image =', self.N)

        xv, yv = np.meshgrid(xp, yp)
        self.Z_bgr = [[self.images[p][yv, xv, c] for p in range(self.P)] for c in range(3)]
        
    # G_bgr need to be log
    def compute_radiance(self, images=None, LnG_bgr=None, log_st=None):
        if images is None: images = self.images
        if LnG_bgr is None: LnG_bgr = self.LnG_bgr
        if log_st is None: log_st = self.log_st
            
        def fmt(x, pos): return '%.3f' % np.exp(x)
        
        h, w = self.height, self.width
        log_radiance_bgr = np.zeros([h, w, self.channel]).astype(np.float32)

        plt.clf()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for c in range(3):

            W_sum = np.zeros([h, w], dtype=np.float32) + 1e-8
            log_radiance_sum = np.zeros([h, w], dtype=np.float32)

            for p in range(self.P):

                print('\r[Radiance] color=' + self.bgr_string[c] + ', st=%.4f' % (self.shutter_times[p]), end='       ')

                im_1D = images[p][:, :, c].flatten()
                log_radiance = (LnG_bgr[c][im_1D] - log_st[p]).reshape(h, w)

                weights = self.get_weights(im_1D, wtype=self.weight_type, p=p).reshape(h, w)
                w_log_radiance = log_radiance * weights
                log_radiance_sum += w_log_radiance
                W_sum += weights

            weighted_log_radiance = log_radiance_sum / W_sum
            log_radiance_bgr[:, :, c] = weighted_log_radiance

            ax = axes[c]
            im = ax.imshow(weighted_log_radiance, cmap='jet')
            ax.set_title(self.bgr_string[c])
            ax.set_axis_off()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fmt))

        print()
        fig.savefig(self.savedir + '/radiance_debevec.png', bbox_inches='tight', dpi=256)
        
        self.log_radiance_bgr = log_radiance_bgr
        self.radiance_bgr = np.exp(log_radiance_bgr)
        cv2.imwrite(self.savedir + '/radiance_debevec.hdr', self.radiance_bgr)
        
        return self.radiance_bgr
    
    def response_curve(self, LnG_bgr=None):
        if LnG_bgr is None: LnG_bgr = self.LnG_bgr
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for c in range(3):
            ax = axes[c]
            ax.plot(LnG_bgr[c], np.arange(256), c=self.bgr_string[c])
            ax.set_title(self.bgr_string[c])
            ax.set_xlabel('[Response Curve] E: Log Exposure')
            ax.set_ylabel('[Response Curve] Z: Pixel Value')
            ax.grid(linestyle=':', linewidth=1)
            
        fig.savefig(self.savedir + '/response_curve.png', bbox_inches='tight', dpi=256)
        
    def solve_bgr(self):
        self.LnG_bgr = [self.solve(Z, self.log_st, self.N, self.P) for Z in self.Z_bgr]
        
    def process(self, dirname):
        self.read_images(dirname)
        #self.display_images()
        self.read_shutter_times(dirname)
        self.sample_points(10, 10)
        
        self.solve_bgr()
        self.compute_radiance()
        self.response_curve()
        
        ldr1 = self.photographic_global(self.radiance_bgr)
        cv2.imwrite(self.savedir + "/result_photographic_global.jpg", ldr1)
        
        ldr2 = self.photographic_local(self.radiance_bgr)
        cv2.imwrite(self.savedir + "/result_photographic_local.jpg", ldr2)
        
        ldr3 = self.bilateral_filtering(self.radiance_bgr)
        cv2.imwrite(self.savedir + "/result_bilateral_filtering.jpg", ldr3)
        
        return ldr2


if __name__ == '__main__':
    parser = ArgumentParser('High Dynamic Range Imaging')
    parser.add_argument('input_dir', default='cksmh', nargs='?', 
                        help='input directory of images with different shutter times.')
    args = vars(parser.parse_args())
    
    # Example Usage
    hdr = HDR()
    res = hdr.process(args['input_dir'])