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
        self.bgr_string = ['blue', 'green', 'red']
        self.weight_type = 2
        self.savedir = savedir
        if not osp.exists(savedir): os.mkdir(savedir)
        print('[Init] savedir:', savedir)
        
    def read_images(self, dirname):
        def is_image(filename):
            name, ext = osp.splitext(osp.basename(filename))
            return ext in ['.jpg', '.png']
        
        self.images = []
        self.images_rgb = []

        for filename in os.listdir(dirname):
            if is_image(filename):
                self.images += [cv2.imread(dirname + '/' + filename)]
                self.images_rgb += [cv2.cvtColor(self.images[-1], cv2.COLOR_BGR2RGB)]

        self.height, self.width, self.channel = self.images[0].shape
        print('[Read] image shape:', self.images[0].shape)
        
        self.P = len(self.images)
        print('[Read] # images:', self.P)

    def display_inputs(self):
        b = np.ceil((np.sqrt(self.P))).astype(int)
        fig, axes = plt.subplots(b, b, figsize=(4 * b, 4 * b))
        
        for p in range(self.P): 
            axes[int(p / b), int(p % b)].imshow(self.images_rgb[p])
            
        fig.savefig(self.savedir + '/input_images.png', bbox_inches='tight', dpi=256)
        
    def read_shutter_times(self, dirname):        
        with open(dirname + '/shutter_times.txt', 'r') as f:
            shutter_times = []
            st_string = []
            
            for line in f.readlines():
                line = line.replace('\n', '')
                st_string += [line]
                
                if '/' in line:
                    a, b = np.float32(line.split('/'))
                    shutter_times += [a/b]
                else:
                    shutter_times += [np.float32(line)]
        
        self.shutter_times = np.array(shutter_times, dtype=np.float32)
        self.log_st = np.log(shutter_times, dtype=np.float32)
        self.st_string = st_string
        
    def sample_points(self, w_points, h_points):
        xp = np.random.randint(0, self.width, w_points)
        yp = np.random.randint(0, self.height, h_points)

        self.N = len(xp) * len(yp) # number of selected pixels
        print('[Sample Points] # samples per image:', self.N)

        xv, yv = np.meshgrid(xp, yp)
        self.Z_bgr = [[self.images[p][yv, xv, c] for p in range(self.P)] for c in range(3)]
        
    def compute_radiance(self, images=None, LnG_bgr=None, log_st=None):
        if images is None: images = self.images
        if LnG_bgr is None: LnG_bgr = self.LnG_bgr
        if log_st is None: log_st = self.log_st
            
        def fmt(x, pos): return '%.3f' % np.exp(x)
        
        h, w = self.height, self.width
        log_radiance_bgr = np.zeros([h, w, self.channel]).astype(np.float32)

        plt.clf()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for c in range(3): # BGR channels

            W_sum = np.zeros([h, w], dtype=np.float32) + 1e-8
            log_radiance_sum = np.zeros([h, w], dtype=np.float32)

            for p in range(self.P): # different shutter times

                print('\r[Radiance] color: ' + self.bgr_string[c] + ', st: ' + self.st_string[p], end='     ')

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
        # self.solve() is inherited from DebevecMethod class
        self.LnG_bgr = [self.solve(Z, self.log_st, self.N, self.P) for Z in self.Z_bgr]
        
    def process(self, indir):        
        self.read_images(indir)
        self.display_inputs()
        self.read_shutter_times(indir)
        self.sample_points(10, 10)
        
        self.solve_bgr()
        self.compute_radiance()
        self.response_curve()
        
        ldr1 = self.photographic_global(self.radiance_bgr) # inherited from ToneMapping class
        cv2.imwrite(self.savedir + "/result_photographic_global.jpg", ldr1)
        
        ldr2 = self.photographic_local(self.radiance_bgr) # inherited from ToneMapping class
        cv2.imwrite(self.savedir + "/result_photographic_local.jpg", ldr2)
        
        ldr3 = self.bilateral_filtering(self.radiance_bgr) # inherited from ToneMapping class
        cv2.imwrite(self.savedir + "/result_bilateral_filtering.jpg", ldr3)
        
        return ldr2


if __name__ == '__main__':
    parser = ArgumentParser('High Dynamic Range Imaging')
    parser.add_argument('input_dir', default='cksmh', nargs='?', 
                        help='input directory of images with different shutter times.')
    args = vars(parser.parse_args())
    
    # Example Usage
    indir = args['input_dir']
    savedir = indir + '_res'
    
    hdr = HDR(savedir=savedir)
    res = hdr.process(indir)
