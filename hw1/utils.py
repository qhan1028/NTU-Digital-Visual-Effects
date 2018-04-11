#
#   HDR Utilities
#   Written by Qhan
#   2018.4.11
#

import numpy as np
import cv2


gaussian = lambda x, mu, s: 1 / (s * (2 * np.pi) ** (1/2)) * np.exp(-(x - mu) ** 2 / (2 * s ** 2))
    
class Weights():
    
    def get_weights(self, Z, wtype=0, p=0, mean=128, sigma=128):
        if wtype == 1:
            zmax = np.max(Z).astype(np.float32)
            zmin = np.min(Z).astype(np.float32)
            zmid = np.ceil(np.mean(Z)).astype(int)
            weights = np.arange(256, dtype=np.float32)
            weights[:zmid] = weights[:zmid] - zmin
            weights[zmid:] = zmax - weights[zmid:]
            return weights[Z] / zmid.astype(np.float32)

        elif wtype == 2:
            w = np.arange(256)
            return gaussian(w, mean, sigma)[Z]

        else:
            return np.ones(Z.shape, dtype=np.float32)

        
#
#   Image Alignment (Pyramid Method)
#

class ImageAlignment():
    
    def __init__(self, threshold=4):
        self.thres = threshold

    def translation_matrix(self, dx, dy):
        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])
        return M

    def find_shift(self, src, tar, x, y):
        h, w = tar.shape[:2]
        min_error = np.inf
        best_dx, best_dy = 0, 0

        median = np.median(src)
        ignore_pixels = np.ones(src.shape)
        ignore_pixels[np.where(np.abs(src - median) <= self.thres)] = 0

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                tmp_src = cv2.warpAffine(src, self.translation_matrix(x + dx, y + dy), (w, h))

                error = np.abs(np.sign(tmp_src - tar))
                error = np.sum(error * ignore_pixels)

                if error < min_error:
                    min_error = error
                    best_dx, best_dy = dx, dy

        return x + best_dx, y + best_dy

    def align(self, src, tar, depth=6):
        if depth == 0:
            dx, dy = self.find_shift(src, tar, 0, 0)

        else:
            h, w = src.shape[:2]
            half_src = cv2.resize(src, (w//2, h//2))
            half_tar = cv2.resize(tar, (w//2, h//2))
            prev_dx, prev_dy = self.align(half_src, half_tar, depth-1)
            dx, dy = self.find_shift(src, tar, prev_dx * 2, prev_dy * 2)

        return dx, dy

    
#
#   Paul Debevec's Method for HDR Imaging
#

class DebevecMethod(Weights):
    
    def __init__(self, wtype=2):
        self.weight_type = wtype
    
    def constructA(self, cols, rows, W, Ln_st, N, P, constraint, lamda):
        A = np.zeros([N * P + 255, 256 + N], dtype=np.float32)

        A[cols, rows] = W

        for p in range(P):
            A[p * N : p * N + N, 256:] = -np.identity(N) * W[p * N : p * N + N]

        for i in range(254):
            A[N * P + i, i : i + 3] = np.array([1, -2, 1]) * lamda

        A[-1, constraint] = 1

        return A

    def constructB(self, cols, rows, W, Ln_st, N, P):
        B = np.zeros(N * P + 255, dtype=np.float32)

        for p in range(P):
            B[p * N : p * N + N] = Ln_st[p]
        B[cols] *= W

        return B

    def construct_matrix(self, samples, Ln_shutter_times, num_samples, num_images, constraint=128, lamda=10):
        Z = samples
        Ln_st = Ln_shutter_times
        N = num_samples
        P = num_images

        cols = np.arange(N * P)
        rows = np.array(Z).flatten()

        W = self.get_weights(rows, wtype=self.weight_type)
        A = self.constructA(cols, rows, W, Ln_st, N, P, constraint, lamda)
        B = self.constructB(cols, rows, W, Ln_st, N, P)

        return A, B
    
    def solve(self, Z, Ln_st, N, P):
        A, B = self.construct_matrix(Z, Ln_st, N, P, constraint=128)
        A_inv = np.linalg.pinv(A)
        lnG = np.dot(A_inv, B)[:256]
        print('[Debevec] A inverse solved:', A_inv.shape)
        
        return lnG

    
#
#   Tone Mapping: Photographic Global / Local, Bilateral
#

class ToneMapping():
    
    def photographic_global(self, hdr, d=1e-6, a=0.7):
        
        ldr = np.zeros_like(hdr, dtype=np.float32)
        
        for c in range(3):
            print('\r[Photographic Global] color:', c, end='')
            Lw = hdr[:, :, c]
            Lw_ave = np.exp(np.mean(np.log(d + Lw)))
            Lm = (a / Lw_ave) * Lw
            Lm_max = np.max(Lm) # Lm_white
            Ld = Lm * (1 + (Lm / Lm_max ** 2)) / (1 + Lm)
            ldr[:, :, c] = np.array(Ld * 255).astype(np.uint8)
        print()
        
        return ldr
        
    def gaussian_blurs(self, im, smax=25, a=0.7, fi=8.0, epsilon=0.01):
        cols, rows = im.shape
        blur_prev = im
        num_s = int((smax+1)/2)
        
        blur_list = np.zeros(im.shape + (num_s,))
        Vs_list = np.zeros(im.shape + (num_s,))
        
        for i, s in enumerate(range(1, smax+1, 2)):
            print('\r[Photographic Local] filter:', s, end=', ')
            blur = cv2.GaussianBlur(im, (s, s), 0)
            Vs = np.abs((blur - blur_prev) / (2 ** fi * a / s ** 2 + blur_prev))
            blur_list[:, :, i] = blur
            Vs_list[:, :, i] = Vs
        
        # 2D index
        print('find index...', end='')
        smax = np.argmax(Vs_list > epsilon, axis=2)
        smax[np.where(smax == 0)] = num_s
        smax -= 1
        
        # select blur size for each pixel
        print(', apply index...')
        I, J = np.ogrid[:cols, :rows]
        blur_smax = blur_list[I, J, smax]
        
        return blur_smax
        
    def photographic_local(self, hdr, d=1e-6, a=0.7):
        ldr = np.zeros_like(hdr, dtype=np.float32)
        
        for c in range(3):
            print('[Photographic Local] color:', c)
            Lw = hdr[:, :, c]
            Lw_ave = np.exp(np.mean(np.log(d + Lw)))
            Lm = (a / Lw_ave) * Lw
            Ls = self.gaussian_blurs(Lm)
            Ld = Lm / (1 + Ls)
            ldr[:, :, c] = np.array(Ld * 255).astype(np.uint8)
        
        return ldr
    
    def bilateral_filtering(self, hdr):
        ldr = np.zeros_like(hdr, dtype=np.float32)
        
        for c in range(3):
            print('[Bilateral Filtering] color:', c)
            
            Lw = hdr[:, :, c]
            log_Lw = np.log(Lw)
            log_base = cv2.bilateralFilter(log_Lw, 9, 5, 5)
            log_detail = log_Lw - log_base
            
            cf = (np.log(5)) / (log_base.max() - log_base.min()) # compression factor
            log_Ld = log_base * cf + log_detail
            Ld = np.exp(log_Ld) / np.exp(cf * log_base.max())
            
            ldr[:, :, c] = Ld * 255
        
        return ldr
