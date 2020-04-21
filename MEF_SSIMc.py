import os
import cv2
import math
import time
import torch
from scipy import signal
import torch.nn.functional as F
import numpy as np
from config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MEF_SSIMc():

    def __init__(self):
        print('init')
        # initial
        self.sigma_g = cfg.sigma_g
        self.sigma_l = cfg.sigma_l
        self.window_size = cfg.window_size
        self.channels = cfg.channels
        self.window = torch.ones([self.window_size, self.window_size, self.channels]) / (self.window_size ** 2 * self.channels)
        self.adaptive_lum = cfg.adaptive_lum
        self.max_iter = cfg.max_iter
        self.cvg_t = cfg.cvg_t
        self.beta_inverse = cfg.beta_inverse
        self.plot_prg = cfg.plot_prg
        self.write_qmap = cfg.write_qmap
        self.write_progress = cfg.write_progress
        self.filename = cfg.filename
        self.algorithm = cfg.algorithm

        self.D1 = cfg.D1
        self.D2 = cfg.D2
        
    def __call__(self, img_seq_color, fused_img):
        print('call')
        img_seq_color = torch.tensor(img_seq_color)
        fused_img = torch.tensor(fused_img)
        self.img_seq_color = torch.unsqueeze(img_seq_color, dim=0)
        self.img_seq_color = self.img_seq_color.permute(0, 3, 1, 2, 4)
        self.fused_img = torch.unsqueeze(fused_img.type(torch.double), dim=0)
        self.fused_img = self.fused_img.permute(0, 3, 1, 2)
        self.window = torch.unsqueeze(self.window, dim=0)
        self.window = self.window.permute(0, 3, 1, 2)

        if cfg.use_cuda:
            self.img_seq_color = self.img_seq_color.to(device)
            self.fused_img = self.fused_img.to(device)
            self.window = self.window.to(device)            

        print(self.img_seq_color.device)

        if self.write_qmap:
            assert(self.filename, None)
            assert(self.algorithm, None)

        return self.MEFO()

    def MEFO(self):
        print('MEFO')
        _, C, H, W = self.fused_img.shape
        self._generate_reference_patches()

        iter_num = 0
        beta_inverse = self.beta_inverse
        lambda_old = 1
        # fused image vector
        x_old = self.fused_img.flatten()
        y_old = x_old
        # the number of over / underflow pixels in each iteration
        flow_num = np.zeros( (self.max_iter, 1) )
        # MEF score in each iter
        S = np.zeros( (self.max_iter, 1) )
        # norm of gradient in each iter
        G = np.zeros( (self.max_iter, 1) )

        Q, grdt, init_gmap = self._cost_fun(x_old)
        # then write them
        print(Q)
        # change them into numpy
        grad_to_save = grdt.reshape(C, H, W).permute(1, 2, 0).cpu().numpy()
        cv2.imwrite('./grdt.png', grad_to_save*255*255)
        #cv2.imwrite('./init_gmap.png', init_gmap.numpy()*255)

        print('begin searching, here are the prints:')

        while iter_num < self.max_iter:
            if iter_num % 250 == 0 and iter_num != 0:
                beta_inverse /= 2

            lambda_new = (1 + math.sqrt(1 + 4 * lambda_old ** 2)) / 2
            gamma = (1 - lambda_old) / lambda_new

            print(beta_inverse * grdt)
            y_new = x_old + beta_inverse * grdt
            x_new = (1 - gamma) * y_new + gamma * y_old
            # deal with overflow and underflow
            flow_num[iter_num, 0] += len(x_new[x_new > 255])
            flow_num[iter_num, 0] += len(x_new[x_new < 0])
            x_new[x_new > 255] = 255
            x_new[x_new < 0] = 0
            #flow numbers may be useful in the future
            #flow_num[iter_num] = 
            Q, grdt, _ = self._cost_fun(x_new)
            S[iter_num, 0] = Q
            G[iter_num, 0] = np.linalg.norm(grdt.cpu().numpy())

            change = np.linalg.norm((x_new - x_old).cpu().numpy())
            self._print_info(iter_num, S[iter_num, 0], G[iter_num, 0], change, flow_num[iter_num, 0])

            #print("iteration: " + str(iter_num) + ',  loss: ' + str(Q) + ', lambda: ' + str(lambda_new))

            if iter_num >= 1:
                if abs(S[iter_num, 0] - S[iter_num-1, 0]) < self.cvg_t:
                    break

            x_old = x_new
            lambda_old = lambda_new
            y_old = y_new

            iter_num += 1

        x = x_new
        opt_image = x.reshape( (C, H, W) ).permute(1, 2, 0).cpu().numpy()

        return opt_image, S

    def _generate_reference_patches(self):
        self.img_seq_color = self.img_seq_color.type(torch.double)
        self.window = self.window.type(torch.double)
        if self.img_seq_color.max() < 1:
            self.img_seq_color *= 255

        _, _, H, W, K = self.img_seq_color.shape
        w_size = self.window.shape[-1]
        self.muY_seq = torch.zeros( (H-w_size+1, W-w_size+1, K) )
        self.muY_sq_seq = torch.zeros( (H-w_size+1, W-w_size+1, K) )
        self.sigmaY_sq_seq = torch.zeros( (H-w_size+1, W-w_size+1, K) )
        self.muY = torch.zeros( (H-w_size+1, W-w_size+1) )
        self.LY = torch.zeros( (H-w_size+1, W-w_size+1, K) )

        if cfg.use_cuda:
            self.muY_seq = self.muY_seq.to(device)
            self.muY_sq_seq = self.muY_sq_seq.to(device)
            self.sigmaY_sq_seq = self.sigmaY_sq_seq.to(device)
            self.muY = self.muY.to(device)
            self.LY = self.LY.to(device)

        if self.adaptive_lum:
            demon_g = 2 * self.sigma_g ** 2
            demon_l = 2 * self.sigma_l ** 2
            
            for k in range(K):
                img = self.img_seq_color[:, :, :, :, k]
                self.muY_seq[:, :, k] = torch.squeeze(F.conv2d(img, self.window))
                self.muY_sq_seq[:, :, k] = torch.mul(self.muY_seq[:, :, k], self.muY_seq[:, :, k])
                self.sigmaY_sq_seq[:, :, k] = torch.squeeze(F.conv2d(torch.mul(img, img), self.window)) - self.muY_sq_seq[:, :, k]
                
                tmp_muY_seq = self.muY_seq[:, :, k] / 255 - 0.5
                tmp_muY_seq = (-torch.mul(tmp_muY_seq, tmp_muY_seq)) / demon_g
                tmp_ly = (torch.mean(img) / 255 - 0.5) ** 2 / demon_l
                self.LY[:, :, k] = torch.exp(tmp_muY_seq - tmp_ly)
                self.muY += torch.mul(torch.squeeze(self.muY_seq[:, :, k]), self.LY[:, :, k])

            self.muY = torch.div(self.muY, torch.sum(self.LY, axis=2))

            tmp_max = torch.max(self.sigmaY_sq_seq, dim=2)
            self.sigmaY_sq = tmp_max[0]
            self.patch_index = tmp_max[1]

        self.muY_sq = torch.mul(self.muY, self.muY)

    def _cost_fun(self, x):
        x = x.type(torch.double)
        self.img_seq_color = self.img_seq_color.type(torch.double)
        self.window = self.window.type(torch.double)

        _, C, M, N, K = self.img_seq_color.shape
        w_size = self.window.shape[-1]

        image = x.reshape( (1, C, M, N) )
        image.requires_grad_(True)

        C1 = (self.D1 * 255) ** 2
        C2 = (self.D2 * 255) ** 2
        Nw = w_size * w_size * C

        H, W = self.muY.shape

        muX = torch.squeeze(F.conv2d(image, self.window))
        muX_sq = muX * muX
        sigmaX_sq = torch.squeeze(F.conv2d(image * image, self.window)) - muX_sq

        A1_patches = 2 * (muX * self.muY) + C1
        B1_patches = muX_sq + self.muY_sq + C1
        B2_patches = sigmaX_sq + self.sigmaY_sq + C2
        B1B2_patches = B1_patches * B2_patches
        B1B2_sq_patches = B1B2_patches * B1B2_patches

        qmap_int = torch.zeros( (H, W, K) )
        sigmaXY = torch.zeros( (H, W, K) )
        qmap = torch.zeros( (H, W) )

        if cfg.use_cuda:
            qmap_int = qmap_int.cuda()
            sigmaXY = sigmaXY.cuda()
            qmap = qmap.cuda()

        if self.adaptive_lum:
            for k in range(K):
                tmp_sigmaXY = torch.squeeze(F.conv2d(image * self.img_seq_color[:, :, :, :, k], self.window))
                sigmaXY[:, :, k] = tmp_sigmaXY - (muX * self.muY_seq[:, :, k])
                tmp_qmap_1 = (2 * (muX * self.muY) + C1) * (2 * sigmaXY[:, :, k] + C2 )
                tmp_qmap_2 = (muX_sq + self.muY_sq  + C1) * (sigmaX_sq + self.sigmaY_sq_seq[:, :, k] + C2)
                qmap_int[:, :, k] = tmp_qmap_1 / tmp_qmap_2

        for k in range(K):
            tmp_patch_index = torch.zeros( (H, W) )
            if cfg.use_cuda:
                tmp_patch_index = tmp_patch_index.cuda()
            tmp_patch_index[self.patch_index == k] = 1
            qmap += (qmap_int[:, :, k] * tmp_patch_index)

        cost = torch.mean(qmap)

        # calculate gradient
        cost_sum = torch.sum(qmap)
        cost_sum.backward()
        grad = image.grad.flatten()

        return cost, grad, qmap

    def _print_info(self, iter_num, loss, grad_norm, change, flow_num):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('iter:        ' + str(iter_num))
        print('loss:        ' + str(loss))
        print('grad norm:   ' + str(grad_norm))
        print('x change:    ' + str(change))
        print('flow number: ' + str(flow_num))

