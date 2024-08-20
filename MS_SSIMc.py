import math
import torch
import torch.nn.functional as F
import numpy as np
from config import cfg

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def rgb2gray(rgb):
    weight = torch.tensor([0.2989, 0.5870, 0.1140])
    if cfg.use_cuda:
        weight = weight.to(device)

    rgb = rgb.permute(1, 2, 0)
    ans = rgb * weight
    ans = ans.sum(dim=2)
    return ans


class MS_SSIMc():

    def __init__(self):
        print('init')
        # initial

        self.K = cfg.K
        self.level = cfg.level
        self.eps = cfg.eps
        self.beta_new = cfg.beta_new
        self.cvg_t = cfg.cvg_t
        self.max_iter = cfg.max_iter
        # just level = 3
        self.weight = torch.tensor([0.0448, 0.2856, 0.3001])
        self.weight /= sum(self.weight)
        print(self.weight)
        self.window_size = cfg.window_size_ms
        self.window = torch.tensor(matlab_style_gauss2D((11, 11), 1.5))

    def __call__(self, img_seq_color, fused_img):
        print('call')

        assert self.level == len(self.weight)

        # # img2gray
        # H, W, _, K = img_seq_color.shape
        # img_seq = np.zeros( (H, W, K) )
        # f_i = np.zeros( (H, W) )
        # for k in range(K):
        #     img_seq[:, : ,k] = rgb2gray(img_seq_color[:, :, :, k])
        # f_i[:, :] = rgb2gray(fused_img)

        img_seq_color = torch.tensor(img_seq_color)
        self.img_seq_color = torch.unsqueeze(img_seq_color, dim=0)
        self.img_seq_color = self.img_seq_color.permute(0, 3, 1, 2, 4)

        fused_img = torch.tensor(fused_img)
        self.fused_img = torch.unsqueeze(fused_img.type(torch.double), dim=0)
        self.fused_img = self.fused_img.permute(0, 3, 1, 2)

        self.window = torch.unsqueeze(self.window, dim=0)
        self.window = torch.unsqueeze(self.window, dim=0)

        if cfg.use_cuda:
            self.img_seq_color = self.img_seq_color.to(device)
            self.fused_img = self.fused_img.to(device)
            self.window = self.window.to(device)
            self.weight = self.weight.to(device)

        print(self.img_seq_color.device)

        # try example
        _, C, M, N, K = self.img_seq_color.shape
        gray_image = torch.zeros((M, N))
        img_seq_gray = torch.zeros((M, N, K))
        if cfg.use_cuda:
            gray_image = gray_image.to(device)
            img_seq_gray = img_seq_gray.to(device)

        # img2gray
        for k in range(K):
            img_seq_gray[:, :, k] = rgb2gray(self.img_seq_color[0, :, :, :, k])
        gray_image = rgb2gray(self.fused_img[0, :, :, :])
        img_seq_gray = torch.unsqueeze(img_seq_gray, dim=0)
        img_seq_gray = torch.unsqueeze(img_seq_gray, dim=0)
        gray_image = torch.unsqueeze(gray_image, dim=0)
        gray_image = torch.unsqueeze(gray_image, dim=0)
        # test the result
        # Q, qmap = self.mef_ssim(img_seq_gray, gray_image)

        # return Q, qmap

        return self.MSO()

    def MSO(self):
        print('MS-ssim')

        _, C, H, W = self.fused_img.shape

        iter_num = 0
        beta_new = self.beta_new
        lambda_old = 1
        x_old = self.fused_img.flatten()
        y_old = x_old

        Q, grdt, init_gmap = self.mef_ms_ssim(x_old)

        print(Q)

        # print(grdt)

        # the number of over / underflow pixels in each iteration
        flow_num = np.zeros((self.max_iter, 1))
        # MEF score in each iter
        S = np.zeros((self.max_iter, 1))
        # norm of gradient in each iter
        G = np.zeros((self.max_iter, 1))

        print('begin searching, here are the prints:')

        while iter_num < self.max_iter:
            if iter_num % 250 == 0 and iter_num != 0:
                beta_new /= 2

            lambda_new = (1 + math.sqrt(1 + 4 * lambda_old ** 2)) / 2
            gamma = (1 - lambda_old) / lambda_new

            print(beta_new * grdt)
            y_new = x_old + beta_new * grdt
            x_new = (1 - gamma) * y_new + gamma * y_old
            # deal with overflow and underflow
            flow_num[iter_num, 0] += len(x_new[x_new > 255])
            flow_num[iter_num, 0] += len(x_new[x_new < 0])
            x_new[x_new > 255] = 255
            x_new[x_new < 0] = 0
            # flow numbers may be useful in the future
            # flow_num[iter_num] =
            Q, grdt, _ = self.mef_ms_ssim(x_new)
            S[iter_num, 0] = Q
            G[iter_num, 0] = np.linalg.norm(grdt.cpu().numpy())

            change = np.linalg.norm((x_new - x_old).cpu().numpy())
            self._print_info(iter_num, S[iter_num, 0], G[iter_num, 0], change, flow_num[iter_num, 0])

            # print("iteration: " + str(iter_num) + ',  loss: ' + str(Q) + ', lambda: ' + str(lambda_new))

            if iter_num >= 1:
                if abs(S[iter_num, 0] - S[iter_num-1, 0]) < self.cvg_t:
                    break

            x_old = x_new
            lambda_old = lambda_new
            y_old = y_new

            iter_num += 1

        x = x_new
        opt_image = x.reshape((C, H, W)).permute(1, 2, 0).cpu().numpy()

        return opt_image, S

    def mef_ms_ssim(self, x):
        print('mef_ms_ssim')

        x = x.type(torch.double)
        self.img_seq_color = self.img_seq_color.type(torch.double)
        self.window = self.window.type(torch.double)

        _, C, M, N, K = self.img_seq_color.shape
        # w_size = self.window.shape[-1]

        image = x.reshape((1, C, M, N))
        image.requires_grad_(True)

        # new parameters
        gray_image = torch.zeros((M, N))
        img_seq = torch.zeros((M, N, K))
        if cfg.use_cuda:
            gray_image = gray_image.to(device)
            img_seq = img_seq.to(device)

        # img2gray
        for k in range(K):
            img_seq[:, :, k] = rgb2gray(self.img_seq_color[0, :, :, :, k])
        gray_image = rgb2gray(image[0, :, :, :])
        img_seq = torch.unsqueeze(img_seq, dim=0)
        img_seq = torch.unsqueeze(img_seq, dim=0)
        gray_image = torch.unsqueeze(gray_image, dim=0)
        gray_image = torch.unsqueeze(gray_image, dim=0)

        # print(img_seq.shape)
        # print(gray_image.shape)

        # new data
        down_sample_filter = torch.ones((2, 2)) / 4
        Q = torch.zeros(self.level)
        qmap = []
        if cfg.use_cuda:
            down_sample_filter = down_sample_filter.to(device)
            Q = Q.to(device)

        for layer in range(0, self.level-1):
            Q[layer], qmap_new = self.mef_ssim(img_seq, gray_image)
            qmap.append(qmap_new)
            _, C, H, W, K = img_seq.shape

            downsampling = torch.nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
            new_img_seq = torch.zeros((1, C, math.floor(H/2), math.floor(W/2), K))
            if cfg.use_cuda:
                new_img_seq = new_img_seq.to(device)
            for k in range(K):
                new_img_seq[:, :, :, :, k] = downsampling(img_seq[:, :, :, :, k])

            # img_seq = F.interpolate(img_seq, (1, 1, math.floor(H/2), math.floor(W/2), K), mode='bilinear')
            img_seq = new_img_seq
            gray_image = downsampling(gray_image)

        Q[self.level-1], qmap_new = self.mef_ssim(img_seq, gray_image)
        qmap.append(qmap_new)
        oQ = torch.prod(torch.pow(Q, self.weight))
        oQ.backward()
        grad = image.grad.flatten()
        # print(Q)

        return oQ, grad, qmap

    def mef_ssim(self, img_seq, gray_image):
        img_seq = img_seq.type(torch.double)
        gray_image = gray_image.type(torch.double)

        _, _, M, N, K = img_seq.shape
        w_size = self.window.shape[-1]
        # new data
        s_window = torch.ones((w_size, w_size)) / w_size ** 2
        s_window = torch.unsqueeze(s_window, dim=0)
        s_window = torch.unsqueeze(s_window, dim=0)
        s_window = s_window.type(torch.double)
        bd = math.floor(w_size / 2)
        mu = torch.zeros((M - 2 * bd, N - 2 * bd, K))
        ed = torch.zeros((M - 2 * bd, N - 2 * bd, K))
        denominator = torch.zeros((M - 2 * bd, N - 2 * bd))
        numerator = torch.zeros((M - 2 * bd, N - 2 * bd))
        zero_tensor = torch.zeros((M - 2 * bd, N - 2 * bd)).type(torch.double)
        R = torch.zeros((M - 2 * bd, N - 2 * bd)).type(torch.double)
        w_map = torch.zeros((M - 2 * bd, N - 2 * bd, K))
        sigma1_sq = torch.zeros((M - 2 * bd, N - 2 * bd))
        sigma2_sq = torch.zeros((M - 2 * bd, N - 2 * bd))
        sigma12 = torch.zeros((M - 2 * bd, N - 2 * bd))
        qmap = torch.zeros((M - 2 * bd, N - 2 * bd))
        if cfg.use_cuda:
            s_window = s_window.to(device)
            mu = mu.to(device)
            ed = ed.to(device)
            denominator = denominator.to(device)
            numerator = numerator.to(device)
            zero_tensor = zero_tensor.to(device)
            R = R.to(device)
            w_map = w_map.to(device)
            sigma1_sq = sigma1_sq.to(device)
            sigma2_sq = sigma2_sq.to(device)
            sigma12 = sigma12.to(device)
            qmap = qmap.to(device)

        # use conv2d instead of filter2 !!!
        for k in range(K):
            img = img_seq[:, :, :, :, k]
            mu[:, :, k] = F.conv2d(img, s_window)
            mu_sq = mu[:, :, k] * mu[:, :, k]
            sigma_sq = F.conv2d(img*img, s_window) - mu_sq
            ed[:, :, k] = torch.sqrt(torch.max((w_size ** 2) * sigma_sq, zero_tensor) + 0.001)

        # print(img_seq[:, :, 0, 0, :])
        # 1/0
        # print(mu[0, 0, :])

        for k in range(K):
            img = img_seq * img_seq
            sqrt_tensor = torch.sqrt(F.conv2d(img[:, :, :, :, k], s_window * w_size ** 2))
            sqrt_tensor = torch.squeeze(sqrt_tensor)
            denominator += sqrt_tensor - mu[:, :, k]

        vecs = torch.sum(img_seq, dim=4)
        vec_mean = torch.squeeze(F.conv2d(vecs, s_window))
        sqrt_vec = torch.sqrt(F.conv2d(vecs * vecs, s_window * w_size ** 2))
        vec_norm = torch.squeeze(sqrt_vec)
        numerator = vec_norm - vec_mean

        R = (numerator + self.eps) / (denominator + self.eps)

        print(torch.mean(R))

        # the old slow way
        # for i in range(bd, M-bd):
        #     for j in range(bd, N-bd):
        #         vecs = img_seq[:, :, i-bd:i+bd+1, j-bd:j+bd+1, :].reshape( (w_size**2, K) )
        #         denominator = 0
        #         for k in range(K):
        #             denominator += torch.norm(vecs[:, k]) - mu[i-bd, j-bd, k]
        #         numerator = torch.norm(torch.sum(vecs, dim=1)) - torch.mean(torch.sum(vecs, dim=1))
        #         R[i-bd, j-bd] = (numerator+self.eps) / (denominator + self.eps)

        # print('error sum:')
        # print(torch.sum(torch.abs(R_tmp-R)))
        # print('error max:')
        # print(torch.max(torch.abs(R_tmp-R)))
        # print('pixel num:')
        # print(R.shape)
        # 1/0
        # print(torch.sum(R, dim=0))

        R[R > 1] = 1 - self.eps
        R[R < 0] = 0 + self.eps

        p = torch.tan(math.pi / 2 * R)
        p[p > 10] = 10
        for k in range(K):
            w_map[:, :, k] = torch.pow(ed[:, :, k] / w_size, p) + self.eps
        normalizer = torch.sum(w_map, dim=2)
        for k in range(K):
            w_map[:, :, k] /= normalizer

        tmp_max = torch.max(ed, dim=2)
        maxEd = tmp_max[0]
        # print(maxEd.shape)

        # old slow way 2 !!!
        C = (0.03 * 255) ** 2
        for i in range(bd, M-bd):
            for j in range(bd, N-bd):
                blocks = torch.squeeze(img_seq[:, :, i-bd:i+bd+1, j-bd:j+bd+1, :])
                r_blocks = torch.zeros((w_size, w_size))
                if cfg.use_cuda:
                    r_blocks = r_blocks.to(device)

                for k in range(K):
                    r_blocks += w_map[i-bd, j-bd, k] * (blocks[:, :, k] - mu[i-bd, j-bd, k]) / ed[i-bd, j-bd, k]

                r_blocks_norm = torch.norm(r_blocks)
                if r_blocks_norm > 0:
                    r_blocks *= maxEd[i-bd, j-bd] / r_blocks_norm

                f_blocks = torch.squeeze(gray_image[:, :, i-bd:i+bd+1, j-bd:j+bd+1])

                mu1 = torch.sum(self.window * r_blocks)
                mu2 = torch.sum(self.window * f_blocks)

                sigma1_sq[i-bd, j-bd] = torch.sum(self.window * (r_blocks - mu1) * (r_blocks - mu1))
                sigma2_sq[i-bd, j-bd] = torch.sum(self.window * (f_blocks - mu2) * (f_blocks - mu2))
                sigma12[i-bd, j-bd] = torch.sum(self.window * (r_blocks - mu1) * (f_blocks - mu2))
                qmap[i-bd, j-bd] = (2 * sigma12[i-bd, j-bd] + C) / (sigma1_sq[i-bd, j-bd] + sigma2_sq[i-bd, j-bd] + C)

        conv_mu1 = torch.zeros((M - 2 * bd, N - 2 * bd))
        conv_rBlocks_2 = torch.zeros((M - 2 * bd, N - 2 * bd))
        conv_rfBlock = torch.zeros((M - 2 * bd, N - 2 * bd))
        conv_rBlock_norm = torch.zeros((M - 2 * bd, N - 2 * bd))
        if cfg.use_cuda:
            conv_mu1 = conv_mu1.to(device)
            conv_rBlocks_2 = conv_rBlocks_2.to(device)
            conv_rfBlock = conv_rfBlock.to(device)
            conv_rBlock_norm = conv_rBlock_norm.to(device)

        for k in range(K):
            # mu1
            tmp_conv_blocks = torch.squeeze(F.conv2d(img_seq[:, :, :, :, k], self.window))
            tmp_conv_blocks -= mu[:, :, k]
            conv_mu1 += tmp_conv_blocks * w_map[:, :, k] / ed[:, :, k]
            # rBlock * fBlock
            tmp_conv_rfBlock = torch.squeeze(F.conv2d(img_seq[:, :, :, :, k] * gray_image, self.window))
            tmp_conv_rfBlock -= mu[:, :, k] * torch.squeeze(F.conv2d(gray_image, self.window))
            conv_rfBlock += tmp_conv_rfBlock * w_map[:, :, k] / ed[:, :, k]

        # for the norm
        for k1 in range(K):
            for k2 in range(K):
                # cal the r_block * r_block
                tmp_rBlocks_2 = F.conv2d(img_seq[:, :, :, :, k1] * img_seq[:, :, :, :, k2], self.window)
                tmp_rBlocks_2 -= F.conv2d(img_seq[:, :, :, :, k1], self.window) * mu[:, :, k2]
                tmp_rBlocks_2 -= F.conv2d(img_seq[:, :, :, :, k2], self.window) * mu[:, :, k1]
                tmp_rBlocks_2 += mu[:, :, k1] * mu[:, :, k2]
                conv_rBlocks_2 += torch.squeeze(tmp_rBlocks_2) * w_map[:, :, k1] * w_map[:, :, k2] / (ed[:, :, k1] * ed[:, :, k2])
                # cal the norm of r_block
                tmp_rBlock_norm = F.conv2d(img_seq[:, :, :, :, k1] * img_seq[:, :, :, :, k2], s_window * w_size ** 2)
                tmp_rBlock_norm -= F.conv2d(img_seq[:, :, :, :, k1], s_window * w_size ** 2) * mu[:, :, k2]
                tmp_rBlock_norm -= F.conv2d(img_seq[:, :, :, :, k2], s_window * w_size ** 2) * mu[:, :, k1]
                tmp_rBlock_norm += mu[:, :, k1] * mu[:, :, k2] * w_size ** 2
                conv_rBlock_norm += torch.squeeze(tmp_rBlock_norm) * w_map[:, :, k1] * w_map[:, :, k2] / (ed[:, :, k1] * ed[:, :, k2])

        conv_rBlock_norm = torch.sqrt(conv_rBlock_norm)
        # whole mu1
        conv_mu1 *= maxEd / (conv_rBlock_norm + self.eps)
        # rBlock * rBlock
        conv_rBlocks_2 *= (maxEd / (conv_rBlock_norm + self.eps)) * (maxEd / (conv_rBlock_norm + self.eps))
        sigma1_sq_tmp = conv_rBlocks_2 - conv_mu1 * conv_mu1

        # mu2
        conv_mu2 = torch.squeeze(F.conv2d(gray_image, self.window))
        conv_fBlocks_2 = torch.squeeze(F.conv2d(gray_image * gray_image, self.window))
        # ok!
        sigma2_sq = conv_fBlocks_2 - conv_mu2 * conv_mu2  # f_blocks.^2 - 2*f_blocks*mu2 - mu2^2

        conv_rfBlock *= maxEd / (conv_rBlock_norm + self.eps)
        sigma12 = conv_rfBlock - conv_mu1 * conv_mu2

        # print(sigma1_sq[0, 0])
        # print(sigma2_sq[0, 0])
        # print(sigma12[0, 0])
        # print(conv_rf_tmp[0, 0])

        C = (0.03 * 255) ** 2
        # qmap_tmp = (2 * sigma12 + C) / (sigma1_sq + sigma2_sq + C)
        print('error sum:')
        print(torch.sum(torch.abs(sigma1_sq-sigma1_sq_tmp)))
        print('error max:')
        print(torch.max(torch.abs(sigma1_sq-sigma1_sq_tmp)))
        print('pixel num:')
        print(sigma1_sq.shape)
        1/0
        Q = torch.mean(qmap)

        return Q, qmap

    def _print_info(self, iter_num, loss, grad_norm, change, flow_num):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('iter:        ' + str(iter_num))
        print('loss:        ' + str(loss))
        print('grad norm:   ' + str(grad_norm))
        print('x change:    ' + str(change))
        print('flow number: ' + str(flow_num))
