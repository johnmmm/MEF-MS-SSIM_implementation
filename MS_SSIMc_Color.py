import math
import torch
import torch.nn.functional as F
import numpy as np
from config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class MS_SSIMc_Color():

    def __init__(self):
        print('init')
        # initial

        self.D1 = cfg.D1
        self.D2 = cfg.D2
        self.sigma_g = cfg.sigma_g
        self.sigma_l = cfg.sigma_l
        self.K = cfg.K
        self.level = cfg.level
        self.eps = cfg.eps
        self.beta_new = cfg.beta_new
        self.cvg_t = cfg.cvg_t
        self.max_iter = cfg.max_iter
        self.channels = cfg.channels
        # just level = 3
        self.weight = torch.tensor([0.0448, 0.2856, 0.3001])
        self.weight /= sum(self.weight)
        print(self.weight)
        self.window_size = cfg.window_size_ms
        self.window = torch.tensor(matlab_style_gauss2D((self.window_size, self.window_size), 1.5))
        self.window = self.window.unsqueeze(dim=0).expand(self.channels, self.window_size, self.window_size)
        self.window = (self.window / self.channels).type(torch.double)

    def __call__(self, img_seq_color, fused_img):
        print('call')

        # assert(self.level == len(self.weight))

        img_seq_color = torch.tensor(img_seq_color)
        self.img_seq_color = torch.unsqueeze(img_seq_color, dim=0)
        self.img_seq_color = self.img_seq_color.permute(0, 3, 1, 2, 4)

        fused_img = torch.tensor(fused_img)
        self.fused_img = torch.unsqueeze(fused_img.type(torch.double), dim=0)
        self.fused_img = self.fused_img.permute(0, 3, 1, 2)

        self.window = torch.unsqueeze(self.window, dim=0)

        if cfg.use_cuda:
            self.img_seq_color = self.img_seq_color.to(device)
            self.fused_img = self.fused_img.to(device)
            self.window = self.window.to(device)
            self.weight = self.weight.to(device)

        print(self.img_seq_color.device)

        return self.MSO()

    def MSO(self):
        print('MS-ssim')

        _, C, H, W = self.fused_img.shape

        self._generate_patches()

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

            # print(beta_new * grdt)
            y_new = x_old + beta_new * grdt
            x_new = (1 - gamma) * y_new + gamma * y_old
            # deal with overflow and underflow
            flow_num[iter_num, 0] += len(x_new[x_new > 255])
            flow_num[iter_num, 0] += len(x_new[x_new < 0])
            x_new[x_new > 255] = 255
            x_new[x_new < 0] = 0
            Q, grdt, _ = self.mef_ms_ssim(x_new)
            S[iter_num, 0] = Q
            G[iter_num, 0] = np.linalg.norm(grdt.cpu().numpy())

            change = np.linalg.norm((x_new - x_old).cpu().numpy())
            self._print_info(iter_num,
                             S[iter_num, 0],
                             G[iter_num, 0],
                             change,
                             flow_num[iter_num, 0])

            if iter_num >= 1:
                if abs(S[iter_num, 0] - S[iter_num-1, 0]) < self.cvg_t:
                    break

            if iter_num >= 1:
                if S[iter_num, 0] - S[iter_num-1, 0] < 0:
                    beta_new /= 10

            x_old = x_new
            lambda_old = lambda_new
            y_old = y_new

            iter_num += 1

        x = x_new
        opt_image = x.reshape((C, H, W)).permute(1, 2, 0).cpu().numpy()

        return opt_image, S

    def mef_ms_ssim(self, x):
        # print('mef_ms_ssim')

        x = x.type(torch.double)
        self.img_seq_color = self.img_seq_color.type(torch.double)
        self.window = self.window.type(torch.double)

        _, C, H, W, K = self.img_seq_color.shape
        # w_size = self.window.shape[-1]

        image = x.reshape((1, C, H, W))
        image.requires_grad_(True)

        # new data
        Q = torch.zeros(self.level)
        qmap = []
        if cfg.use_cuda:
            Q = Q.to(device)

        img_seq = self.img_seq_color
        fused_image = image

        if self.level == 1:
            Q[0], qmap_new = self.mef_ssim(img_seq, fused_image, 1)
            oQ = Q[0]

        else:
            for layer in range(0, self.level-1):
                Q[layer], qmap_new = self.mef_ssim(img_seq, fused_image, l)
                qmap.append(qmap_new)
                _, C, H, W, K = img_seq.shape

                downsampling = torch.nn.Upsample(scale_factor=0.5, mode='bicubic', align_corners=False)
                new_img_seq = torch.zeros((1, C, math.floor(H/2), math.floor(W / 2), K))
                if cfg.use_cuda:
                    new_img_seq = new_img_seq.to(device)
                for k in range(K):
                    new_img_seq[:, :, :, :, k] = downsampling(img_seq[:, :, :, :, k])

                img_seq = new_img_seq
                fused_image = downsampling(fused_image)

            Q[self.level-1], qmap_new = self.mef_ssim(img_seq, fused_image, self.level-1)
            qmap.append(qmap_new)
            oQ = torch.prod(torch.pow(Q, self.weight))

        # for i in range(0, len(qmap)):
        #     qmap_to_save = qmap[i].cpu().detach().numpy()
        #     cv2.imwrite('./init_gmap' + str(i) + '.png', qmap_to_save*255)

        oQ.backward()
        grad = image.grad.flatten()
        # print(Q)
        # print(oQ)

        return oQ, grad, qmap

    def mef_ssim(self, img_seq, fused_image, level):
        img_seq = img_seq.type(torch.double)
        fused_image = fused_image.type(torch.double)

        C1 = (self.D1 * 255) ** 2
        C2 = (self.D2 * 255) ** 2

        _, C, H, W, K = img_seq.shape
        w_size = self.window.shape[-1]
        # new data
        s_window_num = C * w_size ** 2
        s_window = torch.ones((C, w_size, w_size)) / s_window_num
        s_window = torch.unsqueeze(s_window, dim=0)
        s_window = s_window.type(torch.double)
        bd = math.floor(w_size / 2)
        sigma2_sq = torch.zeros((H - 2 * bd, W - 2 * bd))
        sigma12 = torch.zeros((H - 2 * bd, W - 2 * bd))
        qmap = torch.zeros((H - 2 * bd, W - 2 * bd))
        if cfg.use_cuda:
            s_window = s_window.to(device)
            sigma12 = sigma12.to(device)
            qmap = qmap.to(device)

        mu = self.mu[level]
        ed = self.ed[level]
        w_map = self.w_map[level]
        conv_one_tmp = self.conv_one_tmp[level]
        conv_two_tmp = self.conv_two_tmp[level]
        tmp_norm = self.tmp_norm[level]
        maxEd = self.maxEd[level]
        sigma1_sq = self.sigma1_sq[level]
        muX = self.muX[level]
        muX_sq = self.muX_sq[level]

        conv_rf_tmp = torch.zeros((H - 2 * bd, W - 2 * bd))
        if cfg.use_cuda:
            conv_rf_tmp = conv_rf_tmp.to(device)

        for k in range(K):
            tmp_conv_tmp = torch.squeeze(F.conv2d(img_seq[:, :, :, :, k] * fused_image, self.window))
            tmp_conv_tmp -= mu[:, :, k] * torch.squeeze(F.conv2d(fused_image, self.window))
            conv_rf_tmp += tmp_conv_tmp * w_map[:, :, k] / ed[:, :, k]

        conv_one_tmp_g = torch.squeeze(F.conv2d(fused_image, self.window))
        conv_two_tmp_g = torch.squeeze(F.conv2d(fused_image * fused_image, self.window))
        sigma2_sq = conv_two_tmp_g - conv_one_tmp_g * conv_one_tmp_g

        conv_rf_tmp *= maxEd / (tmp_norm + self.eps)
        sigma12 = conv_rf_tmp - conv_one_tmp * conv_one_tmp_g

        # print(sigma1_sq[0, 0])
        # print(sigma2_sq[0, 0])
        # print(sigma12[0, 0])
        # print(conv_rf_tmp[0, 0])

        # try to add the \mu
        muY = conv_one_tmp_g
        muY_sq = conv_one_tmp_g * conv_one_tmp_g
        tmp_qmap_1 = (2 * (muX * muY) + C1) * (2 * sigma12 + C2)
        tmp_qmap_2 = (muX_sq + muY_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        qmap = tmp_qmap_1 / tmp_qmap_2
        # print(2 * (muX * muY) + C1)
        Q = torch.mean(qmap)

        return Q, qmap

    def _generate_patches(self):
        img_seq = self.img_seq_color
        img_seq = img_seq.type(torch.double)

        self.mu = []
        self.ed = []
        self.w_map = []
        self.conv_one_tmp = []
        self.conv_two_tmp = []
        self.tmp_norm = []
        self.maxEd = []
        self.sigma1_sq = []
        self.muX = []
        self.muX_sq = []

        _, C, _, _, _ = img_seq.shape
        w_size = self.window.shape[-1]
        # new data
        s_window_num = C * w_size ** 2
        s_window = torch.ones((C, w_size, w_size)) / s_window_num
        s_window = torch.unsqueeze(s_window, dim=0)
        s_window = s_window.type(torch.double)
        if cfg.use_cuda:
            s_window = s_window.to(device)

        for _ in range(self.level):
            _, C, H, W, K = img_seq.shape
            img_seq = img_seq.type(torch.double)
            bd = math.floor(w_size / 2)
            mu = torch.zeros((H - 2 * bd, W - 2 * bd, K))
            ed = torch.zeros((H - 2 * bd, W - 2 * bd, K))
            denominator = torch.zeros((H - 2 * bd, W - 2 * bd))
            numerator = torch.zeros((H - 2 * bd, W - 2 * bd))
            zero_tensor = torch.zeros((H - 2 * bd, W - 2 * bd)).type(torch.double)
            R = torch.zeros((H - 2 * bd, W - 2 * bd)).type(torch.double)
            w_map = torch.zeros((H - 2 * bd, W - 2 * bd, K))
            sigma1_sq = torch.zeros((H - 2 * bd, W - 2 * bd))
            muX = torch.zeros((H - 2 * bd, W - 2 * bd))
            muX_sq = torch.zeros((H - 2 * bd, W - 2 * bd))
            LX = torch.zeros((H - 2 * bd, W - 2 * bd, K))
            if cfg.use_cuda:
                mu = mu.to(device)
                ed = ed.to(device)
                denominator = denominator.to(device)
                numerator = numerator.to(device)
                zero_tensor = zero_tensor.to(device)
                R = R.to(device)
                w_map = w_map.to(device)
                sigma1_sq = sigma1_sq.to(device)
                muX = muX.to(device)
                muX_sq = muX_sq.to(device)
                LX = LX.to(device)

            # use conv2d instead of filter2 !!!
            for k in range(K):
                img = img_seq[:, :, :, :, k]
                mu[:, :, k] = F.conv2d(img, s_window)
                mu_sq = mu[:, :, k] * mu[:, :, k]
                sigma_sq = F.conv2d(img*img, s_window) - mu_sq
                ed[:, :, k] = torch.squeeze(torch.sqrt(torch.max( s_window_num * sigma_sq, zero_tensor) + 0.001))

            # print(mu[0, 0, :])

            for k in range(K):
                img = img_seq * img_seq
                sqrt_tensor = torch.sqrt(F.conv2d(img[:, :, :, :, k], s_window * s_window_num))
                sqrt_tensor = torch.squeeze(sqrt_tensor)
                denominator += sqrt_tensor - mu[:, :, k]

            vecs = torch.sum(img_seq, dim=4)
            vec_mean = torch.squeeze(F.conv2d(vecs, s_window))
            sqrt_vec = torch.sqrt(F.conv2d(vecs * vecs, s_window * s_window_num))
            vec_norm = torch.squeeze(sqrt_vec)
            numerator = vec_norm - vec_mean

            R = (numerator + self.eps) / (denominator + self.eps)
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
            patch_index = tmp_max[1]

            conv_one_tmp = torch.zeros((H - 2 * bd, W - 2 * bd))
            conv_two_tmp = torch.zeros((H - 2 * bd, W - 2 * bd))
            conv_norm_tmp = torch.zeros((H - 2 * bd, W - 2 * bd))
            if cfg.use_cuda:
                conv_one_tmp = conv_one_tmp.to(device)
                conv_two_tmp = conv_two_tmp.to(device)
                conv_norm_tmp = conv_norm_tmp.to(device)

            for k in range(K):
                tmp_conv = torch.squeeze(F.conv2d(img_seq[:, :, :, :, k], self.window))
                tmp_conv -= mu[:, :, k]
                conv_one_tmp += tmp_conv * w_map[:, :, k] / ed[:, :, k]

            for k1 in range(K):
                for k2 in range(K):
                    tmp_conv_2 = F.conv2d(img_seq[:, :, :, :, k1] * img_seq[:, :, :, :, k2], self.window)
                    tmp_conv_2 -= 2 * F.conv2d(img_seq[:, :, :, :, k1], self.window) * mu[:, :, k2]
                    tmp_conv_2 += mu[:, :, k1] * mu[:, :, k2]
                    conv_two_tmp += torch.squeeze(tmp_conv_2) * w_map[:, :, k1] * w_map[:, :, k2] / (ed[:, :, k1] * ed[:, :, k2])

                    tmp_conv_n = F.conv2d(img_seq[:, :, :, :, k1] * img_seq[:, :, :, :, k2], s_window * s_window_num)
                    tmp_conv_n -= 2 * F.conv2d(img_seq[:, :, :, :, k1], s_window * s_window_num) * mu[:, :, k2]
                    tmp_conv_n += mu[:, :, k1] * mu[:, :, k2] * s_window_num
                    this_norm = torch.squeeze(tmp_conv_n) * w_map[:, :, k1] * w_map[:, :, k2] / (ed[:, :, k1] * ed[:, :, k2])
                    conv_norm_tmp += this_norm

            tmp_norm = torch.sqrt(conv_norm_tmp)
            conv_one_tmp *= maxEd / (tmp_norm + self.eps)
            conv_two_tmp *= (maxEd / (tmp_norm + self.eps)) * (maxEd / (tmp_norm + self.eps))
            sigma1_sq = conv_two_tmp - conv_one_tmp * conv_one_tmp

            # calculate the u_k
            demon_g = 2 * self.sigma_g ** 2
            demon_l = 2 * self.sigma_l ** 2

            for k in range(K):
                img = img_seq[:, :, :, :, k]
                muX_k = torch.squeeze(F.conv2d(img, self.window))

                tmp_muX_seq = muX_k / 255 - 0.5
                tmp_muX_seq = (- tmp_muX_seq * tmp_muX_seq) / demon_g
                tmp_lx = (torch.mean(img) / 255 - 0.5) ** 2 / demon_l
                LX[:, :, k] = torch.exp(tmp_muX_seq - tmp_lx)
                muX += torch.squeeze(muX_k) * LX[:, :, k]

            muX = muX / torch.sum(LX, dim=2)
            muX_sq = muX * muX

            self.mu.append(mu)
            self.ed.append(ed)
            self.w_map.append(w_map)
            self.conv_one_tmp.append(conv_one_tmp)
            self.conv_two_tmp.append(conv_two_tmp)
            self.tmp_norm.append(tmp_norm)
            self.maxEd.append(maxEd)
            self.sigma1_sq.append(sigma1_sq)
            self.muX.append(muX)
            self.muX_sq.append(muX_sq)

            downsampling = torch.nn.Upsample(scale_factor=0.5, mode='bicubic', align_corners=False)
            new_img_seq = torch.zeros((1, C, math.floor(H/2), math.floor(W/2), K))
            if cfg.use_cuda:
                new_img_seq = new_img_seq.to(device)
            for k in range(K):
                new_img_seq[:, :, :, :, k] = downsampling(img_seq[:, :, :, :, k])
            img_seq = new_img_seq

    def _print_info(self, iter_num, loss, grad_norm, change, flow_num):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('iter:        ' + str(iter_num))
        print('loss:        ' + str(loss))
        print('grad norm:   ' + str(grad_norm))
        print('x change:    ' + str(change))
        print('flow number: ' + str(flow_num))
