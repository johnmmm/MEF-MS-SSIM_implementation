import time
import os
import cv2
import math
import argparse
import pprint
import numpy as np
from MEF_SSIMc import MEF_SSIMc
from MS_SSIMc import MS_SSIMc
from MS_SSIMc_Color import MS_SSIMc_Color
from config import cfg, cfg_from_list

image_path = './images'
sample_name = 'seq'
sample_image_name = 'Tower_Mertens07.png'
# sample_image_name = 'Tower_lsaverage.png'
sample_type = 'png'


def parse_args():
    parser = argparse.ArgumentParser(description='mef_ssimc')
    # args set
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    arg = parser.parse_args()
    return arg


def load_images(reduce=1):
    assert 0 < reduce and reduce <= 1

    sample_path = os.path.join(image_path, sample_name)
    filename_list = os.listdir(sample_path)
    sample_list = []
    for files in filename_list:
        if files.split('.')[-1] == sample_type:
            sample_list.append(files)

    assert len(sample_list) > 0

    # open the first one
    img1 = cv2.imread(os.path.join(sample_path, sample_list[0]))
    rows, columns, channels = img1.shape
    resize_rows = math.floor(rows * reduce)
    resize_columns = math.floor(columns * reduce)
    img_seq = np.zeros((resize_rows, resize_columns, channels, len(sample_list)))

    for sample_num in range(len(sample_list)):
        sample_img = cv2.imread(os.path.join(sample_path, sample_list[sample_num])).astype(np.float64)
        r, c, ch = sample_img.shape
        assert r == rows
        assert c == columns
        assert ch == channels

        if reduce < 1:
            sample_img = cv2.resize(sample_img, (resize_columns, resize_rows),
                                    interpolation=cv2.INTER_CUBIC)

        # if channels are not 3 ...
        img_seq[:, :, :, sample_num] = sample_img

    # load fused image
    image = cv2.imread(os.path.join(image_path, sample_image_name)).astype(np.float64)

    return img_seq, image


if __name__ == "__main__":
    args = parse_args()
    print("Called with args:")
    print(args)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    # something about the writing ...
    print("Using config:")
    pprint.pprint(cfg)

    # read a list json to do the experiment after ...

    img_seq, init_image = load_images()

    time1 = time.time()

    if cfg.model == 'MEF_SSIMc':
        mef_ssimc = MEF_SSIMc()
        output_image, score = mef_ssimc(img_seq, init_image)

    elif cfg.model == 'MS_SSIMc':
        ms_ssimc = MS_SSIMc()
        output_image, score = ms_ssimc(img_seq, init_image)

    elif cfg.model == 'MS_SSIMc_Color':
        ms_ssimc = MS_SSIMc_Color()
        output_image, score = ms_ssimc(img_seq, init_image)

    else:
        raise NotImplementedError(f"[****] '{cfg.model}' no such model.")

    cv2.imwrite('./opt_image3.png', output_image)

    # ms_ssimc = MS_SSIMc()
    # Q, qmap = ms_ssimc(img_seq, init_image)

    time2 = time.time()
    print('time used: ' + str(time2 - time1))
