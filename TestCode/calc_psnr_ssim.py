import skimage.io as io
import skimage.color as color
import numpy as np 
import math
import cv2
import os
import glob
def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
########################################
#              calc_metrics            #
########################################
def calc_metrics(img1, img2, crop_border=0, test_Y=True):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        # img1 = rgb2ycbcr(img1)
        # img2 = rgb2ycbcr(img2)
        img1 = color.rgb2gray(img1)
        img2 = color.rgb2gray(img2)

    # if im1_in.ndim == 3:
    #     cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border, :]
    #     cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border, :]
    # elif im1_in.ndim == 2:
    #     cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border]
    #     cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border]
    # else:
    #     raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(img1 * 255, img2 * 255)
    ssim = calc_ssim(img1 * 255, img2 * 255)
    return psnr, ssim

def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def is_image_file(filename):
    return any(
        filename.lower().endswith(extension) for extension in ['png','bmp','jpg'])

def get_filenames(source, image_format):

    # If image_format is a list
    if source is None:
        return []
    # Seamlessy load single file, list of files and files from directories.
    source_fns = []
    if isinstance(source, str):
        if os.path.isdir(source) or source[-1] == '*':
            if isinstance(image_format, list):
                for fmt in image_format:
                    source_fns += get_filenames(source, fmt)
            else:
                source_fns = glob.glob("{}/*.{}".format(source, image_format))
        elif os.path.isfile(source):
            source_fns = [source]
        assert (all([is_image_file(f) for f in source_fns
                     ])), "Given files contain files with unsupported format"
    elif len(source) and isinstance(source[0], str):
        for s in source:
            source_fns.extend(get_filenames(s, image_format=image_format))
    return source_fns

#######################################################################################################
# 这里是针对我的文件夹来单独进行处理的，如果是测试单独的图片的psnr和ssim，单独调用calc_metrics()这个函数就行了
# 计算的结果和matlab计算的结果相同
if __name__ == '__main__':
    # model = "BI" # degradation model
    # scale = 4 # scale
    # dataset = "Manga109" # Set5/Set14/BSD100/Urban100/Manga109
    # method = "RDAN"
    # #######################
    # psnr_add = ssim_add = 0
    # sr_dir = "F:/研究生资料/SR实验筛选/" + model + "/" + method + "/" + dataset # sr文件夹
    # hr_dir = "F:/研究生资料/SR实验筛选/" + model + "/HR/" +  dataset # hr文件夹
    # log_dir = "F:/研究生资料/SR实验筛选/"+ model + "/" + method # 计算后保存的日志文件
    # sr_files = get_filenames(sr_dir, "png")
    # hr_files = get_filenames(hr_dir, "png")
    # psnr_log = open(log_dir + "/psnr_log_" + model + "_" + method +".txt", 'a')
    # psnr_log.write("==================dataset ={}==================\n".format(dataset))
    # for iid, (sr_filename, hr_filename) in enumerate(
    #         zip(sr_files, hr_files)):
    #     # 为了获得文件名
    #     (filepath, tempfilename) = os.path.split(sr_filename)
    #     (filename, extension) = os.path.splitext(tempfilename)
    #     # print(iid)
    #     hr_img = io.imread(hr_filename)
    #     sr_img = io.imread(sr_filename)
    #     psnr_val, ssim_val = calc_metrics(hr_img, sr_img, scale) # 计算每个图片的psnr 和 ssim.
    #     psnr_add += psnr_val
    #     ssim_add += ssim_val
    #     # 计算平均的psnr/ssim
    #     psnr_log.write("{}:\tpsnr = {} ,\tssim = {}\n".format(filename.ljust(30,' '), round(psnr_val,5),round(ssim_val,6)))
    # psnr_log.write("{}:\tpsnr = {} ,\tssim = {} \n".format('AVG'.ljust(30,' '), round(psnr_add/len(sr_files),5), round(ssim_add/len(sr_files),6)))
    hr_img = io.imread('1.png')
    sr_img = io.imread('2.png')
    psnr_val, ssim_val = calc_metrics(hr_img, sr_img) # 计算每个图片的psnr 和 ssim.
    print(psnr_val, ssim_val)