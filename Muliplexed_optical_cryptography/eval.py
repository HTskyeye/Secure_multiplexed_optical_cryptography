"""
Eval the optical network
Simulate the result of the optical DEVICE

DATE: 2025/11/6
"""
import math
import os
import torch
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from nn_module import NetWork2

from mydataset import MyDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def rotate_matrix_45(matrix):
    """
    rotate the input matrix by 45 degree
    :param matrix:
    :return: rotated matrix
    """
    theta=math.radians(45)
    rotation_matrix=np.array([
        [math.cos(theta),-math.sin(theta)],
        [math.sin(theta),math.cos(theta)]
    ])
    h,w=matrix.shape
    new_size=int(math.sqrt(h**2+w**2))+1
    rotated=np.zeros((new_size,new_size),dtype=matrix.dtype)
    center_original=np.array([h//2,w//2])
    centre_new=np.array([new_size//2,new_size//2])
    for i in range(h):
        for j in range(w):
            original_coord=np.array([i,j])-center_original
            rotated_coord=np.dot(rotation_matrix,original_coord)
            new_i,new_j=(rotated_coord+centre_new).astype(int)
            if 0<=new_i<new_size and 0<=new_j<new_size:
                rotated[new_i,new_j]=matrix[i,j]
    return rotated

def psnr1(img1, img2):
    """
    calculate PSNR of two images

    input: image1, image2
    output: PSNR value
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-4:
        return 40
    return 10 * math.log10(1 / mse)


def plt_permute(imag):
    """
    permute image dimensions
    """
    if imag.shape[0] == 3:
        return imag.permute(1, 2, 0).detach().numpy()  # cpu tensor to array
    return imag.squeeze(0).detach().numpy()


def normalize(imag):
    """
    normalize the output images
    """
    return (imag - np.min(imag)) / (np.max(imag) - np.min(imag)) * 255


def pearson_correlation(x, y):
    """
    calculate the PCC of two images
    """
    if len(x) != len(y):
        raise ValueError("not the same length")
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2)) * np.sqrt(np.sum((y - mean_y) ** 2))

    if denominator == 0:
        return 0
    return numerator / denominator


gpu_available = torch.cuda.is_available()
if gpu_available:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

test_data = MyDataset(
    changed_image_path='./dataset/jiaguwen',
    patch_size=[256, 256],
    size=8
)

model = NetWork2(size=[512, 512], m_in_require_grad=False, inter=2)

model.load_state_dict(
    torch.load('./model_parameters/best_model_weight_pnn.pth', \
               map_location=lambda storage, loc: storage))
#print(model.state_dict())

NUMBER = 8
with torch.no_grad():
    for i in range(NUMBER):
        changed_imag, item = test_data[i]

        item = [i]

        out_imag, active_area, phase_matrix, Phase_matrix = model(item)

        active_area = active_area.cpu()

        active_area = active_area.detach().numpy()

        Phase_image = phase_matrix.cpu()

        Phase_image = Phase_image.detach().numpy()

        Phase_Image = Phase_matrix.cpu()

        Phase_Image = Phase_Image.detach().numpy()

        out_imag = out_imag.cpu()

        out_imag = out_imag.squeeze(-4)

        out_imag = plt_permute(out_imag)
        #print(active_area)
        #print(noise)

        changed_imag = plt_permute(changed_imag)
        if i < 8:
            print(pearson_correlation(out_imag, 1-changed_imag))
        else :
            changed_imag1, _ = test_data[i-8]
            changed_imag1 = plt_permute(changed_imag1)
            print(pearson_correlation(out_imag, 1-changed_imag1))



        #Phase_Image[0,0,384:512,256:512]=2
        #Phase_Image[0, 0, 128:256, 128:256] = 2
        #print(Phase_Image)
        x1 = active_area[0,0,:,:] * np.exp(-1j * Phase_Image[0, 0, 192:448, 192:448])
        # active_area2 = np.random.uniform(low=-0.1,high=0.1,size=(8,8))
        # zoom_factor = 32
        # active_area2 = zoom(active_area2, zoom_factor, order=0)  # order=0表示最近邻插值
        #print(active_area2)
        #active_area2 = 0.5*np.ones((256, 256))
        #print(active_area2)
        # x1 = active_area2 * np.exp(-1j * Phase_image[0,0,:,:])
        y1 = np.fft.fftshift(x1)
        y2 = np.fft.fft2(y1)
        y = np.fft.fftshift(np.abs(y2) ** 2) / 256 / 256
        #y = 1 / (1 + np.exp(-y + 2.5))
        print(pearson_correlation(y, 1-changed_imag))

        ssim_value, _ = ssim(changed_imag, out_imag, data_range=1, full=True)
        #ssim_value1, _ = ssim(changed_imag, y, data_range=1, full=True)
        # print("SSIM:", ssim_value)
        # print("SSIM1:", ssim_value1)

        # input = model.state_dict()["m1.phase_matrix"]

        # input = input.cpu()

        # input = input.detach().numpy()




        if i > 9:
            cv2.imwrite(os.path.join('./result/movie_result/in/', 'input' + str(item[0]) + '.jpg'),
                        normalize(active_area[0, 0, :, :]))
            cv2.imwrite(os.path.join('./result/movie_result/out/', \
                                     'output' + str(item[0]) + '.jpg'),normalize(out_imag))
            #cv2.imwrite(os.path.join('./result/movie_result/out/',
            #'offset1' + str(item[0]) + '.jpg'),
            #            normalize(y))
            np.savetxt('phase' + str(item[0]) + '.csv', \
                       Phase_image[0, 0, :, :], delimiter=",", fmt="%.2f")
        else:
            cv2.imwrite(os.path.join('./result/movie_result/in/', \
                                     'Phase_image' + str(0) + str(item[0]) + '.jpg'), \
                                        normalize(Phase_image[0, 0, :, :]))
            cv2.imwrite(os.path.join('./result/movie_result/in/', \
                                     'active_area' + str(0) + str(item[0]) + '.jpg'), \
                                        normalize(active_area[0, 0, :, :]))
            # cv2.imwrite(os.path.join('./result/movie_result/in/', \
            #               'active_area1' + str(0) + str(item[0]) + '.jpg'), \
            #             normalize(noise+active_area[0, 0, :, :]))
            # cv2.imwrite(os.path.join('./result/movie_result/in/', \
            #               'active_area2' + str(0) + str(item[0]) + '.jpg'), \
            #             normalize(active_area2))
            cv2.imwrite(os.path.join('./result/movie_result/out/', \
                                     'output' + str(0) + str(item[0]) + '.jpg'), \
                                        normalize(out_imag[20:236,20:236]))
            cv2.imwrite(os.path.join('./result/movie_result/out/', \
                                     'offset1' + str(0) + str(item[0]) + '.jpg'), \
                                        normalize(y[20:236,20:236]))
            cv2.imwrite(os.path.join('./result/movie_result/changed/', \
                                     'output' + str(0) + str(item[0]) + '.jpg'), \
                                        normalize(1-changed_imag))
            np.savetxt('./result/movie_result/out/phase' + str(item[0]) + \
                       '.csv', Phase_image[0, 0, :, :], \
                       delimiter=",", fmt="%.5f")
            np.savetxt('./result/movie_result/out/phaseoffset.csv', \
                       Phase_Image[0, 0, 20:276, 20:276],delimiter=",",fmt="%.5f")
            # np.savetxt('./result/movie_result/out/amplitude3.csv', \
            #           active_area2,delimiter=",",fmt="%.5f")
            np.savetxt('./result/movie_result/out/decryption'+str(item[0])+'.csv', \
                       out_imag, delimiter=",", fmt="%.5f")


        plt.subplot(NUMBER, 5, i * 5 + 1)
        plt.imshow(active_area[0, 0, :, :], plt.cm.gray)

        # plt.subplot(NUMBER, 5, i * 5 + 2)
        # plt.imshow(noise, plt.cm.gray)

        # plt.subplot(NUMBER, 5, i * 5 + 3)
        # plt.imshow(active_area2, plt.cm.gray)

        plt.subplot(NUMBER, 5, i * 5 + 4)
        plt.imshow(out_imag, plt.cm.gray)

        plt.subplot(NUMBER, 5, i * 5 + 5)
        plt.imshow(y, plt.cm.gray)

    plt.show()
