"""
The optical network architecture
The foward process of optical network

DATE: 2025/11/6
"""
import torch
from torch import nn
from torch import fft
from torch.nn import functional as F

class Modulator(nn.Module):
    """
    Light Spatial Modulator

    init:   PICTURE SIZE size (optional; list, for example, [512, 512])
            NUM (number of input picture)
            IN_REQUIRE_GRAD (if input requires grad)
            INTER (the restoration of input QR code)

    input:  PIXEL SIZE (list, Tensor, for example, [Tensor, Tensor])
            ITEM (item of the picture)

    output: PICTURE MODULATED,
            PIXEL SIZE

    """
    def __init__(self, size, in_require_grad, inter):
        super().__init__()
        self.phase_matrix = nn.Parameter(torch.rand(
            [1, 1, size[0], size[1]], requires_grad=True) * 2 * torch.pi)
        whole_size = [1, 1, 8, 8]
        self.active_area = nn.Parameter(torch.rand(whole_size, requires_grad=False))
        self.noise = nn.Parameter(torch.rand(whole_size, requires_grad=False))
        self.size = size
        self.in_require_grad = in_require_grad
        self.inter = inter

    def forward(self, item, pixel_size):

        # if item[0]<4:
        #    phase_matrix = self.phase_matrix[:, :,
        #                  (item[0] % 3) * self.size[0] // self.inter // 2: \
        #                   (item[0] % 3 + 2) * self.size[0] // self.inter // 2, \
        #                  (item[0] // 3) * self.size[1] // self.inter // 2: \
        #                    (item[0] // 3 + 2) * self.size[1] // self.inter // 2] \
        # else:
        #    phase_matrix = self.phase_matrix[:, :,
        #                  ((item[0]+1) % 3) * self.size[0] // self.inter // 2: \
        #                   ((item[0]+1) % 3 + 2) * self.size[0] // self.inter // 2, \
        #                  ((item[0]+1) // 3) * self.size[1] // self.inter // 2: \
        #                   ((item[0]+1) // 3 + 2) * self.size[1] // self.inter // 2]
        # with torch.no_grad():
        # noise = self.noise.to("cuda")
            #noise1 = F.interpolate(noise, scale_factor=32, mode='nearest')
        active_area = self.active_area.to("cuda")
        active_area1 = F.interpolate(active_area, scale_factor=32, mode='nearest')
        if item[0] == 0:
            phase_matrix = self.phase_matrix[:, :, 0:256, 0:256]
            phase_matrix = phase_matrix.to("cuda")
            y = active_area1 * torch.exp(-1j * phase_matrix)
        # if item[0] == 1:
        #     phase_matrix = self.phase_matrix[:, :, 0:256, 256:512]
        #     phase_matrix = phase_matrix.to("cuda")
        #     y = active_area1 * torch.exp(-1j * phase_matrix)
        # if item[0] == 2:
        #     phase_matrix = self.phase_matrix[:, :, 256:512, 0:256]
        #     phase_matrix = phase_matrix.to("cuda")
        #     y = active_area1 * torch.exp(-1j * phase_matrix)
        # if item[0] == 3:
        #     phase_matrix = self.phase_matrix[:, :, 256:512, 256:512]
        #     phase_matrix = phase_matrix.to("cuda")
        #     y = active_area1 * torch.exp(-1j * phase_matrix)
        if item[0] == 1:
            phase_matrix = self.phase_matrix[:, :, 0:256, 128:384]
            phase_matrix = phase_matrix.to("cuda")
            y = active_area1 * torch.exp(-1j * phase_matrix)
        if item[0] == 2:
            phase_matrix = self.phase_matrix[:, :, 0:256, 256:512]
            phase_matrix = phase_matrix.to("cuda")
            y = active_area1 * torch.exp(-1j * phase_matrix)
        if item[0] == 3:
            phase_matrix = self.phase_matrix[:, :, 128:384, 0:256]
            phase_matrix = phase_matrix.to("cuda")
            y = active_area1 * torch.exp(-1j * phase_matrix)
        if item[0] == 4:
            phase_matrix = self.phase_matrix[:, :, 128:384, 256:512]
            phase_matrix = phase_matrix.to("cuda")
            y = active_area1 * torch.exp(-1j * phase_matrix)
        if item[0] == 5:
            phase_matrix = self.phase_matrix[:, :, 256:512, 0:256]
            phase_matrix = phase_matrix.to("cuda")
            y = active_area1 * torch.exp(-1j * phase_matrix)
        if item[0] == 6:
            phase_matrix = self.phase_matrix[:, :, 256:512, 128:384]
            phase_matrix = phase_matrix.to("cuda")
            y = active_area1 * torch.exp(-1j * phase_matrix)
        if item[0] == 7:
            phase_matrix = self.phase_matrix[:, :, 256:512, 256:512]
            phase_matrix = phase_matrix.to("cuda")
            y = active_area1 * torch.exp(-1j * phase_matrix)
        if item[0] == 8:
            phase_matrix = self.phase_matrix[:, :, 64:320, 64:320]
            phase_matrix = phase_matrix.to("cuda")
            y = active_area1 * torch.exp(-1j * phase_matrix)
        if item[0] == 9:
            phase_matrix = self.phase_matrix[:, :, 64:320, 192:448]
            phase_matrix = phase_matrix.to("cuda")
            y = active_area1 * torch.exp(-1j * phase_matrix)
        if item[0] == 10:
            phase_matrix = self.phase_matrix[:, :, 192:448, 64:320]
            phase_matrix = phase_matrix.to("cuda")
            y = active_area1 * torch.exp(-1j * phase_matrix)
        if item[0] == 11:
            phase_matrix = self.phase_matrix[:, :, 192:448, 192:448]
            phase_matrix = phase_matrix.to("cuda")
            y = active_area1 * torch.exp(-1j * phase_matrix)
        # if item[0] == 8:
        #     phase_matrix = self.phase_matrix[:, :, 0:256, 0:256]
        #     phase_matrix = phase_matrix.to("cuda")
        #     y = noise * torch.exp(-1j * phase_matrix)
        # if item[0] == 9:
        #     phase_matrix = self.phase_matrix[:, :, 0:256, 128:384]
        #     phase_matrix = phase_matrix.to("cuda")
        #     y = noise * torch.exp(-1j * phase_matrix)
        # if item[0] == 10:
        #     phase_matrix = self.phase_matrix[:, :, 0:256, 256:512]
        #     phase_matrix = phase_matrix.to("cuda")
        #     y = noise * torch.exp(-1j * phase_matrix)
        # if item[0] == 11:
        #     phase_matrix = self.phase_matrix[:, :, 128:384, 0:256]
        #     phase_matrix = phase_matrix.to("cuda")
        #     y = noise * torch.exp(-1j * phase_matrix)
        # if item[0] == 12:
        #     phase_matrix = self.phase_matrix[:, :, 128:384, 256:512]
        #     phase_matrix = phase_matrix.to("cuda")
        #     y = noise * torch.exp(-1j * phase_matrix)
        # if item[0] == 13:
        #     phase_matrix = self.phase_matrix[:, :, 256:512, 0:256]
        #     phase_matrix = phase_matrix.to("cuda")
        #     y = noise * torch.exp(-1j * phase_matrix)
        # if item[0] == 14:
        #     phase_matrix = self.phase_matrix[:, :, 256:512, 128:384]
        #     phase_matrix = phase_matrix.to("cuda")
        #     y = noise * torch.exp(-1j * phase_matrix)
        # if item[0] == 15:
        #     phase_matrix = self.phase_matrix[:, :, 256:512, 256:512]
        #     phase_matrix = phase_matrix.to("cuda")
        #     y = noise * torch.exp(-1j * phase_matrix)
        #print("active_area1:", active_area1)
        #print("noise1:", noise1)
        #phase_matrix = phase_matrix.to("cuda")
        #print("phase_matrix:", phase_matrix)
        #y = active_area1 * torch.exp(-1j * phase_matrix)
        return y, pixel_size, active_area1, phase_matrix, self.phase_matrix


class NetWork(nn.Module):
    """
    The network with element-to-element consider

    init: size, m_in_require_grad, inter
    input: plaintext
    output: output, active_area
    """
    def __init__(self, size, m_in_require_grad, inter):
        super().__init__()
        self.m1 = Modulator(size=size, in_require_grad=m_in_require_grad, inter=inter)

    def forward(self, item, pixel_size=torch.ones(2) * 8e-6):
        x1, _, active_area = self.m1(item, pixel_size)
        #X2, pixel_size2 = self.f1(x1, pixel_size1)
        #X3, pixel_size3 = self.l1(X2, pixel_size2)
        #X4, pixel_size4 = self.f2(X3, pixel_size3)
        #y, pixel_size5 = self.o1(X4, pixel_size4)
        y = fft.fftshift(torch.abs(fft.fft2(x1))) / 256 / 256
        return y, active_area

class NetWork2(nn.Module):
    """
    The network considered as a whole

    init: size, m_in_require_grad, inter
    input: plaintext
    output: output, active_area, phase_matrix, phase_matrix_all
    """
    def __init__(self, size, m_in_require_grad, inter):
        super().__init__()
        self.m1 = Modulator(size=size, in_require_grad=m_in_require_grad, inter=inter)
    def forward(self, item, pixel_size=torch.ones(2) * 8e-6):
        x1, _, active_area, phase_matrix, phase_matrix_all = self.m1(item, pixel_size)
        #y = fft.fftshift(torch.abs(fft.fft2(fft.fftshift(x1)))**2) / 1200 / 1200
        y1 = fft.fftshift(x1)
        y2 = fft.fft2(y1)
        y = fft.fftshift(torch.abs(y2)**2)/ 256 / 256
        #y = torch.sigmoid(y)
        return y, active_area, phase_matrix, phase_matrix_all
