"""
Training process of optical neural network

DATE: 2025/11/6
"""
import os
import time
from datetime import timedelta
import torch
from torch.utils.data import DataLoader
# from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from mydataset import MyDataset
from nn_module import NetWork2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_loop(dataloader, model_pnn, optimizer_pnn, loss_fn, scheduler_pnn, DEVICE='cpu'):
    model_pnn.to(DEVICE)

    loss_fn.to(DEVICE)

    size = len(dataloader.dataset)

    batch_size = dataloader.batch_size

    Loss = 0

    for batch, (changed_image, item) in enumerate(dataloader):
        changed_image = changed_image.to(DEVICE)
        optimizer_pnn.zero_grad()

        out_image1, _, _, _ = model_pnn(item)

        phase_matrix = model_pnn.state_dict()["m1.phase_matrix"]
        input1 = model_pnn.state_dict()["m1.active_area"]

        if (item >= 0 and item < 8):
            loss_pnn = loss_fn(out_image1, 1-changed_image)
        else:
            loss_pnn = loss_fn(out_image1, changed_image)

        #    loss_pnn = 0.0005 * torch.sum(torch.pow \
        #    (torch.diff(out_image1, dim=2), 2)) + 0.0005 * torch.sum( \
        #    torch.pow(torch.diff(out_image1, dim=3), 2))

        Loss = loss_pnn + Loss

        loss_pnn.backward()
        optimizer_pnn.step()

        phase_matrix.data.clamp_(0, 2 * torch.pi)
        input1.data.clamp_(0, 1)
        #print(noise)

        if (batch + 1) % 1 == 0:
            loss_pnn, current = loss_pnn.item(), (batch + 1) * batch_size

            (f"loss_pnn: {loss_pnn:>7f} [{current:>5d}/{size:>5d}]")

        if size - (batch + 1) * batch_size < batch_size:
            break

    print(f"learning rate of pnn: {optimizer_pnn.param_groups[0]['lr']}")

    scheduler_pnn.step()

    return Loss / (size / batch_size)


if __name__ == '__main__':
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    training_data = MyDataset(
        changed_image_path='./dataset/jiaguwen/',
        patch_size=[256, 256],
        size=8
    )

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False, num_workers=0)

    model_pnn = NetWork2(size=[512, 512], m_in_require_grad=False, inter=2)

    #model_pnn.load_state_dict(torch.load('./model_parameters/best_model_weight_pnn.pth'))

    learning_rate = 0.001

    epochs = 5000

    loss_fn = torch.nn.MSELoss(reduction='mean')

    optimizer_pnn = torch.optim.Adam(model_pnn.parameters(), lr=0.001)

    scheduler_pnn = lr_scheduler.StepLR(optimizer_pnn, 5000, gamma=0.1, last_epoch=-1)

    # train
    train_loss = []
    step = []
    Loss = 1

    # plt.ion()
    start_time = time.time()

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        loss = train_loop(train_dataloader, model_pnn, optimizer_pnn, \
                          loss_fn, scheduler_pnn, DEVICE)
        print(f'average loss: {loss}')
        if loss < Loss:
            torch.save(model_pnn.state_dict(), './model_parameters/best_model_weight_pnn.pth')
            Loss = loss
        step += [t]
        train_loss += [loss.cpu().detach().numpy()]

        # plt.clf()
        # plt.plot(step, train_loss)
        # plt.pause(0.1)
        # plt.ioff()
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Done! Elapsed time of training = {str(timedelta(seconds=elapsed))}")
