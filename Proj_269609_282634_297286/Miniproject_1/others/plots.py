import matplotlib.pyplot as plt
import torch
from others.DataLoading import load_and_split_dataset, normalizing_data, denormalize_data


def plot_losses(model):
    """
    Plot the losses of the given model during training
    :param model: The model for which we want to plot the losses
    """
    plt.figure(figsize=(20, 10))
    plt.plot(range(0, len(model.test_losses)), model.test_losses, 'b', lw=4, label="Test loss")
    plt.plot(range(0, len(model.train_losses)), model.train_losses, 'orange', lw=4, label="Train loss")
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Evolution of train and test losses during training', fontsize=20)
    plt.legend(loc=2)
    plt.grid()
    plt.show()


def plot_psnr_validation_training(model):
    """
    Plot the performance of the given model during training
    :param model: The model for which we want to plot the performance
    """
    plt.figure(figsize=(20, 10))
    plt.plot(range(0, len(model.psnrs)), model.psnrs, 'b', lw=4)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel("PSNR", fontsize=20)
    plt.title('Evolution of PSNR during training', fontsize=20)
    plt.grid()
    plt.show()


def show(img_1, img_2, img_3):
    """
    Show a triplet of images
    :param img_1: The noisy image
    :param img_2: The target image
    :param img_3: the denoised image (by the network)
    """
    with torch.no_grad():
        fig = plt.figure(figsize=(10, 7))

        rows = 1
        columns = 3

        fig.add_subplot(rows, columns, 1)
        plt.imshow(torch.clip(denormalize_data(img_1[0]).int(), min=0, max=255).permute(1, 2, 0).cpu())

        fig.add_subplot(rows, columns, 2)
        plt.imshow(torch.clip(denormalize_data(img_2[0]).int(), min=0, max=255).permute(1, 2, 0).cpu())

        fig.add_subplot(rows, columns, 3)
        plt.imshow(torch.clip(denormalize_data(img_3[0]).int(), min=0, max=255).permute(1, 2, 0).cpu())

        plt.close(fig)
