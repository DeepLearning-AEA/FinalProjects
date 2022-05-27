import matplotlib.pyplot as plt

def plot_losses(model):
    """
    Plot the losses of the given model during training
    :param model: The model for which we want to plot the losses
    """
    plt.figure(figsize=(20, 10))
    plt.plot(range(0, len(model.test_losses)), model.test_losses, 'b', lw=4, label="Test loss")
    plt.plot(range(0, len(model.train_losses)), model.train_losses, 'orange', lw=4, label="Train loss")
    plt.xlabel('Epoch', fontsize= 20)
    plt.ylabel('Loss', fontsize = 20)
    plt.title('Evolution of train and test losses during training', fontsize = 20)
    plt.legend(loc=2)
    plt.grid()
    plt.show()
    
def plot_psnr_validation_training(model):
    """
    Plot the performance of the given model during training
    :param model: The model for which we want to plot the performance
    """
    plt.figure(figsize=(20, 10))
    plt.plot(range(0,  len(model.psnrs)), model.psnrs, 'b', lw=4)
    plt.xlabel('Epoch', fontsize= 20)
    plt.ylabel("PSNR", fontsize = 20)
    plt.title('Evolution of PSNR during training', fontsize = 20)
    plt.grid()
    plt.show()
    
