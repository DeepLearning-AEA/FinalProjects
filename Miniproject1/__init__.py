from model import *
from others.metrics import psnr

if __name__ == '__main__':
    DATA_FOLDER = "others/data/"
    epochs=10
    model = Model()
    # Fix the seed for repoducibility purposes
    # Load the validation set 
    noisy_imgs, clean_imgs = torch.load(DATA_FOLDER + 'val_data.pkl')
    noisy_imgs_1, noisy_imgs_2 = torch.load(DATA_FOLDER + 'train_data.pkl')

    model.load_pretrained_model()

    # Compute the predictions on the testing set
    pred = model.predict(noisy_imgs)

    print(f"PSNR on test set : {psnr(normalizing_data(pred), normalizing_data(clean_imgs)):.4f}")