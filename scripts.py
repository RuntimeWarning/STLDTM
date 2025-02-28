import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from core.datasets import get_datasets
from core.model.Autoencoder import get_model


np.random.seed(0)
torch.manual_seed(0)


def main(resolution=128, train_test='train'):
    train_dataset_loader = get_datasets(name='sevir', opt=train_test,
                                        batch_size=32, 
                                        num_workers=4, 
                                        shuffle=False)
    vae = get_model('../vae_weights_sevir.pth', is_train=False)
    vae = nn.DataParallel(vae)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vae.to(device)
    save_dir = f'../PN_Datasets/sevir_{resolution}_features/{train_test}/'
    os.makedirs(save_dir, exist_ok=True)
    idx = 0
    feature_size = 128
    num_pixels = feature_size * feature_size
    rain_amount_thresh = 0.5
    for img in tqdm(train_dataset_loader):
        B, T, H, W = img.shape
        img = img.to(device)

        flags = []
        for im in img:
            if torch.sum(im[0] > 0) >= num_pixels * rain_amount_thresh:
                flags.append(True)
            else:
                flags.append(False)

        img = img.reshape(B*T, -1, H, W)
        img = img * 2. - 1
        img = torch.repeat_interleave(img, 3, dim=1)
        moments = vae(img, fn='encode_moments')
        moments = moments.detach().cpu().numpy()
        _, C, H, W = moments.shape
        moments = moments.reshape(B, T, C, H, W)
        for moment, flag in zip(moments, flags):
            if flag:
                np.save(os.path.join(save_dir, f'{idx}.npy'), moment)
                idx += 1
    print(f'save {idx} files')
main()