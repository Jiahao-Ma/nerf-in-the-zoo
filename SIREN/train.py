from core import SirenImage, PixelDataset, GradientUtils

import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image 
import matplotlib.pyplot as plt
from functools import partial

def imgs2gif(imgs, saveName, duration=None, loop=0, fps=None):
    if fps:
        duration = 1 / fps
    duration *= 1000
    if isinstance(imgs[0], np.ndarray):
        imgs = [Image.fromarray(img) for img in imgs]
    imgs[0].save(saveName, save_all=True, append_images=imgs, duration=duration, loop=loop)

init_functions = {
        "ones": torch.nn.init.ones_,
        "eye": torch.nn.init.eye_,
        "default": partial(torch.nn.init.kaiming_uniform_, a=5 ** (1 / 2)),
        "paper": None,
}

if __name__ == '__main__':
    in_feats = 2
    out_feats = 1
    hidden_layers = 3
    hidden_feats = 256
    model_name = 'siren' # siren or mlp_relu

    target = "intensity"
    init = 'default'

    epochs = 200

    img = np.array(Image.open('dog.png'))
    down_sample_factor = 4
    # resize
    img = img[::down_sample_factor, ::down_sample_factor]
    # 0~255 -> -1 ~ 1
    img = ((img / 255) - 0.5) * 2
    data = PixelDataset(img)

    dataloader = DataLoader(data, batch_size=len(data), shuffle=False)
    if model_name == 'siren':
        model = SirenImage(hidden_feats=hidden_feats, 
                           hidden_layers=hidden_layers, 
                           bias=True, first_omega=30, 
                           hidden_omega=30,
                           custum_function_init=init_functions[init])

    elif model_name == 'relu':
        nets = [nn.Linear(in_features=in_feats, out_features=hidden_feats, bias=True), nn.ReLU()]
        
        for _ in range(hidden_layers):
            nets.append(nn.Linear(in_features=hidden_feats, out_features=hidden_feats, bias=True))
            nets.append(nn.ReLU())
        
        nets.append(nn.Linear(in_features=hidden_feats, out_features=out_feats, bias=True))

        model = nn.Sequential(*nets)

        for m in model.modules():
            if not isinstance(m, nn.Linear):    
                continue
            torch.nn.init.xavier_normal_(m.weight)
    else:
        raise('Unkown model name.')

    if init == 'paper':
        save_path = model_name
    else:
        save_path = model_name + '_' + init

    device = torch.device('cuda:0')
    model.to(device)

    optim = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        losses = []
        model.train()
        for batch in tqdm(dataloader):
            if target == "intensity":
                gt = batch['intensity'].to(device=device)
                gt = gt[:, None]

                coord = batch['coord'].to(device=device)

                pred = model(coord)
                loss = ((gt - pred) ** 2).mean()
            
            losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

        print('Epoch: {}  Loss: {:.4f}'.format(epoch, np.mean(losses)))
        
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            coord = batch['coord'].to(device=device)
            coord.requires_grad_(True)
            pred = model(coord)
            pred_g = (GradientUtils.gradient(pred, coord)
                                   .norm(dim=-1)
                                   .squeeze(dim=-1)
                                   .detach()
                                   .cpu()
                     )
            pred_l = (GradientUtils.laplace(pred, coord)
                                   .squeeze(dim=-1)
                                   .detach()
                                   .cpu()
                     )

            coord_abs = batch['coord_abs'].to(torch.long)
            
            pred = pred.squeeze().cpu().detach()
            pred_img = torch.zeros_like(torch.tensor(data.img), dtype=torch.float32).cpu()
            pred_img_grad_norm = torch.zeros_like(torch.tensor(data.img), dtype=torch.float32).cpu()
            pred_img_laplace = torch.zeros_like(torch.tensor(data.img), dtype=torch.float32).cpu()
            
            pred_img[coord_abs[:, 0], coord_abs[:, 1]] = pred
            pred_img_grad_norm[coord_abs[:, 0], coord_abs[:, 1]] = pred_g
            pred_img_laplace[coord_abs[:, 0], coord_abs[:, 1]] = pred_l

            fig, axes = plt.subplots(3, 2, constrained_layout=True)
            # gt
            axes[0, 0].imshow(data.img, cmap='gray')
            axes[1, 0].imshow(data.grads_norm, cmap='gray')
            axes[2, 0].imshow(data.laps, cmap='gray')

            # pred
            axes[0, 1].imshow(pred_img, cmap='gray')
            axes[1, 1].imshow(pred_img_grad_norm, cmap='gray')
            axes[2, 1].imshow(pred_img_laplace, cmap='gray')

            for rows in axes:
                for ax in rows:
                    ax.set_axis_off()
            
            fig.suptitle(f'Iteration: {epoch}')
            axes[0, 0].set_title("Ground truth")
            axes[0, 1].set_title('Prediction')

            if not os.path.exists('visualization'):
                os.mkdir('visualization')
            if not os.path.exists(f'visualization/{save_path}'):
                os.mkdir(f'visualization/{save_path}')
            plt.savefig(f'visualization/{save_path}/{epoch}.jpg')

    imgs = list()
    for p in os.listdir(f'visualization/{save_path}'):
        img = Image.open(os.path.join(f'visualization/{save_path}', p))
        imgs.append(img)
    imgs2gif(imgs, f'{save_path}.gif', duration=0.033 * 10)

            

