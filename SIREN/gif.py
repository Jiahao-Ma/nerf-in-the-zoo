import os
from PIL import Image
import numpy as np
def imgs2gif(imgs, saveName, duration=None, loop=0, fps=None):
    if fps:
        duration = 1 / fps
    duration *= 1000
    if isinstance(imgs[0], np.ndarray):
        imgs = [Image.fromarray(img) for img in imgs]
    imgs[0].save(saveName, save_all=True, append_images=imgs, duration=duration, loop=loop)


save_path = 'siren_ones'
imgs = list()
paths = [os.path.join(f'visualization/{save_path}/{i}.jpg', ) for i in range(0, 500, 10)]
for p in paths:
    img = Image.open(p)
    imgs.append(img)
imgs2gif(imgs, f'{save_path}.gif', duration=0.033 * 10)