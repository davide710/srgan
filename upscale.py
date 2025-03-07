import numpy as np
import cv2
import torch.nn as nn
import torch
from models import Generator

filename = 'messi.jpeg'
im = cv2.imread(filename)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

gen = Generator()
gen_state_dict = torch.load('gen.pth', weights_only=True, map_location=torch.device('cpu'))
gen.load_state_dict(gen_state_dict)

x = torch.tensor(im).permute(2, 0, 1) / 255
gen.eval()
up = gen(x.unsqueeze(0))
im_up = up.squeeze().permute(1, 2, 0)
im_up = np.array(im_up.detach())
im_up = np.uint8(im_up*255)
print('Writing...')
cv2.imwrite(f'{filename.split(".")[0]}_upscaled.{filename.split(".")[1]}', cv2.cvtColor(im_up, cv2.COLOR_RGB2BGR))
