import torch.nn as nn
import logging
from SANet import Net
import torch.backends.cudnn
from Infloader import eval_transform
from PIL import Image
import torch
from torchvision.utils import save_image
from decoder import decoder
from vgg import vgg
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import os
import shutil


def gen_image(content_pth, style_pth, output_folder, serial_num, vgg, decoder, ext='.png'):
    vgg_pth = r'./vgg_normalised.pth'
    decoder = decoder()
    vgg = vgg()
    start_iter = 0
    content_weight = 2.
    style_weight = 3.
    torch.backends.cudnn.benchmark = True

    vgg.load_state_dict(torch.load(vgg_pth))
    vgg = nn.Sequential(*list(vgg.children())[:44])
    network = Net(vgg, decoder, start_iter)
    network.decoder.load_state_dict(torch.load(r'E:\SANet CP\style1content3\decoder_iter_300000.pth'))
    network.transform.load_state_dict(torch.load(r'E:\SANet CP\style1content3\transformer_iter_300000.pth'))
    network.train()
    network.cuda()

    content_tf = eval_transform()
    style_tf = eval_transform()

    content = content_tf(Image.open(content_pth))
    style = style_tf(Image.open(style_pth))

    content = content.unsqueeze(dim=0).cuda()
    style = style.unsqueeze(dim=0).cuda()

    with torch.no_grad():

        image, loss_c, loss_s, l_identity1, l_identity2 = network(content, style, gen=True)
        # loss_c = content_weight * loss_c
        # loss_s = style_weight * loss_s
        # loss = loss_c + loss_s + l_identity1 + l_identity2 * 50

        # if loss.item() >= 130:
        #     logging.info('Loss is too high')
        #     return False
        # logging.info('Loss qualified')
        image = image.clamp(0, 255)
        image.cpu()
        output_name = '{:s}/{:s}{:s}'.format(output_folder, str(serial_num), ext)
        save_image(image, output_name)
        return True

def gen_pair(content_folder, style_folder, n_pair):
    suffix = '.png'
    content = [str(Path(x).stem) for x in glob(content_folder)]
    style = [str(Path(x).stem) for x in glob(style_folder)]
    content = list(map(lambda x: str(Path(content_folder).parent / Path(x + suffix)),
                       list(np.random.choice(content, n_pair, replace=True, p=None))))
    style = list(map(lambda x: str(Path(style_folder).parent / Path(x + suffix)),
                     list(np.random.choice(style, n_pair, replace=True, p=None))))
    zipped = dict(zip(content, style))
    content_masks = list(map(lambda x: Path(x).parent.parent / 'train_mask' / Path(str(Path(x).stem) + '.png'), zipped))
    return dict(zip(content, style)), content_masks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    content_folder = r'F:/XiyuUnderGradThesis/data/train_img/*.png'
    style_folder = r'F:/XiyuUnderGradThesis/data/cropped_cz_src/*.png'
    output_folder = Path(r'F:/XiyuUnderGradThesis/data/generated_cz_data_c3s1')
    pairs, content_masks = gen_pair(content_folder, style_folder, n_pair=1000)
    logging.info('total length', len(pairs))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(output_folder / 'generate_img'):
        os.mkdir(output_folder / 'generate_img')
    if not os.path.exists(output_folder / 'generate_mask'):
        os.mkdir(output_folder / 'generate_mask')
    for i, key in tqdm(enumerate(pairs)):
        content_mask = content_masks[i]
        valid = gen_image(key, pairs[key], str(output_folder / 'generate_img'),
                          serial_num=str(i + 6226),
                          vgg=vgg,
                          decoder=decoder)
        if valid:
            shutil.copy(content_mask, output_folder / 'generate_mask' / Path(str(i + 6226) + '.png'))




