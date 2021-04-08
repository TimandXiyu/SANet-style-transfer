import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from generator import generator


class RandomPairGen(object):
    def __init__(self, content_path, style_path, fake_pair_num):
        self.content_path = content_path
        self.style_path = style_path
        self.fake_pair_num = fake_pair_num
        self.content_dirlist = self.namefinder(self.content_path)
        self.style_dirlist = self.namefinder(self.style_path)
        self.content_ids = np.random.choice(len(self.content_dirlist), self.fake_pair_num, replace=True, p=None)
        for i, x in enumerate(self.content_ids):
            self.content_ids[i] = self.content_dirlist[x]
        self.style_ids = np.random.choice(len(self.style_dirlist), self.fake_pair_num, replace=True, p=None)
        for i, x in enumerate(self.style_ids):
            self.style_ids[i] = self.style_dirlist[x]

    def __call__(self):
        return self.content_ids, self.style_ids

    @staticmethod
    def namefinder(path):
        dirlist = [int(i.split('.')[0]) for i in os.listdir(path)]
        return dirlist



def RenameDataset(src_path, src_mask_path, target_path, target_mask_path):
    dirlist = os.listdir(src_path)
    mask_dirlist = os.listdir(src_mask_path)
    print(mask_dirlist[-1], dirlist[-1])
    # start = len(RandomPairGen.namefinder(target_path))
    # start_mask = len(RandomPairGen.namefinder(target_mask_path))
    start = 6226
    start_mask = 6226
    if len(dirlist) != len(mask_dirlist) and start != start_mask:
        raise FileNotFoundError('Expecting to have the same dataset and same elements in target files')
    for i, image in tqdm(enumerate(dirlist)):
        if image.split('.')[0] == mask_dirlist[i].split('.')[0]:
            img = Image.open(os.path.join(src_path, image))
            msk = Image.open(os.path.join(src_mask_path, image.split('.')[0] + '.png'))
            img.save(os.path.join(target_path, str(i + start) + '.png'))
            msk.save(os.path.join(target_mask_path, str(i + start) + '.png'))
        else:
            raise FileExistsError('target mask and image not matching')


if __name__ == "__main__":
    base = r'.\data'
    maskdir = r'.\data\content_mask'
    fake_pair_num = 2500
    target = r'E:\XiyuUnderGradThesis\data\generated_data_c1s3'
    mask_target = r'E:\XiyuUnderGradThesis\data\generated_data_c1s3'
    if not os.path.exists(target):
        os.mkdir(target)
        os.mkdir(os.path.join(target, 'fake_img'))
        os.mkdir(os.path.join(mask_target, 'fake_mask'))

    target = os.path.join(target, 'fake_img')
    mask_target = os.path.join(mask_target, 'fake_mask')
    content_img = os.path.join(base, 'content_cz')
    style_img = os.path.join(base, 'style_cz')
    content_ids, style_ids = RandomPairGen(os.path.join(base, 'content_cz'),
                                           os.path.join(base, 'style_cz'),
                                           fake_pair_num=fake_pair_num)()
    content_ids = [int(i) for i in list(content_ids)]
    style_ids = [int(i) for i in list(style_ids)]
    for i in tqdm(range(fake_pair_num)):
        content = os.path.join(content_img, str(content_ids[i]) + '.png')
        style = os.path.join(style_img, str(style_ids[i]) + '.png')
        if not os.path.exists(content) and os.path.exists(style):
            print(content, '\n', style, '\n', 'file not exist')
        generator(content, style, steps=1, output=target, counter=i)
        content_base = os.path.basename(content)
        mask_pth = os.path.join(content_base)
        save_pth = os.path.join(mask_target, content_base)
        mask = Image.open(os.path.join(maskdir, content_base))
        mask.save(os.path.join(mask_target, str(i) + '.png'))
    gen = RandomPairGen(content_path=os.path.join(base, r'train_img'),
                        style_path=os.path.join(base, r'cropped_cz_src'),
                        fake_pair_num=fake_pair_num)
    # base = r'E:\XiyuUnderGradThesis\data'
    # RenameDataset(os.path.join(base, r'generated_data_c1s3\fake_img'),
    #                 os.path.join(base, r'generated_data_c1s3\fake_mask'),
    #                 os.path.join(base, r'dp+c1s3x2500\train_img'),
    #                 os.path.join(base, r'dp+c1s3x2500\train_mask'))
