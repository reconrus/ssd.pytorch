"""VOC Dataset Classes

Authors: Vyacheslav Yastrebov, Danil Ginzburg
"""
from .config import HOME
import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd

from skimage.measure import label, regionprops

# note: if you used our download scripts, this should be right
SHIPS_ROOT = os.path.join(HOME, "data/ships/")
SHIP_LABEL = 0

class ShipsTargetTransform(object):
    """Transforms a Ships annotation into a Tensor of bbox coords and label index

    Arguments:
        height (int): height
        width (int): width
    """
    def __call__(self, targets, width, height):
        """
        Arguments:
            target (annotation) : the target segmentation in run-length encoding format
                
        Returns:
            a list containing lists of bounding boxes
        """
        bboxes = []
        for target in targets:
            if not target:
                continue
            pixels = self.rle_to_pixels(target)
            bboxes.extend(self.get_bboxes(pixels, width, height))

        return bboxes  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

    # src: https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes#2.-Understanding-and-plotting-rle-bounding-boxes
    def rle_to_pixels(self, rle_code):
        '''
        Transforms a RLE code string into a list of pixels of a (768, 768) canvas
        '''
        rle_code = [int(i) for i in rle_code.split()]
        pixels = [(pixel_position % 768, pixel_position // 768) if pixel_position // 768 != 768 else (pixel_position % 768, 767)
                    for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2])) 
                    for pixel_position in range(start, start + length)]
        return pixels

    # source: https://www.kaggle.com/voglinio/from-masks-to-bounding-boxes
    def get_bboxes(self, pixels, width, height):
        mask = np.zeros((height, width))
        mask[tuple(zip(*pixels))] = 1
        lbl_0 = label(mask) 
        props = regionprops(lbl_0)
        return [(prop.bbox[1]/height, prop.bbox[0]/width, prop.bbox[3]/height, prop.bbox[2]/width, SHIP_LABEL) for prop in props]


class ShipsDetection(data.Dataset):
    """Ships Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to Ships dataset folder.
        image_set (string): imageset to use ('train' or'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_set='train',
                 transform=None, target_transform=ShipsTargetTransform,
                 dataset_name='AirbusShips'):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform()
        self.name = dataset_name
        self.datapath = os.path.join(self.root, f'{image_set}_v2')

        targetpath = os.path.join(self.root, 'train_ship_segmentations_v2.csv')
        self.targets = pd.read_csv(targetpath, index_col='ImageId')

        # do not consider images without ships
        self.ids = [img_id for img_id in os.listdir(self.datapath) if self.targets.loc[img_id]['EncodedPixels'] is not np.nan]


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        img = cv2.imread(os.path.join(self.datapath, img_id), cv2.IMREAD_COLOR)
        height, width, channels = img.shape

        target = self.targets.loc[img_id]['EncodedPixels']
        target = [target] if type(target) == str else target.astype(str).values.tolist()
        target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(os.path.join(self.datapath, img_id), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        target = self.targets.loc[img_id]['EncodedPixels'].astype(str).values.tolist()
        gt = self.target_transform(target, 1, 1)
        return img_id, gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
