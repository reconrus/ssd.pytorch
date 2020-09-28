from __future__ import print_function
import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES, SHIP_CLASSES
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES, ShipsDetection, ShipsTargetTransform, ShipBox
import torch.utils.data as data
from ssd import build_ssd
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to test model')
parser.add_argument('--root', default=VOC_ROOT, help='Location of Data root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh, labelmap):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'ships_boxes.txt'
    num_images = len(testset)
    results = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])

    for i in tqdm(range(num_images)):
        ships = list()
        # print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id = testset.ids[i]
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = x.unsqueeze(0)
        
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.3:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('image ' + str(img_id) + '\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                ships.append(ShipBox(pt))
                j += 1
                pred_num += 1

        # check intersections of ship boxes
        removed_ships = set()
        for i, box in enumerate(ships):
            for j, other_box in enumerate(ships):
                if i == j:
                    continue
                if box.intersects(other_box):
                    # remove ship with less area box
                    if box > other_box:
                        removed_ships.add(j)
                    else:
                        removed_ships.add(i)

        # write non-eleminated ship boxes to file
        for i, box in enumerate(ships):
            if i in removed_ships:
                continue
            pt = box.pt
            coords = (pt[0], pt[1], pt[2], pt[3])

            results = results.append({"ImageId": img_id,
                            "EncodedPixels": ShipsTargetTransform.pixels_to_rle(pt)},
                            ignore_index=True)
            with open(filename, mode='a') as f:
                f.write(' '.join(str(c) for c in coords) + '\n')

        if len(ships) - len(removed_ships) == 0:
            results = results.append({"ImageId": img_id,
                                      "EncodedPixels": ''},
                                     ignore_index=True)

    results.to_csv(save_folder+'submission.csv', index=False)


def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection(args.root, [('2007', 'test')], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold, labelmap=VOC_CLASSES)


def test_ships():
    num_classes = 1 + 1  # ships + background
    net = build_ssd('test', 300, num_classes)
    net.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cpu')))
    net.eval()
    print('Finished loading model!')
    testset = ShipsDetection(args.root, image_set='test')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold, labelmap=SHIP_CLASSES)


if __name__ == '__main__':
    # 42f290388.jpg
    torch.cuda.empty_cache()
    test_ships()
