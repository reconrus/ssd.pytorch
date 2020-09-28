import cv2
import argparse
from data.config import *
from math import ceil

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection Vizualization Part')
parser.add_argument('--filepath', default='eval/ships_boxes.txt',
                    type=str, help='Path to file with ship boxes')
parser.add_argument('--in_dir', default='train_partial/test_v2/',
                    type=str, help='Path to directory with original images')
parser.add_argument('--out_dir', default='eval/results/',
                    type=str, help='Path to directory where vizualaized results are stored')
args = parser.parse_args()


def draw_from_file(filepath, in_dir, out_dir):
    ships = list()
    with open(filepath, 'r') as file:
        for line in file.readlines():
            if line.startswith('image'):
                if ships:  #
                    draw_rectangles(os.path.join(in_dir, image_name), os.path.join(out_dir, image_name), ships)
                image_name = line.split(' ')[1]
                if image_name.endswith('\n'):
                    image_name = image_name[:-1]
                ships = list()
            else:
                ships.append(tuple(map(lambda x: int(float(x)), line.split(' '))))


def draw_rectangles(image_in, image_out, bboxes_pred, bboxes_gt=None):
    image = cv2.imread(image_in)
    for bbox_pred in bboxes_pred:
        bbox_pred = tuple(map(ceil, bbox_pred))
        image = cv2.rectangle(image, (bbox_pred[0], bbox_pred[1]), (bbox_pred[2], bbox_pred[3]), (0, 255, 0), 2)

    if bboxes_gt is not None:
        for bbox_gt in bboxes_gt:
            image = cv2.rectangle(image, (bbox_gt[0], bbox_gt[1]), (bbox_gt[2], bbox_gt[3]), (0, 0, 255), 2)
        image = cv2.putText(image, 'ground truth', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    image = cv2.putText(image, 'predicted', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(image_out, image)


if __name__ == '__main__':
    # bbox = (74.860214, 599.56323, 240.58575, 739.6669)
    # bbox2 = (480, 260, 553, 314)
    # bboxes_gt = [(72, 599, 237, 740), (476, 253, 556, 322)]
    # draw_rectangles("/Users/danilginzburg/ssd.pytorch/data/test.jpg",
    # "/Users/danilginzburg/ssd.pytorch/data/testResult.jpg", [bbox, bbox2], bboxes_gt)
    draw_from_file(args.filepath, args.in_dir, args.out_dir)

