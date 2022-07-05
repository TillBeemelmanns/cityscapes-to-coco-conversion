# This file heavily borrows from https://github.com/facebookresearch/Detectron/tree/master/tools

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

# Image processing
# Check if PIL is actually Pillow as expected
try:
    from PIL import __version__
except:
    print("Please install the module 'Pillow' for image processing, e.g.")
    print("pip install pillow")
    sys.exit(-1)

try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

import argparse
import json
import os
import cv2
import numpy as np

from utils.instance_class import *
from utils.labels import *

def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4
    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy

def instances2dict_with_polygons(imageFileList, verbose=False):
    imgCount     = 0
    instanceDict = {}

    if not isinstance(imageFileList, list):
        imageFileList = [imageFileList]

    if verbose:
        print("Processing {} images...".format(len(imageFileList)))

    for imageFileName in imageFileList:
        # Load image
        img = Image.open(imageFileName)

        # Image as numpy array
        imgNp = np.array(img)

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            if instanceId < 1000:
                continue
            instanceObj = Instance(imgNp, instanceId)
            instanceObj_dict = instanceObj.toDict()

            if id2label[instanceObj.labelID].hasInstances:
                mask = (imgNp == instanceId).astype(np.uint8)
                contour, hier = findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                polygons = [c.reshape(-1).tolist() for c in contour]
                instanceObj_dict['contours'] = polygons

            instances[id2label[instanceObj.labelID].name].append(instanceObj_dict)

        instanceDict[imageFileName] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=' ')
            sys.stdout.flush()

    if verbose:
        print("")

    return instanceDict

def poly_to_box(poly):
    """Convert a polygon into a tight bounding box."""
    x0 = min(min(p[::2]) for p in poly)
    x1 = max(max(p[::2]) for p in poly)
    y0 = min(min(p[1::2]) for p in poly)
    y1 = max(max(p[1::2]) for p in poly)
    box_from_poly = [x0, y0, x1, y1]
    return box_from_poly

def xyxy_to_xywh(xyxy_box):
    xmin, ymin, xmax, ymax = xyxy_box
    TO_REMOVE = 1
    xywh_box = (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE)
    return xywh_box

def convert_cityscapes_instance_only(data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    sets = [
        'leftImg8bit/train',
        'leftImg8bit/val'
    ]

    ann_dirs = [
        'gtFine/train',
        'gtFine/val',
    ]

    json_name = 'instancesonly_filtered_%s.json'
    polygon_json_file_ending = '_polygons.json'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    category_instancesonly = [
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',
    ]

    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []

        for root, _, files in os.walk(os.path.join(data_dir, ann_dir)):
            for filename in files:
                if filename.endswith(polygon_json_file_ending):

                    if len(images) % 50 == 0:
                        print("Processed %s images, %s annotations" % (len(images), len(annotations)))

                    json_ann = json.load(open(os.path.join(root, filename)))

                    image = {}
                    image['id'] = img_id
                    img_id += 1
                    image['width'] = json_ann['imgWidth']
                    image['height'] = json_ann['imgHeight']
                    image['file_name'] = os.path.join("leftImg8bit",
                                                      data_set.split("/")[-1],
                                                      filename.split('_')[0],
                                                      filename.replace("_gtFine_polygons.json", '_leftImg8bit.png'))
                    image['seg_file_name'] = filename.replace("_polygons.json", "_instanceIds.png")
                    images.append(image)

                    fullname = os.path.join(root, image['seg_file_name'])
                    objects = instances2dict_with_polygons([fullname], verbose=False)[fullname]

                    for object_cls in objects:
                        if object_cls not in category_instancesonly:
                            continue  # skip non-instance categories

                        for obj in objects[object_cls]:
                            if obj['contours'] == []:
                                print('Warning: empty contours.')
                                continue  # skip non-instance categories

                            len_p = [len(p) for p in obj['contours']]
                            if min(len_p) <= 4:
                                print('Warning: invalid contours.')
                                continue  # skip non-instance categories

                            ann = {}
                            ann['id'] = ann_id
                            ann_id += 1
                            ann['image_id'] = image['id']
                            ann['segmentation'] = obj['contours']

                            if object_cls not in category_dict:
                                category_dict[object_cls] = cat_id
                                cat_id += 1
                            ann['category_id'] = category_dict[object_cls]
                            ann['iscrowd'] = 0
                            ann['area'] = obj['pixelCount']

                            xyxy_box = poly_to_box(ann['segmentation'])
                            xywh_box = xyxy_to_xywh(xyxy_box)
                            ann['bbox'] = xywh_box

                            annotations.append(ann)

        ann_dict['images'] = images
        categories = [{"id": category_dict[name], "name": name} for name in category_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        if not os.path.exists(os.path.abspath(out_dir)):
            os.mkdir(os.path.abspath(out_dir))
        with open(os.path.join(out_dir, json_name % ann_dir.replace("/", "_")), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--dataset', help="cityscapes", default='cityscapes', type=str)
    parser.add_argument('--outdir', help="output dir for json files", default='data/cityscapes/annotations', type=str)
    parser.add_argument('--datadir', help="data dir for annotations to be converted", default="data/cityscapes", type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "cityscapes":
        convert_cityscapes_instance_only(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)
