import sys
import random
import argparse

import utils
from utils import visualize
from utils.utils import CocoDataset


def main(coco_dir, num_plot_examples):

    dataset = CocoDataset()
    dataset.load_coco(coco_dir, "train")
    dataset.prepare()

    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    # plot masks for each class
    for _ in range(num_plot_examples):
        random_image_id = random.choice(dataset.image_ids)
        image = dataset.load_image(random_image_id)
        mask, class_ids = dataset.load_mask(random_image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

    # Plot display instances
    for _ in range(num_plot_examples):
        random_image_id = random.choice(dataset.image_ids)
        image = dataset.load_image(random_image_id)
        mask, class_ids = dataset.load_mask(random_image_id)
        bbox = utils.utils.extract_bboxes(mask)
        visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO dataset')
    parser.add_argument('--coco_dir', help="Data dir of the coco dataset", default="data/cityscapes", type=str)
    parser.add_argument('--num_examples', help="Number of examples to be plotted", default=5, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.coco_dir, args.num_examples)

