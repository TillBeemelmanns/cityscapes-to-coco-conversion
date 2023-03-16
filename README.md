## Cityscapes to COCO Conversion Tool
![](assets/preview.png)

Forked from https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion

This script allows to convert the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) to Mircosoft's [COCO Format](http://cocodataset.org/). The code heavily relies on Facebook's [Detection Repo](https://github.com/facebookresearch/Detectron/blob/master/tools/convert_cityscapes_to_coco.py) and [Cityscapes Scripts](https://github.com/mcordts/cityscapesScripts).

The converted annotations can be easily used for [Mask-RCNN](https://github.com/matterport/Mask_RCNN) or other deep learning projects.


## Folder Structure
Download the Cityscapes Dataset and organize the files in the following structure. Create an empty `annotations` directory.
```
data/
└── cityscapes
    ├── annotations
    ├── gtFine
    │   ├── test
    │   ├── train
    │   └── val
    └── leftImg8bit
        ├── test
        ├── train
        └── val
utils/
main.py
inspect_coco.py
README.md
requirements.txt
```

## Installation
```shell
pip install -r requirements.txt 
```

## Run
To run the conversion execute the following
```shell
python main.py --dataset cityscapes --datadir data/cityscapes --outdir data/cityscapes/annotations
```

Takes about 12 minutes to execute.

The script will create the files

- ```instancesonly_filtered_gtFine_train.json```
- ```instancesonly_filtered_gtFine_val.json```

in the directory ```annotations``` for the ```train``` and ```val``` split which contain the Coco annotations.

The variable category_instancesonly defines which classes should be considered in the conversion process. By default has this value:

```python
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
```

which in COCO format (in .yaml file format) is

```yaml
NUM_CLASSES: 9
CLASSES: [
  { 'supercategory': 'none', 'id': 0, 'name': 'background' },
  { 'supercategory': 'none', 'id': 1, 'name': 'person' },
  { 'supercategory': 'none', 'id': 2, 'name': 'rider' },
  { 'supercategory': 'none', 'id': 3, 'name': 'car' },
  { 'supercategory': 'none', 'id': 4, 'name': 'bicycle' },
  { 'supercategory': 'none', 'id': 5, 'name': 'motorcycle' },
  { 'supercategory': 'none', 'id': 6, 'name': 'bus' },
  { 'supercategory': 'none', 'id': 7, 'name': 'truck' },
  { 'supercategory': 'none', 'id': 8, 'name': 'train' },
]
```

Sometimes the segmentation annotations are so small that no reasonable big enough object could be created. In this case the, the object will be skipped and the following message is printed:

```
Warning: invalid contours.
```

In order to run the visualization of the COCO dataset you may run
```shell
python inspect_coco.py --coco_dir data/cityscapes
```

## Output
![vis1](assets/plot1.png "Cityscapes in COCO format") ![vis2](assets/plot2.png "Cityscapes in COCO format")