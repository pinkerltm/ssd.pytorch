from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
''' Superclasses   
-1.   undefined traffic sign (TS) class type; %-1
0.    other defined TS = [all the other defined TS ids besides the following 11]; %0
1.    triangles = [2 3 4 7 8 9 10 12 13 15 17 18 22 26 27 28 29 34 35];   %1 (corresponds to Danger superclass in GTSDB)
2.    redcircles  = [36 43 48 50 55 56 57 58 59 61 65]; %2 (corresponds to Prohibitory superclass in GTSDB)
3.    bluecircles = [72 75 76 78 79 80 81];    %3 (corresponds to Mandatory superclass in GTSDB)
4.    redbluecircles = [82 84 85 86];    %4
5.    diamonds = [32 41]; %5
6.    revtriangle = [31]; %6
7.    stop = [39]; %7
8.    forbidden = [42];%8
9.    squares = [118 151 155 181]; %9
10.    rectanglesup  = [37,87,90,94,95,96,97,149,150,163];%10
11.    rectanglesdown= [111,112]; %11
'''
BTS_SUPERCLASSES = (
    'undefined traffic sign (TS) class type',  # -1 ... 2040
    'other defined TS',  # 0 ... 1705
    'triangles',  # 1 ... 765
    'redcircles',  # 2 ... 891
    'bluecircles',  # 3 ... 1026
    'redbluecircles',  # 4 ... 455
    'diamonds',  # 5 ... 291
    'revtriangle',  # 6 ... 252
    'stop',  # 7 ... 43
    'forbidden',  # 8 ... 375
    'squares',  # 9 ... 414
    'rectanglesup',  # 10 ... 540
    'rectanglesdown',  # 11 ... 54
)
# Total 8851 Annotations (Training) in 5905 images

BTS_ROOT = osp.join(HOME, "data/BelgiumTS")


class BTSAnnotationTransform(object):
    """Transforms a BTS annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:

        height (int): height
        width (int): width
    """

    def __init__(self, only_superclass=True):
        self.superclass_to_label = dict(
            zip(BTS_SUPERCLASSES, range(-1, len(BTS_SUPERCLASSES)))
        )
        self.label_to_superclass = dict(
            zip(range(-1, len(BTS_SUPERCLASSES)), BTS_SUPERCLASSES)
        )
        if not only_superclass:
            self.class_to_label = {}
            self.label_to_class = {}
            # TODO implement Classes loading
        self.only_superclass = only_superclass

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                iterable/list of annotation tuples from the csv file
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []

        for line in target:
            obj = line #.split(';') # as these are already tuples
            bndbox = []
            for i in range(1, 4):
                cur_pt = int(float(obj[i])) - 1
                # scale height or width
                cur_pt = cur_pt / height if i % 2 == 0 else cur_pt / width
                bndbox.append(cur_pt)

                label = obj[6] if self.only_superclass else obj[5]
                #label_idx = self.class_to_ind[int()] # TODO is here a mapping needed?
                bndbox.append(int(label))
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class BTSDetection(data.Dataset):
    """BTS Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to BelgiumTS folder.
        image_set (string): imageset to use (eg. 'training', 'testing', 'all')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_set='all',
                 transform=None,
                 target_transform=BTSAnnotationTransform(),
                 dataset_name='BTS'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'BelgiumTSD_annotations', 'BTSD_%s_GTclear.txt')
        self._imgpath = root
        self.anno = list()
        for set in ['training', 'testing']:
            if image_set == set or image_set == 'all':
                for line in open(self._annopath % (root, set)):
                    row = line.split(';')
                    (cam, image) = row[0].split('/')
                    self.anno.append(
                        (   osp.join(root, cam, image)
                            , row[1]
                            , row[2]
                            , row[3]
                            , row[4]
                            , row[5]
                            , row[6] #if self.target_transform.only_superclass else row[5]
                        )
                    )

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.anno)

    '''
    Annotation format
    =================
    The format of the annotation files is as follows:

    [camera]/[image];[x1];[y1];[x2];[y2];[class id];[superclass id];[pole id];[number on pole];[camera number];[frame number];[class label]

    Example:
    "01/image.000945.jp2;1066.23;383.12;1109.31;429.08;80;3;4;1;1;945;D7;"

    The *clear.txt have only the following information:
    [camera]/[image];[x1];[y1];[x2];[y2];[class id];[superclass id];

    Example:
    "01/image.000945.jp2;1066.23;383.12;1109.31;429.08;80;3;"
    '''

    def pull_item(self, index):
        item = self.anno[index]
        target = [item]
        img = cv2.imread(item[0])
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        # TODO this will not work I am afraid
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        item = self.anno[index]
        return cv2.imread(item[0], cv2.IMREAD_COLOR)


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
        item = self.anno[index]
        # anno = ET.parse(self._annopath % img_id).getroot()
        # gt = self.target_transform(anno, 1, 1)
        return index, [tuple(item[6], tuple(item[1:4]))]


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
