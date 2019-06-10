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
BTS_CLASSES = (
    'undefined traffic sign (TS) class type',   #-1 ... 2040
    'other defined TS',                         # 0 ... 1705
    'triangles',                                # 1 ... 765
    'redcircles',                               # 2 ... 891
    'bluecircles',                              # 3 ... 1026
    'redbluecircles',                           # 4 ... 455
    'diamonds',                                 # 5 ... 291
    'revtriangle',                              # 6 ... 252
    'stop',                                     # 7 ... 43
    'forbidden',                                # 8 ... 375
    'squares',                                  # 9 ... 414
    'rectanglesup',                             # 10 ... 540
    'rectanglesdown',                           # 11 ... 54
    )
                                                # Total 8851 Annotations (Training) in 5905 images

BTS_ROOT = osp.join(HOME, "data/BelgiumTS/")

class BTSAnnotationTransform(object):
    """Transforms a BTS annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    def __init__(self, only_superclass=True):
        self.class_to_ind = dict(
            zip(BTS_CLASSES, range(-1, len(BTS_CLASSES)))
            )
        self.only_superclass = only_superclass

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                training|testing|all
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
		obj = target.split(';')
			
		pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i in range(1,4):
            cur_pt = int(obj[i]) - 1
            # scale height or width
            cur_pt = cur_pt / height if i % 2 == 0 else cur_pt / width
            bndbox.append(cur_pt)
				
		label_idx = self.class_to_ind[int(obj[6])]
        bndbox.append(label_idx)
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

    def __init__(self, root, image_set, transform=None, target_transform=BTSAnnotationTransform()):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = osp.join('%s', 'BelgiumTSD_annotations', 'BTSD_%s.txt')
        self._imgpath = osp.join('%s','%s')
        self.ids = list()
        for set in ['training', 'testing']:
			if image_set==set or image_set=='all':
				for line in open(osp.join(rootpath, 'BelgiumTSD_annotations', 'BTSD_%s.txt' % set)):
					row = line.split(';');
					self.ids.append(
						(
							osp.join(rootpath, row[0])
							, row[1]
							, row[2]
							, row[3]
							, row[4]
							, row[6] if self.target_transform.only_superclass else row[5]
						)
					)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

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
		img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

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
        img_id = self.ids[index]
        return cv2.imread(img_id[0], cv2.IMREAD_COLOR)

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
        #anno = ET.parse(self._annopath % img_id).getroot()
        #gt = self.target_transform(anno, 1, 1)
        return index, [tuple(img_id[5], tuple(img_id[1:4]))]
		
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

    
        
