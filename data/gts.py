from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

GTS_CLASSES = (
	'0 = speed limit 20 (prohibitory)',
	'1 = speed limit 30 (prohibitory)',
	'2 = speed limit 50 (prohibitory)',
	'3 = speed limit 60 (prohibitory)',
	'4 = speed limit 70 (prohibitory)',
	'5 = speed limit 80 (prohibitory)',
	'6 = restriction ends 80 (other)',
	'7 = speed limit 100 (prohibitory)',
	'8 = speed limit 120 (prohibitory)',
	'9 = no overtaking (prohibitory)',
	'10 = no overtaking (trucks) (prohibitory)',
	'11 = priority at next intersection (danger)',
	'12 = priority road (other)',
	'13 = give way (other)',
	'14 = stop (other)',
	'15 = no traffic both ways (prohibitory)',
	'16 = no trucks (prohibitory)',
	'17 = no entry (other)',
	'18 = danger (danger)',
	'19 = bend left (danger)',
	'20 = bend right (danger)',
	'21 = bend (danger)',
	'22 = uneven road (danger)',
	'23 = slippery road (danger)',
	'24 = road narrows (danger)',
	'25 = construction (danger)',
	'26 = traffic signal (danger)',
	'27 = pedestrian crossing (danger)',
	'28 = school crossing (danger)',
	'29 = cycles crossing (danger)',
	'30 = snow (danger)',
	'31 = animals (danger)',
	'32 = restriction ends (other)',
	'33 = go right (mandatory)',
	'34 = go left (mandatory)',
	'35 = go straight (mandatory)',
	'36 = go right or straight (mandatory)',
	'37 = go left or straight (mandatory)',
	'38 = keep right (mandatory)',
	'39 = keep left (mandatory)',
	'40 = roundabout (mandatory)',
	'41 = restriction ends (overtaking) (other)',
	'42 = restriction ends (overtaking (trucks)) (other)'
)
''' Belgian Superclasses   
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
GTS_SUPERCLASSES = (
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

 
  
GTS_ROOT = osp.join(HOME, "data", "GTSDB")

prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16] # (circular, white ground with red border) --> redcircles
mandatory = [33, 34, 35, 36, 37, 38, 39, 40] # (circular, blue ground) --> bluecircles
danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # (triangular, white ground with red border) --> triangles
	

class GTSAnnotationTransform(object):
    """Transforms a GTS annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:

        height (int): height
        width (int): width
    """

    def __init__(self, only_superclass=True):
        self.superclass_to_label = dict(
            zip(GTS_SUPERCLASSES, range(-1, len(GTS_SUPERCLASSES)))
        )
        self.label_to_superclass = dict(
            zip(range(-1, len(GTS_SUPERCLASSES)), GTS_SUPERCLASSES)
        )
        self.label_to_id = dict(
            zip(range(-1, len(GTS_SUPERCLASSES)-1), range(0, len(GTS_SUPERCLASSES)))
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
        if height == 0 or width == 0:
            raise Exception('Division by zero possible!')

        for line in target:
            #print(line)
            #print('w: {} - h: {}'.format(width, height))
            obj = line #.split(';') # as these are already tuples
            bndbox = []
            for i in range(1, 5):
                cur_pt = int(float(obj[i]))-1
                # scale height or width
                cur_pt = cur_pt / height if i % 2 == 0 else cur_pt / width
                bndbox.append(cur_pt)

            label = self.label_to_btssuperclass(int(obj[5])) if self.only_superclass else obj[5]
            #label_idx = self.class_to_ind[int()] # TODO is here a mapping needed?
            bndbox.append(int(label))
                
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        # img_id = target.find('filename').text[:-4]
        #print(res)
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]
		
    def label_to_btssuperclass(self, label):
        if label in danger:
            return 2
        if label in prohibitory:
            return 3
        if label in mandatory:
            return 4

        return 0	

class GTSDetection(data.Dataset):
    """GTS Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to BelgiumTS folder.
        image_set (string): imageset to use (eg. 'Train', 'Test', 'Full')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_set='Full',
                 transform=None,
                 target_transform=GTSAnnotationTransform(),
                 dataset_name='GTS'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', '%sIJCNN2013', 'gt.txt')
        self._imgpath = osp.join('%s', '%sIJCNN2013')
        self.anno = list()
        for line in open(self._annopath % (root, image_set)):
            row = line.split(';')
            
            image = row[0]
            self.anno.append(
                (   osp.join(self._imgpath % (root, image_set), image)
                        , row[1]
                        , row[2]
                        , row[3]
                        , row[4]
                        , row[5]
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
        return index, [tuple(item[5], tuple(item[1:4]))]


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
