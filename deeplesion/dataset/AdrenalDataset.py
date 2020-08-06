import sys
sys.path.append('/cluster/home/it_stu167/wwj/alignshift/AlignShift/')
import numpy as np
import random
import os
import csv
import cv2
import logging 
#from pycocotools import mask as mutils
from mmcv import Config
import torch
import os
from mmdet.datasets.registry import DATASETS
import pickle
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.custom import CustomDataset
from skimage import measure

@DATASETS.register_module
class AdrenalDataset(CustomDataset):

    CLASSES = ('anomoly','adrenal')# maybe wrong
    def __init__(self, 
                 ann_file, 
                 pipeline,
                 pre_pipeline,
                 dicm2png_cfg,
                 data_root=None, 
                 image_path='/cluster/home/it_stu167/wwj/adrenal/x/',
                 label_path='/cluster/home/it_stu167/wwj/adrenal/y/',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 ratio=0.5):
        self.data_path = data_root
        self.classes = ['__background__', 'anomoly','adrenal']
        self.num_classes = len(self.classes)
        self.load_annotations(ann_file)
        self.img_ids = [a['filename'] for a in self.ann]
        self.cat_ids = self.classes
        self.cfg = Config(dicm2png_cfg)   
        self.pipeline = Compose(pipeline)
        self.pre_pipeline = Compose(pre_pipeline)
        self.img_path = image_path
        self.seg_prefix = seg_prefix
        self.ratio=ratio
        self.label_path=label_path
        self.proposals = None
        if proposal_file is not None:
            self.proposals = None
        self.slice_num=3
        # self.slice_num = self.cfg.NUM_SLICES # 3
        self.is_train = not test_mode
        # if self.is_train:
        #     self._set_group_flag()  # ???


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, info).
        """
        ann = self.ann[index]
        image_fn = ann['filename']
        slice_intv = ann['pixdim'][2]
        spacing = ann['pixdim']
        space_origin=ann['space origin'] # maybe unnecessary
        ratio=self.ratio        
        rg=ann['rg']
        im, im_scale, idx = load_prep_np(self.img_path, image_fn, spacing, slice_intv,
                                            num_slice=self.slice_num, is_train=self.is_train, ratio=ratio, rg=rg)        
        label=get_labels(self.label_path,image_fn,idx)
        masks=get_masks_from_labels(label) # n*b*w
        boxes = get_boxes_from_masks(masks) # n*4
        boxes = self.clip_to_image(boxes, im.shape, False) 

        #masks = masks.transpose((2, 0, 1))
        boxes = boxes.astype(np.float32)
        infos = {'masks': masks,    #dict
                 'bboxes': boxes,
                 'labels': label,
                }
        results = dict()#img_info=ann, ann_info=infos
        results['filename'] = image_fn
        # results['flage'] = flage
        results['img'] = im
        results['img_shape'] = im.shape #512*512

        results['ori_shape'] = im.shape#[ann['height'], ann['width']] # ???
        if self.proposals is not None:  
            results['proposals'] = self.proposals[index]

        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['gt_bboxes'] = boxes
        results['bbox_fields'].append('gt_bboxes')
        results['gt_labels'] = label.astype(np.int64)
        results['gt_masks'] = masks
        results['mask_fields'].append('gt_masks')
        results['thickness'] = slice_intv
        results = self.pre_pipeline(results)
        # results['gt_masks'] = masks_scaled
        # results['mask_fields'].append('gt_masks')
        
        return self.pipeline(results)


    def __len__(self):
        return len(self.ann)
        # return 160
    def clip_to_image(self, bbox, shape, remove_empty=True):
        TO_REMOVE = 1
        bbox[:, 0] = bbox[:, 0].clip(min=0, max=shape[1] - TO_REMOVE)
        bbox[:, 1] = bbox[:, 1].clip(min=0, max=shape[0] - TO_REMOVE)
        bbox[:, 2] = bbox[:, 2].clip(min=0, max=shape[1] - TO_REMOVE)
        bbox[:, 3] = bbox[:, 3].clip(min=0, max=shape[0] - TO_REMOVE)
        if remove_empty:
            box = bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return bbox[keep]
        return bbox

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        if not self.cfg.GROUNP_ZSAPACING: return
        for i in range(len(self)):
            img_info = self.ann[i]
            if img_info['slice_intv'] < 2.0:
                self.flag[i] = 1
        logging.info(f'slice_intv grounped by 2.0: {sum(self.flag)}/{len(self)-sum(self.flag)}')

    def load_annotations(self, ann_file):
        """load annotations and meta-info from ann.csv"""
        with open(ann_file,'rb') as f:
            self.ann = pickle.load(f)
    


def load_prep_np(data_dir, imname, spacing, slice_intv, cfg, ratio, rg, num_slice=3, is_train=False):
    """load volume, windowing, interpolate multiple slices, clip black border, resize according to spacing"""
    im,idx = load_multislice_randomly(data_dir, imname, slice_intv, num_slice, ratio, rg)

    #im = windowing(im, cfg.WINDOWING) # 0~255 float
    im_shape = im.shape[0:2] # [512,512]
    im_scale = 1.0

    return im, im_scale, idx

def load_multislice_randomly(data_dir, imname, slice_intv, num_slice, ratio, rg):
    data_cache = {}
    def _load_data_from_np(imname, delta=0):
        imname1 = imname # get_slice_name(data_dir, imname, delta)
        if imname1 not in data_cache.keys():
            data_cache[imname1] = np.load(os.path.join(data_dir, imname1+'.npy'))[2]
            assert data_cache[imname1] is not None, 'file reading error: ' + imname1
            # if data_cache[imname1] is None:
            #     print('file reading error:', imname1)
        return data_cache[imname1]

    _load_data = _load_data_from_np
    im_cur = _load_data(imname) # 512*512*d
    depth=np.shape(im_cur)[2]
    # def get_offset(data_dir,imname):
    #     path=os.path.join(data_dir,imname+'_nrrd.csv')
    #     with open(path) as rf:
    #         reader = csv.DictReader(rf)
    #         items = list(reader)
    #     offset=items[0]['Segmentation_ReferenceImageExtentOffset'].split(" ")
    #     return [int(offset[0]),int(offset[1]),int(offset[2])]       
    idx=1
    if random.random()>ratio:
        idx=random.randint(1,depth-2)
    else:
        idx=random.randint(rg[0]+1,rg[1]-1) 
    ims=[im_cur[:,:,idx]] # 512*512
    ims=[im_cur[:,:,idx-1]]+ims+[im_cur[:,:,idx+1]] # 3*512*512 list


    ims = [im.astype(float) for im in ims]
    im = cv2.merge(ims)
    im = im.astype(np.float32,
                       copy=False) # 512*512*3 -1~1


    return im,idx

def get_labels(path,fn,idx):
    labels=np.load(os.path.join(path,fn+'.npy'))[:,:,idx]
    return labels

def get_masks_from_labels(labels):
    lb=labels.copy()
    masks=[]
    lb[lb==2]=1
    if 1 in lb[0:256,:]:
        masks.append(np.hstack(lb[0:256,:],np.zeros((512,256),dtype=np.int8)))
    if 1 in lb[256:512,:]:
        masks.append(np.hstack(np.zeros((512,256),dtype=np.int8),lb[256:512,:]))
    if 1 in labels[0:256,:]:
        left=labels[0:256,:].copy()
        left[left==2]=0
        masks.append(np.hstack(left,np.zeros((512,256),dtype=np.int8)))
    if 1 in labels[256:512,:]:
        right=labels[256:512,:].copy()
        right[right==2]=0
        masks.append(np.hstack(np.zeros((512,256),dtype=np.int8),right))
    masks=np.array(masks)
    return masks 

def get_boxes_from_masks(masks):
    n=masks.shape[0]
    bbox=[]
    for i in range(n):
        tmp=measure.regionprops(masks[i])
        bbox.append(list(tmp[0].bbox))
    bbox=np.array(bbox)
    return bbox

def windowing(im, win=None): #  ???
    """scale intensity from win[0]~win[1] to float numbers in 0~255"""
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    im1 -= 50
    return im1

'''
def windowing_rev(im, win):
    """backward windowing"""
    im1 = im.astype(float)#/255
    im1 *= win[1] - win[0]
    im1 += win[0]
    return im1


# def get_mask(im):
#     """use a intensity threshold to roughly find the mask of the body"""
#     th = 32000  # an approximate background intensity value
#     mask = im > th
#     mask = binary_opening(mask, structure=np.ones((7, 7)))  # roughly remove bed
#     # mask = binary_dilation(mask)
#     # mask = binary_fill_holes(mask, structure=np.ones((11,11)))  # fill parts like lung

#     if mask.sum() == 0:  # maybe atypical intensity
#         mask = im * 0 + 1
#     return mask.astype(dtype=np.int32)

     

def get_range(mask, margin=0):
    """Get up, down, left, right extreme coordinates of a binary mask"""
    idx = np.nonzero(mask)
    u = max(0, idx[0].min() - margin)
    d = min(mask.shape[0] - 1, idx[0].max() + margin)
    l = max(0, idx[1].min() - margin)
    r = min(mask.shape[1] - 1, idx[1].max() + margin)
    return [u, d, l, r]


def map_box_back(boxes, cx=0, cy=0, im_scale=1.):
    """Reverse the scaling and offset of boxes"""
    boxes /= im_scale
    boxes[:, [0,2]] += cx
    boxes[:, [1,3]] += cy
    return boxes

def crop_or_pading(img, fixsize):
    h,w,c = img.shape
    fh,fw = fixsize
    mh,mw = max(h, fh),max(w,fw)
    img_new = np.zeros((mh,mw,c))
    img_new[(mh-h)//2:(mh+h)//2, (mw-w)//2:(mw+w)//2, :] = img

    return img_new[(mh-fh)//2:(mh+fh)//2, (mw-fw)//2:(mw+fw)//2,:], [(mh-h)//2-(mh-fh)//2, (mw-w)//2-(mw-fw)//2]
'''
