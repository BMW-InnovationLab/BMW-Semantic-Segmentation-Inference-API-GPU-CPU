"""Pascal VOC Semantic Segmentation Dataset."""
import os
import numpy as np
from PIL import Image
from mxnet import cpu
import mxnet.ndarray as F
from ..segbase import SegmentationDataset

class VOCSegmentation(SegmentationDataset):
    """Pascal VOC Semantic Segmentation Dataset.
    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is '$(HOME)/mxnet/datasets/voc'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from mxnet.gluon.data.vision import transforms
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = gluoncv.data.VOCSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = gluon.data.DataLoader(
    >>>     trainset, 4, shuffle=True, last_batch='rollover',
    >>>     num_workers=4)
    """
    BASE_DIR = 'VOC2012'
    NUM_CLASS = 0
    def __init__(self,oclasses=0,featurespath=None,labelspath=None, root=os.path.expanduser('~/.mxnet/datasets/voc'),
                 split='train', mode=None, transform=None,**kwargs):
        super(VOCSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        entries = os.listdir(featurespath)
        entries=sorted(entries)
        self.images=[]
        VOCSegmentation.NUM_CLASS=oclasses
        for entry in entries :
                 self.images.append(featurespath+'/'+entry)

        entries = os.listdir(labelspath)
        entries=sorted(entries)
        self.masks=[]
        for entry in entries :
                self.masks.append(labelspath+'/'+entry)





    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')

        return F.array(target, cpu(0))

    @property
    def classes(self):
        """Category names."""
        return apocalibsyy
