import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose
from . import CustomDataset


@DATASETS.register_module()
class iCartoonFacePlus(CustomDataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4),
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        super(iCartoonFacePlus, self).__init__(ann_file,
                 pipeline,
                 classes=classes,
                 data_root=data_root,
                 img_prefix=img_prefix,
                 seg_prefix=seg_prefix,
                 proposal_file=proposal_file,
                 test_mode=test_mode,
                 filter_empty_gt=filter_empty_gt)

        if not self.test_mode:
            self._set_wf_flag()

    def _set_wf_flag(self):
        """
                set this flag to distinguish whether picture is from widerface data set
                :return:
                """
        self.wf_flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            self.wf_flag[i] = img_info['is_widerface']


