import pandas as pd
from PIL import Image
import numpy as np
import mmcv
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm

# import os
# os.chdir('..')

DATA_ROOT = '../data/icartoonface'

# extra dataset: WiderFace
WiderFace_ROOT = '../data/WiderFace'



def iter_bboxes(pd_annotations):
    for _, row in pd_annotations.iterrows():
        x1, y1, x2, y2 =  row.values[1:].astype(np.float32)
        yield x1, y1, x2, y2


def prepare_train():
    pd_annotations = pd.read_csv(f'{DATA_ROOT}/personai_icartoonface_dettrain_anno_updatedv1.0/personai_icartoonface_dettrain_anno_updatedv1.0.csv',
                                 names=['image', 'xmin', 'ymin', 'xmax', 'ymax'])


    # add images to COO
    pd_annotations  =pd_annotations.groupby(['image'])
    images = []
    for img_name, pd_group in tqdm(pd_annotations):
        img_file =f'{DATA_ROOT}/personai_icartoonface_dettrain/icartoonface_dettrain/{img_name}'
        img= Image.open(img_file)

        image= {
            # 'filename': img_name,
            'filename': img_file[8:],
            'width': img.width,
            'height': img.height,

            # indicator for widerface data
            'is_widerface': 0
        }

        bboxes = []
        labels = []
        for x1, y1, x2, y2 in iter_bboxes(pd_group):

            # check bbox correctable
            if x1 >= x2 or y1 >= y2:
                # print(img_file)
                continue

            bboxes.append([x1, y1, x2, y2])
            labels.append(0)
        image['ann'] = {
            'bboxes': np.array(bboxes).astype(np.float32).reshape(-1,4 ),
            'labels': np.array(labels).astype(np.int64).reshape(-1),
            'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
            'labels_ignore': np.array([], dtype=np.int64).reshape(-1),
        }
        # if len(image['ann']['bboxes']) > 100:
        #     print(len(image['ann']['bboxes']))
        images.append(image)


    # add widerface part
    with open(f'{WiderFace_ROOT}/wider_face_split/wider_face_train_bbx_gt.txt', 'r') as f:
        draw = f.read().splitlines()



    # re-organize structure
    line= 0
    extra_images = []
    while(line < len(draw)):
        img_name = draw[line]
        line += 1
        img_file = f'{WiderFace_ROOT}/WIDER_train/images/' + img_name
        img = Image.open(img_file)

        image = {
            'filename': img_file[8:],
            'width': img.width,
            'height': img.height,
            'is_widerface': 1
        }

        bboxes = []
        labels = []
        n_bbox = int(draw[line])
        # if n_bbox > 1000:
        #     print(n_bbox)
        line += 1
        for i in range(0, n_bbox):
            x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = \
                map(int, draw[line].split())
            line += 1
            if invalid:
                continue
            if w == 0 or h ==0:
                # print(img_file)
                continue

            bboxes.append([x1, y1, x1 + w, y1 + h])
            labels.append(0)
        if n_bbox == 0:
            line += 1 # pass row 0,0,0,0,0,0,0,0,0,0
        if n_bbox > 10: # pass crow scenario
            continue

        image['ann'] = {
            'bboxes': np.array(bboxes).astype(np.float32).reshape(-1, 4),
            'labels': np.array(labels).astype(np.int64).reshape(-1),
            'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
            'labels_ignore': np.array([], dtype=np.int64).reshape(-1),
        }
        extra_images.append(image)

    train, val = train_test_split(images, test_size=0.05, random_state=1111)

    mmcv.dump(train, f'{DATA_ROOT}/dtrain.pkl')
    print('extra images', len(extra_images))
    train += extra_images
    mmcv.dump(train, f'{DATA_ROOT}/dtrain_wf.pkl')
    mmcv.dump(val, f'{DATA_ROOT}/dval.pkl')

    mmcv.dump(images, f'{DATA_ROOT}/dtrainval.pkl')
    images += extra_images
    mmcv.dump(images, f'{DATA_ROOT}/dtrainval_wf.pkl')


def prepare_test():
    images =[]
    for img_file in tqdm(glob(f'{DATA_ROOT}/personai_icartoonface_detval/*.jpg')):
        img = Image.open(img_file)

        images.append({
            # 'filename': img_file.split('/')[-1],
            'filename': img_file[8:],
            'width' : img.width,
            'height': img.height,
            'ann':{
                'bboxes': np.array([], dtype=np.float32).reshape(-1, 4),
                'labels': np.array([], dtype=np.int64).reshape(-1),
                'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array([],dtype=np.int64).reshape(-1)
            }
        })

    mmcv.dump(images, f'{DATA_ROOT}/dtest.pkl')


if __name__  =='__main__':
    prepare_train()
    prepare_test()



