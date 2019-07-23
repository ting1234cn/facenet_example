##被调用的函数包。使用到的函数：validate_on+lfw.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet
from scipy import misc
import random


def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
                                               np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
                                              np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 1))# 把3 改成1， 灰度图
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i],mode='L')
        img = misc.imresize(img, (image_size, image_size), interp='bilinear')

        #注释掉为了处理灰度图
        # if img.ndim == 2:
        #     img = to_rgb(img)
        if do_prewhiten:
            img = facenet.prewhiten(img)
        # 注释掉为了处理灰度图
        # img = crop(img, do_random_crop, image_size)
        # img = flip(img, do_random_flip)

        images[i,:,:,:] = img[:,:,np.newaxis]

    return images

def get_paths(template_dir, validate_dir):
    nrof_skipped_pairs = 0
    images_per_person=50
    path_list = []
    issame_list = []
    template_img_paths=[]
    validate_img_paths=[]
    sub_dirs = [os.path.join(template_dir, path) for path in os.listdir(template_dir) \
                if os.path.isdir(os.path.join(template_dir, path))]
    for sub_dir in sub_dirs:
        template_img_paths += [os.path.join(sub_dir, path) for path in os.listdir(sub_dir) \
                              if os.path.isfile(os.path.join(sub_dir, path))]

    sub_dirs = [os.path.join(validate_dir, path) for path in os.listdir(validate_dir) \
                if os.path.isdir(os.path.join(validate_dir, path))]

    for sub_dir in sub_dirs:
        img_path=[os.path.join(sub_dir, path) for path in os.listdir(sub_dir) \
                                                   if os.path.isfile(os.path.join(sub_dir, path))]
        random.shuffle(img_path)
        validate_img_paths += img_path[:min(images_per_person,len(img_path))]




    for template_filename in template_img_paths:
        UID_index=template_filename.rfind('U')
        UID=template_filename[UID_index:UID_index+10]
        for validate_filename in validate_img_paths:
            if str.find(validate_filename,UID)!=-1:
                issame=True
                issame_list.append(issame)
            else:
                issame=False
                issame_list.append(issame)
            path_list += (template_filename, validate_filename)


    return path_list, issame_list
"""       
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
"""



