from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import time


def main(args):
    dist_list = []
    aligned = []
    template_emb=[]

    #设置GPU Option
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)  # set GPU option

    template_exp = os.path.expanduser(args.template)
    if (os.path.isfile(template_exp)):
        template_emb=np.load(template_exp)
        num_template = len(template_emb)

    else:
        template = facenet.get_dataset(args.template)
        num_template=len(template[0].image_paths)
        aligned = load_and_align_data(template[0].image_paths, args.image_size, args.margin,
                                                  args.gpu_memory_fraction)
        template_images=get_images(aligned)

    dataset=facenet.get_dataset(args.to_be_verified)
    num_target_class=len(dataset)
    class_index=0
    target_image_index=[]
    with tf.Graph().as_default():

        with tf.Session() as sess:
            facenet.load_model(args.model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            if len(template_emb)==0:
                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: template_images, phase_train_placeholder: False}
                print("start run model", time.time())  # run model start timestamp
                template_emb = sess.run(embeddings, feed_dict=feed_dict)
                print("run model finished", time.time())

            while class_index <num_target_class:
                target_image_list=load_and_align_data(dataset[class_index].image_paths, args.image_size, args.margin,
                                                          args.gpu_memory_fraction)
                target_image_index.append(len(target_image_list))
                aligned=aligned[:num_template]
                aligned.extend(target_image_list)

                num_to_be_verified=len(dataset[class_index].image_paths)
                target_images=get_images(aligned)



                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: target_images, phase_train_placeholder: False}


                print("start run model",time.time())#run model start timestamp
                emb=sess.run(embeddings, feed_dict=feed_dict)

                emb=np.vstack((template_emb,emb))#把templdate emb 和target emb 组合成一个数组，这样不用修改以前的比较算法

                print("run model finished", time.time())

                for i in range(num_to_be_verified):
                    best_dist = []  # 设置初始值空
                    for j in range(num_template):
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb[j, :], emb[i + num_template, :]))))
                        best_dist.append(dist)
                    average_dist = min(best_dist)
                    if average_dist < 0.544:
                        dist_list.append((dataset[class_index].image_paths[i], "match", average_dist))
                        print(
                            '匹配 %s  %d 相似度 %1.4f ' % (
                            dataset[class_index].image_paths[i], best_dist.index(average_dist), average_dist))
                    else:
                        dist_list.append((dataset[class_index].image_paths[i], "no_match", average_dist))
                        print('不匹配 %s  %d 相似度 %1.4f ' % (
                        dataset[class_index].image_paths[i], best_dist.index(average_dist), average_dist))

                if (os.path.isdir(template_exp)):
                    np.save(template_exp + "template.npy", template_emb[:])

                class_index+=1

    np.savetxt("data.csv", np.asarray(dist_list), fmt="%s", delimiter="\t")


def get_images(aligned):
    img_list = []
    for i in aligned:
        prewhitened = facenet.prewhiten(i)
        # prewhitened =i
        img_list.append(prewhitened[:, :, np.newaxis])  # 增加一维，满足灰度model 160x160x1的输入维度要求，如果是RGB model则不需要
    images = np.stack(img_list)
    return images







# image_paths
# image_size
# margin
# gpu_memory_fraction
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):


    tmp_image_paths = image_paths.copy()
    img_list = []

    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='L')
        img_size = np.asarray(img.shape)[0:2]



        aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')

        img_list.append(aligned)


    return img_list
    # return img


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('template', type=str,
                        help='template data file or Path to the data directory containing template fingerprint.')
    parser.add_argument('to_be_verified', type=str,
                        help=' Path to the data directory containing to be verified fingerprint.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.2)
    return parser.parse_args(argv)


if __name__ == '__main__':
    # 这是一个从外部输入参数的代码。
    main(parse_arguments(sys.argv[1:]))
