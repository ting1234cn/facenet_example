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
import matplotlib.pyplot as plt
import time


def main(args):
    aligned = load_and_align_data(args.image_files, args.image_size, args.margin,
                                                  args.gpu_memory_fraction)
    img_list = []
    for i in aligned:
        prewhitened = facenet.prewhiten(i)
        img_list.append(prewhitened)
    images = np.stack(img_list)

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}


            print("start run model",time.time())#run model start timestamp
            emb = sess.run(embeddings, feed_dict=feed_dict)
            #run model finished time
            print("run model finished", time.time())

            nrof_images = len(args.image_files)

            plt.subplot(1,nrof_images,1)
            plt.imshow(aligned[0])
            plt.title("模板指纹" + args.image_files[0], fontproperties="SimHei", color="r")
            for i in range(1,nrof_images):
                dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[i, :]))))

                print('%s   相似度 %1.4f ' % (args.image_files[i], dist))


                if dist >= 0.5:
                    print("dist %f ,no similar fingerprint" % dist)

                    plt.subplot(1,nrof_images,i+1)
                    plt.imshow(aligned[i])
                    plt.title("不相似"+args.image_files[i], fontproperties="SimHei", color="r")

                else:
                    print(" dist %f" % dist)
                    plt.subplot(1,nrof_images,i+1)
                    plt.imshow(aligned[i])
                    plt.title("相似"+args.image_files[i], fontproperties="SimHei", color="r")

            plt.show()




# image_paths
# image_size
# margin
# gpu_memory_fraction
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):


    tmp_image_paths = image_paths.copy()
    img_list = []

    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]



        aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')

        img_list.append(aligned)


    return img_list
    # return img


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    # 这是一个从外部输入参数的代码。
    main(parse_arguments(sys.argv[1:]))
