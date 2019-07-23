
#用来计算两张人脸图像之间的距离矩阵。需要输入的参数：
#预训练模型 图片1  图片220170512-110547 1.png 2.png

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
import align.detect_face
# import detect_face
#from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main(args):

    aligned,nrof_face_list= load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    img_list=[]
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
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            
            nrof_images = len(args.image_files)



            loc_face=nrof_face_list[0]
            for i in range(nrof_face_list[0]):
                print('%1d  ' % i, end='')
                for j in range(1,nrof_images):
                    min_dist=1
                    min_face=0
                    for k in range(nrof_face_list[j]):
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[loc_face,:]))))

                        print('%s  face %i 相似度 %1.4f ' % (args.image_files[j] , k , dist))
                        if dist<min_dist:
                            min_dist=dist
                            min_face=loc_face
                        loc_face += 1


                    if min_dist>=0.8:
                        print("no similar face")
                        pass
                    else:
                        plt.figure("最相似的人脸")
                        plt.subplot(121)
                        plt.imshow(aligned[i])
                        plt.subplot(122)
                        plt.imshow(aligned[min_face])
                        #plt.text(0,0,"相似度高",fontproperties="SimHei",color="r")
                        plt.show()




                print('')


# image_paths
# image_size
# margin
# gpu_memory_fraction
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            # pnet, rnet, net = detect_face.create_mtcnn(sess, None)

    tmp_image_paths = image_paths.copy()
    img_list = []
    nrof_face_list=[]
    for image in tmp_image_paths:
        img = misc.imread(image, mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        # bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        if len(bounding_boxes) < 1:
           image_paths.remove(image)
           print("can't detect face, remove ", image)
           continue
        nrof_faces = bounding_boxes.shape[0]  # 人脸数目
        nrof_face_list.append(nrof_faces)
        print('{}找到人脸数目为：{}'.format(image,nrof_faces))
        for i in range(nrof_faces):
            det = np.squeeze(bounding_boxes[i,0:4])  #去掉了最后一个元素？
            bb = np.zeros(4, dtype=np.int32)
            ##np.maximum：(X, Y, out=None) #X 与 Y 逐位比较取其大者；相当于矩阵个元素比较
            bb[0] = np.maximum(det[0]-margin/2, 0)#margin：人脸的宽和高？默认为44
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')

            img_list.append(aligned)

            plt.figure(image)
            plt.imshow(img)
            #框出人脸
            currentAxis = plt.gca()
            rect = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], linewidth=1, edgecolor='r', facecolor='none')
            currentAxis.add_patch(rect)
        plt.show()
    return img_list,nrof_face_list
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
    #这是一个从外部输入参数的代码。
    main(parse_arguments(sys.argv[1:]))
