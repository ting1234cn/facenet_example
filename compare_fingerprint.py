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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # set GPU id
    dist_list = []
    template_emb=[]
    best_threshold = 1
    target_image_path=[]

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
    for path in dataset[:]:
        target_image_path.extend(path.image_paths)
    num_target_class=len(dataset)
    class_index=0

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
            target_emb=[]
            while class_index <num_target_class:
                target_image_list=load_and_align_data(dataset[class_index].image_paths, args.image_size, args.margin,
                                                          args.gpu_memory_fraction)

                num_to_be_verified=len(dataset[class_index].image_paths)
                target_images=get_images(target_image_list)



                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: target_images, phase_train_placeholder: False}


                print("start run model",time.time())#run model start timestamp
                emb=sess.run(embeddings, feed_dict=feed_dict)
                target_emb.extend(emb)
                #np.concatenate((target_emb,emb))

                print("run model finished", time.time())

                for i in range(num_to_be_verified):
                    best_dist = []  # 设置初始值空
                    finger_id_index = dataset[class_index].image_paths[i].rfind('U')
                    finger_id = dataset[class_index].image_paths[i][finger_id_index:finger_id_index + 10]
                    issame=False
                    for path in template[0].image_paths:
                        if path.find(finger_id)==-1:
                            issame = False
                        else:
                            issame= True
                            break


                    for j in range(num_template):
                        dist = np.sqrt(np.sum(np.square(np.subtract(template_emb[j,:], emb[i,:]))))
                        best_dist.append(dist)
                    average_dist = min(best_dist)
                    if average_dist<best_threshold and issame==False:
                        best_threshold=average_dist

                    if average_dist < args.threshold:
                        dist_list.append((dataset[class_index].image_paths[i], "match", average_dist))
                        print(
                            '匹配 %s  %d 相似度 %1.4f ' % (
                            dataset[class_index].image_paths[i], best_dist.index(average_dist), average_dist))
                    else:
                        dist_list.append((dataset[class_index].image_paths[i], "no_match", average_dist))
                        print('不匹配 %s  %d 相似度 %1.4f ' % (
                        dataset[class_index].image_paths[i], best_dist.index(average_dist), average_dist))
                class_index += 1
            for threshold in np.arange(0,2,0.05):
                evaluate(threshold,template_emb,template[0].image_paths,np.asarray(target_emb),target_image_path)
            frr,far=evaluate(best_threshold, template_emb, template[0].image_paths, np.asarray(target_emb), target_image_path)
            if (os.path.isdir(template_exp)):
                np.save(template_exp + ".npy", template_emb[:])
            print("best threshold %f with 0 FAR" % best_threshold)

    np.savetxt("data.csv", np.asarray(dist_list), fmt="%s", delimiter="\t")
    model_exp = os.path.expanduser(args.model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = facenet.get_model_filenames(model_exp)
    with open('validation_result.txt','at') as f:
        f.write('model %s\t%s\t threshold %.5f\t frr %.5f\t far %.5f\n' % (meta_file,ckpt_file,best_threshold, frr, far))

def evaluate(threshold,template_emb, template_paths,target_emb,target_paths):
    issame=[]

    false_reject=0
    true_accept=0
    false_accept=0
    true_reject=0
    i=0
    for target_path in target_paths:

        dist_list=[]
        finger_id_index=target_path.rfind('U')
        finger_id=target_path[finger_id_index:finger_id_index+10]
        j = 0
        for template_path in template_paths:
            dist = np.sqrt(np.sum(np.square(np.subtract(template_emb[j,:], target_emb[i,:]))))
            dist_list.append(dist)
            j += 1
        best_dist=min(dist_list)
        if template_paths[0].find(finger_id)==-1:
            issame.append(False)
            if best_dist<=threshold:
                false_accept+=1
            else:
                true_reject+=1
        else:
            issame.append(True)
            if best_dist>threshold:
                false_reject+=1
            else:
                true_accept+=1

        i+=1
    if false_reject > 0:
        frr=float(false_reject/issame.count(True))
    else:
        frr=0
    if false_accept > 0:
        far=float(false_accept/issame.count(False))
    else:
        far=0
    print("threshold %f ; FRR %f  ; FAR %f" %(threshold,frr,far))
    return frr, far



def get_images(aligned):
    img_list = []
    for i in aligned:
        prewhitened = facenet.prewhiten(i)
        #prewhitened =i
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
    parser.add_argument('--gpu', type=str,
                        help='gpu id .',
                        default='0')
    parser.add_argument('--threshold', type=float,
                        help='threshold to identify fingerprint',default=0.8)
    return parser.parse_args(argv)


if __name__ == '__main__':
    # 这是一个从外部输入参数的代码。
    main(parse_arguments(sys.argv[1:]))
