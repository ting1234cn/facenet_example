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
from datetime import datetime
import random



def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # set GPU id

    #设置GPU Option
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    tf.Session(config=config)  # set GPU option


    with tf.Graph().as_default():

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter('~/logs/facenet_finger', sess.graph)
            facenet.load_model(args.model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


            frr,far,best_threshold=compare(args.template,args.to_be_verified,args.image_size,args.threshold,
                                           images_placeholder,phase_train_placeholder,embeddings,sess,summary_writer)



    model_exp = os.path.expanduser(args.model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = facenet.get_model_filenames(model_exp)
    # if frr>0.1 and os.path.isdir(model_exp) :
    #     os.remove(os.path.join(model_exp, ckpt_file + '.data-00000-of-00001'))
    #     os.remove(os.path.join(model_exp, ckpt_file + '.index'))
    with open('validation_result.txt','at') as f:
        f.write('model %s\t%s\t threshold %.5f\t frr %.5f\t far %.5f\n' % (meta_file,ckpt_file,best_threshold, frr, far))

def compare_new(template_dir, target_dir,image_size,threshold,images_placeholder,
            phase_train_placeholder,embeddings,sess,summary_writer):
    start_time=time.time()
    dist_list = []

    k_threshold = 1.0
    best_threshold = 0.25
    max_threshold = best_threshold + 0.6
    min_threshold=0
    target_image_path = []

    false_reject_path=[]
    false_accept_path=[]
    ############获取target的embedding
    dataset = facenet.get_dataset(target_dir)
    for path in dataset[:]:
        target_image_path.extend(path.image_paths)
    num_target_class = len(dataset)
    class_index = 0
    target_emb = []
    while class_index < num_target_class:
        target_image_list = load_and_align_data(dataset[class_index].image_paths, image_size)


        target_images = get_images(target_image_list)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: target_images, phase_train_placeholder: False}

        #print("start run model", time.time())  # run model start timestamp
        emb = sess.run(embeddings, feed_dict=feed_dict)
        #print("target emb",[emb,dataset[class_index].name])
        target_emb.append([emb,dataset[class_index].image_paths])

        #print("run model finished", time.time())
        class_index+=1

    ############获取模板的embedding
    # template_classes = [path for path in os.listdir(template_dir) \
    #                 if os.path.isdir(os.path.join(template_dir, path))]
    # template_classes.sort()

    template = facenet.get_dataset(template_dir)
    #random.shuffle(template)
    template_emb = []
    for temp in template:
        #print("template image path",temp.image_paths)
        aligned = load_and_align_data(temp.image_paths, image_size)
        template_images = get_images(aligned)
        #num_template = len(template_images)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: template_images, phase_train_placeholder: False}
        #print("start run model", time.time())  # run model start timestamp
        emb = sess.run(embeddings, feed_dict=feed_dict)
        template_emb.append([emb,temp.name])

        #print("run model finished", time.time())
        #######################################
    pair_list=[]
    finger_list=[]

    for target in target_emb:
        #print("target ",target)
        for temp in template_emb:

            for target_index,emb in enumerate(target[0][:]):
                dist_list.clear()
                for temp_emb in temp[0][:]:
                    dist = np.sqrt(np.sum(np.square(np.subtract(temp_emb, emb))))
                    dist_list.append(dist)
                #dist_list.sort()
                #best_dist=np.average(dist_list[:3])
                best_dist=np.min(dist_list)

                finger_id_index = target[1][0].rfind('U')
                finger_id = target[1][0][finger_id_index:finger_id_index + 10]
                #print("finger_id",finger_id)
                if finger_id in temp[1]:
                    issame=True
                else:
                    issame=False
                pair_list.append([best_dist,issame])
                finger_list.append([target[1][target_index],temp[1]])
                


    frr=far=0.0
    #print("pair list",pair_list)
    for threshold in np.arange(max_threshold,min_threshold, -0.02):
        frr,far=evaluate_new(threshold,pair_list)
        if far<=0.01 or threshold==min_threshold:
            k_frr, k_far= evaluate_new(threshold+0.01, pair_list)
            if k_far<=0.01:
                k_threshold=threshold+0.01
                frr=k_frr
                far=k_far
            else:
                k_threshold = threshold
            #k_threshold = threshold
            break
    print("%s,frr:%f,far:%f,best_threshold:%f"%(datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'),frr,far,k_threshold))
    _,_=evaluate_new(k_threshold,pair_list,finger_list,get_frr_path=True)



    end_time = time.time()
    duration=end_time-start_time
    print("validation duration",duration)
    return frr,far,k_threshold

# def evaluate_threshold(max_threshold,min_threshold):
#     threshold=np.mean(max_threshold,min_threshold)
#     frr,far=evaluate_new(threshold,pair_list)
#     if far<0.01:
def evaluate_new(threshold,pair_list,finger_list=None,get_frr_path=False):

    print("threshold",threshold)
    pair_list=np.array(pair_list,ndmin=2)
    # print("Pair list shape",pair_list.shape)
    # print("pair list 0", pair_list[:,0])
    # print("pair list 1", pair_list[:,1])
    thresholds=np.ones((pair_list.shape[0]))
    thresholds=thresholds*threshold

    predict_issame=np.asarray(np.less(pair_list[:,0],thresholds))
    actual_issame=np.asarray(pair_list[:,1])
#    print("predict same",predict_issame)
#    print("predict same non zero",np.count_nonzero(predict_issame))

    if get_frr_path==True:
        far_index= np.asarray(np.logical_and(predict_issame,np.logical_not(actual_issame)))
        i = 0
        first_flag = True
        while i <len(far_index):
            if far_index[i] == True:
                if first_flag:
                    print("total %d far finger id %s",(np.sum(far_index),finger_list[i]))
                    first_flag=False
                else:
                    print("far finger id", finger_list[i])
            i += 1



    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    #print("false reject",np.logical_and(np.logical_not(predict_issame), actual_issame))

    # tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    # fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    #acc = float(tp+tn)/dist.size

    frr= 0 if(fn==0) else float(fn)/float(np.sum(actual_issame))
    far= 0 if(fp==0) else float(fp)/float(np.sum(np.logical_not(actual_issame)))
    # print("actual same",np.sum(actual_issame))
    # print("predict same",np.sum(predict_issame))
    # print("fn:%d,fp:%d"%(fn,fp))
    #print("frr: %f, far:%f"%(frr,far))
    return frr, far



def compare(template_dir, target_dir,image_size,threshold,images_placeholder,
            phase_train_placeholder,embeddings,sess,summary_writer):
    dist_list = []

    k_frr = k_far = 0
    k_threshold = 1
    best_threshold = 0.25
    max_threshold = best_threshold + 0.3
    min_threshold=best_threshold - 0.15
    target_image_path = []
    false_reject=0
    false_accept=0
    false_reject_path=[]
    false_accept_path=[]
    issame_true=0
    issame_false=0
    template_exp = os.path.expanduser(template_dir)


    dataset = facenet.get_dataset(target_dir)
    for path in dataset[:]:
        target_image_path.extend(path.image_paths)
    num_target_class = len(dataset)
    class_index = 0
    target_emb = []
    ############获取模板的embedding
    template = facenet.get_dataset(template_dir)
    # num_template = len(template[0].image_paths)
    random.shuffle(template)
    for temp in template:
        #print("template image path",temp.image_paths)
        aligned = load_and_align_data(temp.image_paths, image_size)
        template_images = get_images(aligned)
        #num_template = len(template_images)
        template_emb = []
        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: template_images, phase_train_placeholder: False}
        print("start run model", time.time())  # run model start timestamp
        template_emb = sess.run(embeddings, feed_dict=feed_dict)
        print("run model finished", time.time())
        #######################################


        while class_index < num_target_class:
            target_image_list = load_and_align_data(dataset[class_index].image_paths, image_size)

            num_to_be_verified = len(dataset[class_index].image_paths)
            target_images = get_images(target_image_list)

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: target_images, phase_train_placeholder: False}

            print("start run model", time.time())  # run model start timestamp
            emb = sess.run(embeddings, feed_dict=feed_dict)
            target_emb.extend(emb)
            # np.concatenate((target_emb,emb))

            print("run model finished", time.time())

            # for i in range(num_to_be_verified):
            #     best_dist = []  # 设置初始值空
            #     finger_id_index = dataset[class_index].image_paths[i].rfind('U')
            #     finger_id = dataset[class_index].image_paths[i][finger_id_index:finger_id_index + 10]
            #     issame = False
            #     for path in temp.image_paths:
            #         if path.find(finger_id) == -1:
            #             issame = False
            #         else:
            #             issame = True
            #             break
            #
            #     for j in range(num_template):
            #         dist = np.sqrt(np.sum(np.square(np.subtract(template_emb[j, :], emb[i, :]))))
            #         best_dist.append(dist)
            #     #average_dist = min(best_dist)
            #     #best_dist.sort()
            #     average_dist = np.min(best_dist)
            #     if average_dist < best_threshold and issame == False:
            #         best_threshold = average_dist
            #
            #     # if average_dist < threshold:
            #     #     dist_list.append((dataset[class_index].image_paths[i], "match", average_dist))
            #     #     # print(
            #     #     #     '匹配 %s  %d 相似度 %1.4f ' % (
            #     #     #     dataset[class_index].image_paths[i], best_dist.index(average_dist), average_dist))
            #     # else:
            #     #     dist_list.append((dataset[class_index].image_paths[i], "no_match", average_dist))
            #     #     # print('不匹配 %s  %d 相似度 %1.4f ' % (
            #     #     # dataset[class_index].image_paths[i], best_dist.index(average_dist), average_dist))
            class_index += 1

        if k_threshold==1:
            for threshold in np.arange(min_threshold, max_threshold, 0.02):
                k_false_reject,k_false_accept,k_issame_true,k_issame_false,k_false_reject_path,k_false_accept_path=evaluate(threshold, template_emb, temp.image_paths, np.asarray(target_emb), target_image_path)
                k_frr,k_far=calculate_frr_far(k_false_reject,k_false_accept,k_issame_true,k_issame_false)
                if k_far>0.01:
                    k_threshold=threshold
                    break
        else:
            for threshold in np.arange(min_threshold, max_threshold, 0.02):
                k_false_reject,k_false_accept,k_issame_true,k_issame_false,k_false_reject_path,k_false_accept_path=evaluate(threshold, template_emb, temp.image_paths, np.asarray(target_emb), target_image_path)
                k_frr,k_far=calculate_frr_far(k_false_reject,k_false_accept,k_issame_true,k_issame_false)
                if k_far>0.01:
                    if threshold<k_threshold:
                        k_threshold=threshold
                    break


        if k_threshold<1:
            t_false_reject,t_false_accept,t_issame_true,t_issame_false,t_false_reject_path,t_false_accept_path=\
                evaluate(k_threshold, template_emb, temp.image_paths, np.asarray(target_emb), target_image_path)
        else:
            t_false_reject, t_false_accept, t_issame_true, t_issame_false,t_false_reject_path,t_false_accept_path =\
                evaluate(max_threshold, template_emb, temp.image_paths, np.asarray(target_emb),target_image_path)
        false_reject+=t_false_reject
        false_accept+=t_false_accept
        issame_true+=t_issame_true
        issame_false+=t_issame_false
        false_accept_path.extend(t_false_accept_path)
        false_reject_path.extend(t_false_reject_path)



        print("k threshold %f with 0 FAR" % k_threshold)
        print("false reject: %d,false accept:%d,issame_true:%d; issame_false:%d"%(false_reject, false_accept,issame_true,issame_false))

    # save false reject/accept image into file
    # if (os.path.isdir(template_exp)):
    #     np.savetxt(template_exp + "false_reject.csv", false_reject_path, fmt=['%s', '%f'])
    #     np.savetxt(template_exp + "false_accept.csv", false_accept_path, fmt=['%s', '%f'])
    #        np.save(template_exp + ".npy", template_emb[:])
    frr,far=calculate_frr_far(false_reject, false_accept,issame_true,issame_false)
    print("frr: %f, far: %f"%(frr,far))
    return frr, far ,k_threshold


def evaluate(threshold,template_emb, template_paths,target_emb,target_paths):
    issame=[]

    false_reject=0
    true_accept=0
    false_accept=0
    true_reject=0
    i=0
    false_reject_path=[]
    false_accept_path=[]

    print("target emb len",len(target_emb))
    for target_path in target_paths:

        dist_list=[]
        finger_id_index=target_path.rfind('U')
        finger_id=target_path[finger_id_index:finger_id_index+10]
        j = 0
        for template_path in template_paths:
            #print("template path",template_path)

            dist = np.sqrt(np.sum(np.square(np.subtract(template_emb[j,:], target_emb[i,:]))))
            dist_list.append(dist)
            j += 1
        #best_dist=min(dist_list)
        best_dist = np.min(dist_list)
        if template_paths[0].find(finger_id)==-1:
            issame.append(False)
            if best_dist<threshold:
                false_accept+=1
                false_accept_path.append([target_path,best_dist])
            else:
                true_reject+=1
        else:
            issame.append(True)
            if best_dist>=threshold:
                false_reject+=1
                false_reject_path.append([target_path,best_dist])
            else:
                true_accept+=1

        i+=1
    # if false_reject > 0:
    #     frr=float(false_reject/issame.count(True))
    # else:
    #     frr=0
    # if false_accept > 0:
    #     far=float(false_accept/issame.count(False))
    # else:
    #     far=0
    #print("threshold %f ; FRR %f  ; FAR %f; num of same finger %d; num of different finger %d" %(threshold, frr, far,issame.count(True), issame.count(False)))
    return false_reject, false_accept, issame.count(True), issame.count(False), false_reject_path, false_accept_path



def get_images(aligned):
    img_list = []
    for i in aligned:
        prewhitened = facenet.prewhiten(i)
        #prewhitened =i
        img_list.append(prewhitened[:, :, np.newaxis])  # 增加一维，满足灰度model 160x160x1的输入维度要求，如果是RGB model则不需要
    images = np.stack(img_list)
    return images

def calculate_frr_far(false_reject,false_accept,issame_true,issame_false):
    if false_reject > 0:
        frr=float(false_reject/issame_true)
    else:
        frr=0
    if false_accept > 0:
        far=float(false_accept/issame_false)
    else:
        far=0
    return frr,far





# image_paths
# image_size
# margin
# gpu_memory_fraction
def load_and_align_data_template(image_paths, image_size):


    tmp_image_paths = image_paths.copy()
    img_list = []

    for image in tmp_image_paths:
        print("image file",image)
        img = misc.imread(os.path.expanduser(image), mode='L')
        print("image shape",img.shape)
        img_size = np.asarray(img.shape)[0:2]
        left=top=0
        while img_size[1]>=top+image_size :
            while img_size[0]>=left+image_size:
                # print("img size",img_size[0])
                #print("left+image_size:, top+image_size: ",(left+image_size,top+image_size))
                #aligned=tf.image.crop_to_bounding_box(img,left,top,image_size,image_size)
                aligned=img[left:left+image_size,top:top+image_size]


                img_list.append(aligned)
                left+=5
            top+=5

    #print("number of image:",len(img_list))
    return img_list


def load_and_align_data(image_paths, image_size):


    tmp_image_paths = image_paths.copy()
    img_list = []

    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='L')
        img_size = np.asarray(img.shape)[0:2]


        aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')

        img_list.append(aligned)


    return img_list

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
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.5)
    parser.add_argument('--gpu', type=str,
                        help='gpu id .',
                        default='0')
    parser.add_argument('--threshold', type=float,
                        help='threshold to identify fingerprint',default=0.8)
    return parser.parse_args(argv)


if __name__ == '__main__':
    # 这是一个从外部输入参数的代码。
    main(parse_arguments(sys.argv[1:]))
