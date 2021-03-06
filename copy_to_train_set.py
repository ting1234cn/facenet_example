import sys
import os,shutil
import argparse
import cv2 as cv
import numpy as np


def main(args):
    if os.path.isdir(args.input_dir) and os.path.isdir(args.output_dir):
        input_path_exp = os.path.expanduser(args.input_dir)
        output_path_exp=os.path.expanduser(args.output_dir)
        copy_image(input_path_exp,output_path_exp)


def rename_file(input_dir):
    filenames = [path for path in os.listdir(input_dir) \
                 if os.path.isfile(os.path.join(input_dir, path))]
    for filename in filenames:
        old_name=os.path.join(input_dir,filename)
        new_name=os.path.join(input_dir,filename)
        new_name=new_name.replace('Pre','')
        new_name=new_name.replace('.jpg','Pre.jpg')
        os.rename(old_name, new_name)


def copy_image(input_dir,output_dir):

    sub_dirs = [os.path.join(input_dir, path) for path in os.listdir(input_dir) \
                if os.path.isdir(os.path.join(input_dir, path))]
    for dir in sub_dirs:
        if dir.find('bmp')==-1:
            copy_image(dir,output_dir)
        else:
            #rename_file(dir)
            copy_image(dir, output_dir)
            pass

    output_sub_dirs=[os.path.join(output_dir, path) for path in os.listdir(output_dir) \
                if os.path.isdir(os.path.join(output_dir, path))]

    img_paths = [os.path.join(input_dir, path) for path in os.listdir(input_dir) \
                if os.path.isfile(os.path.join(input_dir, path))]
    filenames=[ path for path in os.listdir(input_dir) \
                if os.path.isfile(os.path.join(input_dir, path))]

    noise_template = np.zeros((180,180))
    img_num = 0
    for filename in filenames:
        img_num+=1
        for output_sub_dir in output_sub_dirs:

            if output_sub_dir.find(filename[:10])!=-1:
                finger = cv.imread(os.path.join(input_dir, filename))  # 字符串加r,避免转义
                # noise_template=(noise_template+255-finger[:,:,0])/2
                # if img_num>10:
                #     #noise_template=np.mean(noise_template)
                #     finger[:,:,0]=finger[:,:,0]-noise_template


                #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # 限制对比度的自适应阈值均衡化
                #dst = clahe.apply(finger[:, :, 0])
                dst = cv.equalizeHist(finger[:,:,0])
                cv.imwrite(os.path.join(output_sub_dir, filename),dst)

                #shutil.copy(os.path.join(input_dir, filename), output_sub_dir)
                print("copy %s to dir %s" %(filename,output_sub_dir))








def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str,
                        help='input dir which include input image files')
    parser.add_argument('output_dir', type=str,
                        help='output file dir.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    # 这是一个从外部输入参数的代码。
    main(parse_arguments(sys.argv[1:]))