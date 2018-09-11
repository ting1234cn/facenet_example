import sys
import os,shutil
import argparse
from PIL import Image


def main(args):
    if os.path.isdir(args.input_dir):
        path_exp = os.path.expanduser(args.input_dir)
        get_image_path(path_exp)



def rename_image(file_name,type="jpeg"):
    im=Image.open(file_name)
    if type is "jpeg":
        file_name=file_name.replace('.bmp','')
        if file_name.find("dry")!=-1:
            new_file_name=file_name+"_dry.jpg"

        elif file_name.find("light")!=-1:
            new_file_name = file_name + "_light.jpg"
        else:
            new_file_name = file_name + ".jpg"
        out = im.resize((180, 180), Image.ANTIALIAS)
        out.save(new_file_name,'JPEG',quality=95)
        im.close()
        return new_file_name

def get_image_path(input_dir):

    sub_dirs = [path for path in os.listdir(input_dir) \
                if os.path.isdir(os.path.join(input_dir, path))]
    for sub_dir in sub_dirs:
        get_image_path(os.path.join(input_dir, sub_dir))
    img_paths = [os.path.join(input_dir, path) for path in os.listdir(input_dir) \
                if os.path.isfile(os.path.join(input_dir, path))]
    filenames=[ path for path in os.listdir(input_dir) \
                if os.path.isfile(os.path.join(input_dir, path))]
    new_dirs=[]
    for filename in filenames:
        filename=filename[:10]
        new_dir=os.path.join(input_dir, filename)
        if new_dir in new_dirs:
            pass
        else:
            new_dirs.append(new_dir)
            os.mkdir(new_dir)

    for path in img_paths:
        new_file_name=rename_image(path)
        for new_dir in new_dirs:
            if path.find(new_dir)!=-1:

                shutil.move(new_file_name,new_dir)
                os.remove(path)
                print("original file %s destination file %s" %(new_file_name, new_dir))
        #os.remove(path)









def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str,
                        help='input dir which include input image files')
    parser.add_argument('--output_dir', type=str,
                        help='output file dir.')
    parser.add_argument('--image_type', type=str,
                        help='output image type', default='jpeg')

    return parser.parse_args(argv)


if __name__ == '__main__':
    # 这是一个从外部输入参数的代码。
    main(parse_arguments(sys.argv[1:]))