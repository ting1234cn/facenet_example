# coding: utf-8
import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def stitch_new(img1,img2,top, bot, left, right,min_match_count=19):
    srcImg = cv.copyMakeBorder(img1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    testImg = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    stitcher = cv.createStitcher(False)
    result = stitcher.stitch((srcImg, testImg))
    if result[1] is None:
        return None
    plt.figure()
    plt.imshow(result[1])

    plt.show()
    return result[1]

def stitch(img1,img2,top, bot, left, right,min_match_count=15):
    srcImg = cv.copyMakeBorder(img1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    testImg = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    #srcImg=img1
    #testImg=img2
   # img1gray = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
   # img2gray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)
    img1gray=srcImg
    img2gray=testImg
    sift = cv.xfeatures2d_SIFT().create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    kp2, des2 = sift.detectAndCompute(img2gray, None)
     # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i] = [1, 0]

    #draw_params = dict(matchColor=(0, 255, 0),singlePointColor=(255, 0, 0), matchesMask=matchesMask,flags=0)
    #img3 = cv.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None, **draw_params)
    #plt.imshow(img3, ), plt.show()
    rows, cols = srcImg.shape[:2]
    MIN_MATCH_COUNT = min_match_count
    MAX_MATCH_COUNT=1300
    if len(good) > MIN_MATCH_COUNT and len(good)<MAX_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2.0)
       # M[:,2]=np.clip(M[:,2],-3.0,3.0)
        if M[2,2]!=1:
            print("M33",M[2,2])
            return None

        warpImg = cv.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
                                     flags=cv.WARP_INVERSE_MAP)
        warp_rows,warp_cols=warpImg.shape[:2]
        print("warp_rows, warp_cols",warp_rows, warp_cols)

        rows=min(rows,warp_rows)
        cols=min(cols,warp_cols)
        for row in range(rows - 1, 0, -1):
            if warpImg[row, :].any():
                top = row
                break
        for row in range(0, rows):
            if warpImg[row, :].any():
                bot = row
                break
        for col in range(0, cols):
            if warpImg[:, col].any():
                left = col
                break
        if bot-top>240 :
            return None
        for col in range(0, cols):
            if srcImg[:, col].any() or warpImg[:, col].any():
                left = col
                break
        for col in range(cols - 1, 0, -1):
            if warpImg[:, col].any():
                right = col
                break
        if right-left>300 or right-left<120:
            return None
        for col in range(cols - 1, 0, -1):
            if srcImg[:, col].any() or warpImg[:, col].any():
                right = col
                break
        for row in range(0, rows):
            if srcImg[row, :].any() or warpImg[row, :].any():
                bot = row
                break
        for row in range(rows - 1, 0, -1):
            if srcImg[row, :].any() or warpImg[row, :].any():
                top = row
                break


        res = np.zeros([rows, cols, 3], np.uint8)
        for row in range(0, rows):
            for col in range(0, cols):
                if not srcImg[row, col].any():
                    res[row, col] = warpImg[row, col]
                elif not warpImg[row, col].any():
                    res[row, col] = srcImg[row, col]
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(srcImg[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

        # opencv is bgr, matplotlib is rgb
        #res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
        #cv.imwrite("result.jpg",res[bot:top,left:right])
        # show the result
       # plt.axis('off')
        plt.figure()
        plt.imshow(warpImg)

        plt.show()
        print("Number matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        return (res[bot:top,left:right],len(good))
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return None
if __name__ == '__main__':
    top, bot, left, right = 90, 90, 90, 90
    cwp=r"C:\work\python_code\Facenet_example\template\U300046F03\\"
    img_names=[os.path.join(cwp,path) for path in os.listdir(cwp) if path.find("B0.jpg")!=-1 or path.find("stitch")!=-1]

    res = None
    img_list=[]
    stitched_imgs = []
    for img_name in img_names:
        img=cv.imread(img_name)
        img_list.append(img)

        print("image file",img_name)
        if res is None:
            res=img
        else:


            stitched_img = stitch(res, img, top, bot, left, right)
            if stitched_img is None:

                pass
            else:
                stitched_imgs.append(stitched_img)
                img_list.pop()
                img_list.pop(0)
                res=None
                print("img list len ", len(img_list))



    loop_num=0
    res=None
    while loop_num<=len(img_list):
        loop_num+=1
        print("number of img %d, loop number %d" %(len(img_list),loop_num))
        for i in range(len(img_list)):
            if i >=len(img_list):
                break
            if res is None:

                res=img_list[i]
            else:
                stitched_img=stitch(res, img_list[i], top, bot, left, right)

                if stitched_img is None:
                    pass
                else:
                    res=None
                    stitched_imgs.append(stitched_img)
                    img_list.pop(i)
            if i==len(img_list):
                img_list.pop()
    loop_num = 0
    res = None
    num_stitiched_img=len(stitched_imgs)
    while loop_num <= num_stitiched_img:
        loop_num += 1
        print("number of stitch img %d, loop number %d" % (num_stitiched_img, loop_num))
        for i in range(num_stitiched_img):
            if res is None:

                res = stitched_imgs[i]
            else:
                stitched_img = stitch(res[0], stitched_imgs[i][0], top, bot, left, right,40)

                if stitched_img is None:
                    pass
                else:
                    res = None
                    stitched_imgs.append(stitched_img)
                    #stitched_imgs.pop(i)

    i=0
    for img in stitched_imgs:
        cv.imwrite(cwp+"stitched_"+str(i)+"-"+str(img[1])+".jpg",img[0])
        i=i+1













