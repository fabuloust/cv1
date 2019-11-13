import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


def generate_image():
    img = cv2.imread('origin.jpg', 1)
    img1 = img[:, 0: 500]
    img2 = img[:, 250:]
    # 对img2进行下变换

    M = cv2.getRotationMatrix2D((img2.shape[1] / 2, img2.shape[0] / 2), 30, 1)  # center, angle, scale
    img_rotate = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))

    # cv2.imwrite('img1.jpg', img1)
    cv2.imwrite('img2.jpg', img_rotate)

    # cv2.imshow('affine lenna', dst)
    # cv2.imshow('img1', img1)
    # key = cv2.waitKey(0)
    # if key == 27:
    #     cv2.destroyAllWindows()
    #     cv2.waitKey(1)


def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append((m.trainIdx, m.queryIdx))
    return good


def get_key_points(img1, img2):
    g_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, kp2 = {}, {}
    sift = cv2.xfeatures2d.SIFT_create()
    kp1['kp'], kp1['des'] = sift.detectAndCompute(g_img1, None)
    kp2['kp'], kp2['des'] = sift.detectAndCompute(g_img2, None)
    return kp1, kp2


def paste_2_image(img1, img2, homo_matrix):

    h1, w1 = img1.shape[0], img1.shape[1]
    h2, w2 = img2.shape[0], img2.shape[1]
    rect1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape((4, 1, 2))
    rect2 = np.array([[0, 0], [0, h2], [w1, h2], [w2, 0]], dtype=np.float32).reshape((4, 1, 2))
    trans_rect2 = cv2.perspectiveTransform(rect2, homo_matrix)
    total = np.concatenate((rect1, trans_rect2), axis=0)
    minx, miny = np.int32(total.min(axis=0).ravel())
    maxx, maxy = np.int32(total.max(axis=0).ravel())
    shift_to_zero_matrix = np.array([[1, 0, -minx], [0, 1, -miny], [0, 0, 1]])
    trans_img2 = cv2.warpPerspective(img2, shift_to_zero_matrix.dot(homo_matrix), (maxx - minx, maxy - miny))
    trans_img2[-miny:h1-miny, -minx:w1 - minx] = img1
    return trans_img2


if __name__ == '__main__':

    # generate_image()
    # 1.寻找sift特征点
    img1 = cv2.imread('img1.jpg')
    img2 = cv2.imread('img2.jpg')

    kp1, kp2 = get_key_points(img1, img2)
    good_match = get_good_match(kp1['des'], kp2['des'])

    print(len(kp1), len(good_match), len(kp2))
    ptsA = np.float32([kp1['kp'][i].pt for (_, i) in good_match])
    ptsB = np.float32([kp2['kp'][i].pt for (i, _) in good_match])
    ransacReprojThreshold = 4
    H, status = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 4)
    result = paste_2_image(img1, img2, H)
    cv2.imwrite('result.jpg', result)

