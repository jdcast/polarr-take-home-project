#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Script that scores Gaussian blurred images on scale from 1 to 5.
1 is most blurred and 5 is lease blurred.
Approaches the problem by ranking the natural logarithm of the variance of the Laplacian calculated on each image.
This approach works well for extreme examples, but fails to capture the medium blurred images.
Inspiration for this approach comes from: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/.

Other possible approaches include:
1) highest Laplacian of the Gaussian blurriness estimator: https://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry/7767755#7767755
    This approach was attempted, as shown below.  However, it suffered from the same shortcomings of calculating the
    variance of the Laplacian while also failing to preserve logical ranking of the images' blurriness.
2) http://im.snibgo.com/measblur.htm (https://hal.archives-ouvertes.fr/hal-00232709/document)
3) https://imagemagick.org/discourse-server/viewtopic.php?t=34510
4) analyzing FFT for abundance of high frequency content: https://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry/7767755#7767755
5) potentially a hybridized approach using MSE.  This was started but abandoned.
"""

import cv2

import numpy as np


def mse(image_A=None, image_B=None):
    """
    The 'Mean Squared Error' between the two images is the
    sum of the squared difference between the two images.
    Return the MSE.
    The lower the error, the more "similar" the two images are.
    NOTE: the two images must have the same dimension

    :param image_A: numpy array, data for image
    :param image_B: numpy array, data for image
    :return: float, error
    """
    assert(image_A.shape == image_B.shape)
    err = np.sum((image_A.astype("float") - image_B.astype("float")) ** 2)
    err /= float(image_A.shape[0] * image_A.shape[1])

    return err

def variance_of_laplacian(img=None):
    """
    Calculates the variance of the laplacian: http://optica.csic.es/papers/icpr2k.pdf
    :param img: numpy array, data for image
    :return: float, variance of the laplacian
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()


if __name__ == "__main__":
    img_paths = [
        "/home/nightrider/polarr-take-home-project/q1_low_blur.png",
        "/home/nightrider/polarr-take-home-project/q1_medium_blur.png",
        "/home/nightrider/polarr-take-home-project/q1_high_blur.png"
    ]

    ### Method 1
    # Calculate the variation of the laplacian for each image.
    vl_scores = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        # h, w, d = low_blur_img.shape
        # print("img shape: {}".format(low_blur_img.shape))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vl = variance_of_laplacian(gray)
        vl_scores.append((img_path, vl))
    print("VL Scores: {}".format(vl_scores))

    log_vl_scores = [(item[0], np.log(item[1])) for item in vl_scores]
    print("Log of VL Scores: {}".format(log_vl_scores))

    min_vl_score = min(vl_scores, key=lambda item:item[1])[1]
    max_vl_score = max(log_vl_scores, key=lambda item:item[1])[1]
    print("Max VL Score: {}".format(max_vl_score))

    # Rank the images on scale of 1-5
    final_ranking = []
    for vl_score in log_vl_scores:
        overall_score = (vl_score[1] - min_vl_score) / (max_vl_score - min_vl_score) * (5 - 1) + 1
        final_ranking.append((vl_score[0], overall_score))

    print("Ranking: {}".format(final_ranking))

    ### Method 2
    # Calculate "highest LoG" blurriness estimator
    laplacian_scores = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        lap = cv2.Laplacian(img, ddepth=3, ksize=1, borderType=cv2.BORDER_DEFAULT)
        max_pxl = max(lap.flatten())
        laplacian_scores.append((img_path, max_pxl))
    print("Laplacian Scores: {}".format(laplacian_scores))

    ### Method 3 (incomplete)
    # Calculate the MSE of each image relative to least blurred image
    # blurred = cv2.GaussianBlur(src=img,
    #                            ksize=(111, 111),
    #                            sigmaX=10,
    #                            sigmaY=10,
    #                            borderType=cv2.BORDER_DEFAULT)
    #
    # diff = mse(image_A=img, image_B=blurred)
    #
    # print("mse: {}".format(diff))
    #
    # cv2.imshow("blurred image", blurred)
    # cv2.waitKey(3000)