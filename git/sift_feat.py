# coding: utf-8
# @Author    : LittleAnt
# @Time      :
# @Descrip   : SIFT features
import cv2
import os
import sys
import numpy as np


class FeatsDetect(object):
    def __call__(self, img1, img2, method=0):
        if method == 0:
            return self._orb_bf_match(img1, img2)
        elif method == 1:
            return self._sift_bf_match(img1, img2)
        elif method == 2:
            return self._sift_flann_match(img1, img2)

    def _orb_bf_match(self, img1, img2):
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with orb
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        return (kp1, kp2, good)

    def _sift_bf_match(self, img1, img2):
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        return (kp1, kp2, good)

    def _sift_flann_match(self, img1, img2):
        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        return (kp1, kp2, good)
