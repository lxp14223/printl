import os
import sys
import pdb
import random
import cv2
import numpy as np
import torch
from models.matching import Matching
from models.utils import read_image
torch.set_grad_enabled(False)


class PointMatch(object):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running inference on device \"{}\"'.format(self.device))
        config = {
            'superpoint': {
                'nms_radius': 3,
                'keypoint_threshold': 0.001,
                'max_keypoints': 2048
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 30,
                'match_threshold': 0.1,
            }
        }
        self.matching = Matching(config).eval().to(self.device)
    
    def __call__(self, img1, img2):
        height, width =  img1.shape[:2]
        image0, inp0, scales0 = read_image(
            img1, self.device, [width,height], 0, True)
        image1, inp1, scales1 = read_image(
            img2, self.device, [width,height], 0, True)
        pred = self.matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        img_pts0 = mkpts0 * scales0
        img_pts1 = mkpts1 * scales1
        return img_pts0, img_pts1

if __name__ == '__main__':
    img1 = cv2.imread('./data/match1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./data/match2.png', cv2.IMREAD_GRAYSCALE)
    match_cnn = PointMatch()
    match_cnn(img1, img2)