import os
import random

import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt
import threading

cameleon_set = []
eagle_set = []
einstein_set = []
palm_set = []
set = []


def my_printer(str):
    print(f"----------------------{str}----------------------")


def read_images():
    global cameleon_set, eagle_set, einstein_set, palm_set
    my_printer("Reading Images")
    path_to_dir = "./denoising_sets/"

    for dir in os.listdir(path_to_dir):
        curr_path = path_to_dir + dir + "/"
        curr_images = []
        for image in os.listdir(curr_path):
            curr_img = cv.imread(os.path.join(curr_path, image))
            # curr_img = cv.cvtColor(curr_img, cv.COLOR_BGR2GRAY)
            if image == "target.jpg":
                target = curr_img
            else:
                curr_images.append(curr_img)

        if dir == 'cameleon__N_8__sig_noise_5__sig_motion_103':
            cameleon_set = [target, curr_images]
        elif dir == 'eagle__N_16__sig_noise_13__sig_motion_76':
            eagle_set = [target, curr_images]
        elif dir == 'palm__N_4__sig_noise_5__sig_motion_ROT':
            palm_set = [target, curr_images]
        elif dir == 'einstein__N_5__sig_noise_5__sig_motion_274':
            einstein_set = [target, curr_images]


def sift_calculator(target, images):
    my_printer("SIFTing Images")
    sift = cv.SIFT_create()
    target_des = sift.detectAndCompute(target, None)
    img_descriptors = []
    for img in images:
        img_descriptors.append(sift.detectAndCompute(img, None))

    return target_des, img_descriptors


def calc_descriptor_distance(des1, des2):
    return np.linalg.norm(des1 - des2)


def calc_dist_matrix(des1, des2):
    my_printer("CALCULATING DIST MATRIX")
    distance_matrix = []
    for i in range(len(des1)):
        curr_row = []
        for j in range(len(des2)):
            curr_dist = calc_descriptor_distance(des1[i], des2[j])
            curr_row.append(curr_dist)
        distance_matrix.append(curr_row)
    return distance_matrix


# For each row in descriptor_target, get the best k indexes in descriptor_image that have best matches
# def knn_match(descriptor_target, descriptor_image, k=2):
#
#     dist_matrix = calc_dist_matrix(descriptor_target, descriptor_image)
#     my_printer("MATCHING Images")
#     matches = []
#     for i in range(len(descriptor_target)):
#         curr_dist_row = dist_matrix[i]
#
#         # 0: dist=15, 1: dist=7, 2: dist=2, 3: dist=166, 4: dist=1
#         distances_dict = {j: curr_dist_row[j] for j in range(len(curr_dist_row))}
#
#         # sorted(distances_dict.items(), key = lambda d: d[1])
#         # 4: dist=1, 2: dist=2, 1: dist=7,  0: dist=15,  3: dist=166
#
#         # sorted(distances_dict.items(), key = lambda d: d[1])[:k]
#         # 4: dist=1, 2: dist=2, 1: dist=7,
#         best_k_matches = [(i, j, dist) for j, dist in sorted(distances_dict.items(), key=lambda d: d[1])[:k]]
#
#         matches.append(best_k_matches)
#     return matches
#


# def extract_matches_pass_ratio(matches, ratio=0.8):
#     my_printer("RATIOING Images")
#     passed_matches = []
#     for match1, match2 in matches:
#         dist1, dist2 = match1.distance, match2.distance
#         if dist1 < ratio * dist2:
#             passed_matches.append([match1])
#     return passed_matches


def find_homography(matches, target_kp, img_kp):
    my_printer("PROJECTING Images")

    src_pts = np.zeros((len(matches), 2), dtype=np.float32)
    dst_pts = np.zeros((len(matches), 2), dtype=np.float32)

    for i in range(len(matches)):
        src_pts[i, :] = img_kp[matches[i][0].trainIdx].pt
        dst_pts[i, :] = target_kp[matches[i][0].queryIdx].pt

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    return M, mask


def plotImages(target, denoised, imageName):
    # image = cv.cvtColor(dst_img, cv.COLOR_GRAY2BGR)
    path = 'C:/Users/Samer/Desktop/'
    imageToSave = denoised * 255

    # cv.imwrite(os.path.join(path, f'{imageName}.jpg'), imageToSave)
    # cv.waitKey(0)

    cv.imshow('Denoising Result', denoised)
    cv.imshow('Target Image', target)
    cv.waitKey(0)


def findMatches(target_des, img_des, target_kp, img_kp, ratio=0.8, iterations=10, k=4):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(target_des, img_des, k=2)
    # matches2 = bf.knnMatch(img_des, target_des, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    best_model_inliers = -1
    best_M = None

    for i in range(250):
        counter = 0
        random_matches = random.sample(good_matches,
                                       k)  # Sample random matches according to homographic/affine parameter = k

        src_pts = np.float32([img_kp[random_matches[i].trainIdx].pt for i in range(k)])
        dst_pts = np.float32([target_kp[random_matches[i].queryIdx].pt for i in range(k)])

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        for m in good_matches:
            m_source = img_kp[m.trainIdx].pt
            m_target_old = target_kp[m.queryIdx].pt
            m_target_new = np.matmul(M, [m_source[0], m_source[1], 1])

            if m_target_new[2] != 0:
                m_target_new[0] = m_target_new[0] / m_target_new[2]
                m_target_new[1] = m_target_new[1] / m_target_new[2]

            euclidean_distance = math.sqrt(
                ((m_target_old[0] - m_target_new[0]) ** 2) + ((m_target_old[1] - m_target_new[1]) ** 2))
            if euclidean_distance < 0.5:
                counter += 1

        if counter > best_model_inliers:
            best_model_inliers = counter
            best_M = M

    return best_M, good_matches


def script(images_set, set_name):
    results = []
    target, images = images_set[0], images_set[1]

    target_sift, img_sift = sift_calculator(target, images)

    target_kp, target_des = target_sift

    # bf = cv.BFMatcher()
    counters = np.zeros(shape=target.shape[:2])

    for i, curr_sift in enumerate(img_sift):
        img_kp, img_des = curr_sift
        curr_img = images[i]

        M, matches = findMatches(target_des, img_des, target_kp, img_kp)

        h, w, channels = target.shape

        dst = cv.warpPerspective(curr_img, M, (w, h))

        results.append(dst)

        ones = np.ones(shape=target.shape[:2])
        ones = cv.warpPerspective(ones, M, (w, h))

        # cv.imshow('Denoising Result', dst)
        # cv.imshow('Target Image', ones)
        # cv.waitKey(0)
        new_img = np.copy(dst)
        for i in range(len(new_img)):
            for j in range(len(new_img[i])):
                color = new_img[i][j]
                if color.any():
                    counters[i][j] += 1

        # ones[dst != 0] = 1
        # counters += ones


    denoised_im = np.zeros(shape=target.shape, dtype=np.float32)
    for res in results:
        denoised_im = denoised_im + res

    denoised_im /= 255  # colors in cv2 are saved as values in [0, 1] instead of [0, 255]

    for i in range(denoised_im.shape[0]):
        for j in range(denoised_im.shape[1]):
        #     for k in range(denoised_im.shape[2]):
            if counters[i][j] != 0:  # avoiding dividing by 0
                denoised_im[i][j] /= counters[i][j]

    plotImages(target, denoised_im, set_name)


if __name__ == '__main__':
    read_images()

    my_printer("working on cameleon")
    script(cameleon_set, "cameleon_Denoised")
    # my_printer("working on eagle")
    # script(eagle_set, "eagle_Denoised")
    # my_printer("working on palm")
    # script(palm_set, "palm_Denoised")
    # my_printer("working on einstein")
    # script(einstein_set, "einstein_Denoised")
