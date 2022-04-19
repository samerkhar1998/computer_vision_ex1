import os
import random

import cv2
import cv2 as cv
import math
import numpy as np
from sklearn.metrics import euclidean_distances


cameleon_set = []
eagle_set = []
einstein_set = []
palm_set = []
set = []


def my_printer(str):
    print(f"----------------------{str}----------------------")


def plotImages(target, denoised, imageName):
    my_printer("Plotting images ")
    path = 'C:/Users/Samer/Desktop/'
    imageToSave = denoised * 255

    cv.imwrite(os.path.join(path, f'{imageName}.jpg'), imageToSave)
    cv.waitKey(0)

    cv.imshow('Denoising Result', denoised)
    cv.imshow('Target Image', target)
    cv.waitKey(0)


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


def calc_dist_matrix(des1, des2):
    my_printer("Calculating distance matrix")
    return euclidean_distances(des1, des2)


class match:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance


def knn_match(img_des, target_des, k=2):
    dist_matrix = calc_dist_matrix(img_des, target_des)
    my_printer("Knn Matches")

    toRet = []
    for i in range(len(img_des)):
        x = dist_matrix[i]
        dict = {}
        for j in range(len(x)):
            dict[j] = x[j]
        sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda x: x[1])}
        list = []
        k_counter = 0
        for t in sorted_dict:
            if k_counter == k:
                break
            list.append(match(i, t, sorted_dict[t]))
            k_counter += 1
        toRet.append(list)
    return toRet


def pass_ratio(matches, ratio=0.8):
    my_printer("Calculating matches pass ratio")
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    return good


def RANSAC(matches, target_kp, img_kp, threshold, iterations=100, k=4):
    my_printer("RANSAC is running")

    best_model_inliers = -1
    best_M = None

    for i in range(iterations):
        counter = 0
        random_matches = random.sample(matches,
                                       k)  # Sample random matches according to homographic = k

        src_pts = np.float32([img_kp[random_matches[i].queryIdx].pt for i in range(k)])
        dst_pts = np.float32([target_kp[random_matches[i].trainIdx].pt for i in range(k)])

        M, mask = cv.findHomography(src_pts, dst_pts)

        for m in matches:
            m_source = img_kp[m.queryIdx].pt
            m_target_old = target_kp[m.trainIdx].pt
            m_target_new = np.matmul(M, [m_source[0], m_source[1], 1])

            if m_target_new[2] != 0:
                m_target_new[0] = m_target_new[0] / m_target_new[2]
                m_target_new[1] = m_target_new[1] / m_target_new[2]

            euclidean_distance = math.sqrt(
                ((m_target_old[0] - m_target_new[0]) ** 2) + ((m_target_old[1] - m_target_new[1]) ** 2))
            if euclidean_distance < threshold:
                counter += 1

        if counter > best_model_inliers:
            best_model_inliers = counter
            best_M = M

    return best_M, matches


def script(images_set, threshold, set_name):
    results = []
    target, images = images_set[0], images_set[1]

    target_sift, img_sift = sift_calculator(target, images)

    target_kp, target_des = target_sift

    # bf = cv.BFMatcher()
    counters = np.zeros(shape=target.shape[:2])

    for i, curr_sift in enumerate(img_sift):
        img_kp, img_des = curr_sift
        curr_img = images[i]

        matches = knn_match(img_des, target_des)

        good_matches = pass_ratio(matches, ratio=0.8)

        M, matches = RANSAC(good_matches, target_kp, img_kp, threshold)

        h, w, channels = target.shape

        dst = cv.warpPerspective(curr_img, M, (w, h))

        results.append(dst)

        new_img = np.copy(dst)

        cv2.imshow("warp 1", dst)
        cv2.waitKey(0)

        for i in range(len(new_img)):
            for j in range(len(new_img[i])):
                color = new_img[i][j]
                if color.any():
                    counters[i][j] += 1

    denoised_im = np.zeros(shape=target.shape, dtype=np.float32)
    for res in results:
        denoised_im = denoised_im + res

    denoised_im /= 255  # colors in cv2 are saved as values in [0, 1] instead of [0, 255]

    for i in range(denoised_im.shape[0]):
        for j in range(denoised_im.shape[1]):
            if counters[i][j] != 0:  # avoiding dividing by 0
                denoised_im[i][j] /= counters[i][j]

    plotImages(target, denoised_im, set_name)


if __name__ == '__main__':
    read_images()

    # my_printer("working on cameleon")
    # script(cameleon_set, 0.5, "cameleon_Denoised")

    my_printer("working on eagle")
    script(eagle_set, 0.5, "eagle_Denoised")
    #
    # my_printer("working on palm")
    # script(palm_set,100, "palm_Denoised")
    #
    # my_printer("working on einstein")
    # script(einstein_set,100, "einstein_Denoised")
