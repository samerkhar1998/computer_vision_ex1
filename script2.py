import os
import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt

def my_printer(str):
    print(f"----------------------{str}----------------------")

def read_images():
    my_printer("Reading Images")
    path_to_dir = "./denoising_sets/"
    images_set = []
    for dir in os.listdir(path_to_dir):
            curr_path = path_to_dir + dir + "/"
            curr_images = []
            for image in os.listdir(curr_path):
                curr_img = cv.imread(os.path.join(curr_path, image))
                curr_img = cv.cvtColor(curr_img, cv.COLOR_BGR2GRAY)
                if image == "target.jpg":
                    target = curr_img
                else:
                    curr_images.append(curr_img)

            images_set.append((target, curr_images))
    return images_set


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
def knn_match(descriptor_target, descriptor_image, k=2):
    my_printer("MATCHING Images")
    dist_matrix = calc_dist_matrix(descriptor_target, descriptor_image)

    matches = []
    for i in range(len(descriptor_target)):
        curr_dist_row = dist_matrix[i]

        # 0: dist=15, 1: dist=7, 2: dist=2, 3: dist=166, 4: dist=1
        distances_dict = {j: curr_dist_row[j] for j in range(len(curr_dist_row))}

        # sorted(distances_dict.items(), key = lambda d: d[1])
        # 4: dist=1, 2: dist=2, 1: dist=7,  0: dist=15,  3: dist=166

        # sorted(distances_dict.items(), key = lambda d: d[1])[:k]
        # 4: dist=1, 2: dist=2, 1: dist=7,
        best_k_matches = [(i, j, dist) for j, dist in sorted(distances_dict.items(), key = lambda d: d[1])[:k]]

        matches.append(best_k_matches)
    return matches


def extract_matches_pass_ratio(matches, ratio=0.8):
    my_printer("RATIOING Images")
    passed_matches = []
    for match1, match2 in matches:
        dist1, dist2 = match1[2], match2[2]
        if dist1 < ratio * dist2:
            passed_matches.append(match1)
    return passed_matches

def find_homography(matches, target_kp, img_kp):
    my_printer("PROJECTING Images")
    src_pts = np.float32(img_kp[m[0][1]].pt for m in matches).reshape(-1, 1, 2)
    dst_pts = np.float32(target_kp[m[0][0]].pt for m in matches).reshape(-1, 1, 2)




    M = cv.getPerspectiveTransform(src_pts, dst_pts)

    print(M)

    return M


if __name__ == '__main__':
    images_set = read_images()
    for target, images in images_set:
        target_sift, img_sift = sift_calculator(target, images)
        target_kp, target_des = target_sift
        for i, curr_sift in enumerate(img_sift):
            img_kp, img_des = curr_sift
            curr_img = images[i]
            matches = knn_match(target_des, img_des)
            ratio_passed_matches = extract_matches_pass_ratio(matches, 0.8)

            M = find_homography(matches, target_kp, img_kp)
            h, w = curr_img.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)


            dst_img = cv.warpPerspective(curr_img, M, (h, w))

            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(target)
            plt.title("target")
            plt.subplot(1, 2, 2)
            plt.imshow(curr_img)
            plt.title("source")
            plt.subplot(2, 1, 1)
            plt.imshow(dst_img)
            plt.title("destination image")
            plt.show()








