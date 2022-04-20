import glob
import math
import numpy as np
import cv2
import scipy
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
# TODO Flow:
'''
1.  For each two edge points (P, Q) in edge image:
    If the tangents are not paralell:
    Calculate M = mid point of (P, Q):
        M_x = (P_x + Q_x)/2 (Same for Y)

    e1 = Slope of the Tangent at point P
    e2 = Slope of the Tangent at point Q

    The equation for line MT is:

    y(t1 - m1) = x(t2 - m2) + m2*t1 - m1*t2

    t1 = (y1 - y2 - x1*e1 + x2*e2)/(e2-e1)
    t2 = ( e1*e2*(x2 - x1) - y2*e1 + y1*e2)/(e2 - e1)

2.  Vote for each point on MT line 
-- possible_ellipse = (center_x, center_y, ... )
-- votes[possible_ellipse] += 1
3.  The points with most votes after 2. are the winners for Ellipse Center
-- sorted(votes, reverse=True)[:k]

'''

""" 1 """
votes = {}


def my_printer(str):
    print(f"----------------------{str}----------------------")

def draw_ellipses(votings_img, votes, img):
    accepted_ellipses = []
    counter = 0
    for vote in votes:
        if counter == 60:  # max number of ellipses to draw
            break
        accepted_ellipses.append(vote)
        counter += 1

        img = cv2.circle(img, (vote[0], vote[1]), 3, (0, 0, 255), -1)

    cv2.imshow("result", img)
    cv2.waitKey(0)


def plot(img, reuslt, centers, result):
    plt.figure()
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('canny image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(result, cmap='gray')
    plt.title("result"), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(centers, cmap='gray')
    plt.title("edge_y"), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(result, cmap='gray')
    plt.title("edge_x"), plt.xticks([]), plt.yticks([])

    plt.show()

def readImages():
    images = []
    # image 1
    img = cv2.imread("./ellipses/9_small.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_gray)

    # image 2
    img = cv2.imread("./ellipses/2_small.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_gray)

    # image 3
    img = cv2.imread("./ellipses/3.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_gray)

    # image 4
    img = cv2.imread("./ellipses/5.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_gray)

    # image 5
    img = cv2.imread("./ellipses/8.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_gray)

    # image 6
    img = cv2.imread("./ellipses/7.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_gray)

    return images





def find_TM_line(p, q, e1, e2):
    """ given two points ,and thier slopes ,return its TM line as two parameters
      :param p  an edge point
      :param q  an edge point
      :param e1  slop of q
      :param e2  slop of p
      :return a,b,flag  as in y=ax+b ,flag is true if a and b are relevant False otherwise
    """
    (x1, y1) = p[0], p[1]
    (x2, y2) = q[0], q[1]
    a = b = 0
    m1,m2 = None, None
    intersection = None
    if e1 != e2:  # checks if the edges are parallel
        t1 = int((y1 - y2 - x1 * e1 + x2 * e2) / (e2 - e1))
        t2 = int(((e1 * e2) * (x2 - x1) - y2 * e1 + y1 * e2) / (e2 - e1))
        m1 = int((x1 + x2) / 2)
        m2 = int((y1 + y2) / 2)
        if t1 != m1:
            a = (t2 - m2) / (t1 - m1)
            b = (m2 * t1 - m1 * t2) / (t1 - m1)
            intersection = (t1, t2)
    return a, b, m1, m2, intersection


def vote(a, b, m, votesImg, intersection):
    # y = ax + b
    # x = (y-b)/a
    # temp = np.zeros(votesImg.shape)
    # votesImg = cv2.line(temp, m, intersection, (0.0001, 0, 0), 1) + votesImg

    # cv2.imshow("votes image", votesImg)
    # cv2.waitKey(0)
    # return votesImg
    for x in range(votesImg.shape[1]):
        y = a * x + b
        if 0 <= y < votesImg.shape[0]:
            possible_ellipse = (x, round(y))
            if possible_ellipse not in votes:
                votes[possible_ellipse] = 1
            else:
                votes[possible_ellipse] += 1

def hough_ellipses(edges, image_slops, thetas_para):
    """
    check centering in each iteration for all elements
    :param edges: given image (dont need it )
    :param image_slops: slopes of every element in the image
    :return: an image of votes
    """
    counter = 0
    voting_image = np.zeros((len(edges), len(edges[0])))
    for edge in thetas_para:
        my_printer(counter)
        counter += 1
        for edge2 in thetas_para:
            # col represents y
            # row represents x
            p, q = (edge[1], edge[0]), (edge2[1], edge2[0])
            # if radiuses are bigger than what we expect no need to vote for it

            e1, e2 = image_slops[edge[0]][edge[1]], image_slops[edge2[0]][edge2[1]]
            a, b, m1, m2, intersection = find_TM_line(p, q, e1, e2)
            if abs(p[0] - q[0]) < 60 and abs(p[1] - q[1]) < 60:
                if intersection != None:
                    vote(a, b, (m1, m2), voting_image, intersection)

    my_printer("done")
    sorted_v = {k: v for k, v in sorted(votes.items(), key=lambda item: item[1], reverse=True)}
    return voting_image, sorted_v

def run_script(img, gaussian, threshold1, threshold2, k=10):

    canny_edge = cv2.Canny(gaussian, threshold1=threshold1, threshold2=threshold2)

    edge_x = cv2.Sobel(canny_edge, cv2.CV_64F, 1, 0, ksize=-1)
    edge_y = cv2.Sobel(canny_edge, cv2.CV_64F, 0, 1, ksize=-1)

    my_printer("calculating gradients")
    gradients = np.arctan2(edge_y, edge_x)

    # canny_edge = canny_edge / 255
    gradients_para = []
    for i in range(len(gradients)):
        for j in range(len(gradients[0])):
            if gradients[i][j] != 0:
                gradients_para.append([i, j])

    votingImg, votes = hough_ellipses(canny_edge, gradients, gradients_para)

    # draw_ellipses(votingImg, votes, img)
    plot(votingImg, canny_edge, edge_y, edge_x)
    # displayCircles(result)

if __name__ == "__main__":

    images = readImages()
    # image 1
    edges_0 = cv2.GaussianBlur(images[0], (7, 7), 2)
    # run_script(edges_0, threshold1=50, threshold2=150)

    # image 2
    gaussian_1 = cv2.GaussianBlur(images[1], (7, 7), 1)
    for i in range(4):
        edges_1 = cv2.GaussianBlur(gaussian_1, (7, 7), 1)

    run_script(images[1], gaussian_1, threshold1=10, threshold2=80)

    # image 3
    edges_2 = cv2.GaussianBlur(images[2], (5, 5), 2)
    edges_2 = cv2.GaussianBlur(edges_2, (5, 5), 1)
    # run_script(edges_2, threshold1=50, threshold2=150)

    # image 4
    edges_3 = cv2.GaussianBlur(images[3], (5, 5), 1)
    # run_script(edges_3, threshold1=120, threshold2=50)

    # image 5
    edges_4 = cv2.GaussianBlur(images[4], (7, 7), 6)
    edges_4 = cv2.GaussianBlur(edges_4, (7, 7), 5)
    edges_4 = cv2.GaussianBlur(edges_4, (7, 7), 8)
    # run_script(edges_4, threshold1=50, threshold2=164)

    # image 6
    edges_5 = cv2.GaussianBlur(images[5], (7, 7), 1)
    for i in range(8):
        edges_5 = cv2.GaussianBlur(edges_5, (7, 7), 2)

    # run_script(edges_5, threshold1=10, threshold2=160)
