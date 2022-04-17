import glob
import numpy as np
import cv2
import scipy
from matplotlib import pyplot as plt
from scipy.signal import convolve2d


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
    img = cv2.imread("./ellipses/3_small.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_gray)

    # image 4
    img = cv2.imread("./ellipses/5_small.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_gray)

    # image 5
    img = cv2.imread("./ellipses/6_small.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_gray)

    # image 6
    img = cv2.imread("./ellipses/7_smal.jpeg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_gray)

    return images

def hough_ellipses(image):
    newImg = np.copy(image)
    circles = cv2.HoughCircles(newImg, cv2.HOUGH_GRADIENT, 5, 90, param1=20, param2=10, minRadius=10, maxRadius=30)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(newImg, (i[0], i[1]), i[2], (0, 255, 0), 2)

    plt.figure("part")
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("original")
    plt.subplot(1, 2, 2)
    plt.imshow(newImg, cmap='gray')
    plt.title("hough circles")
    plt.show()


if __name__ == "__main__":
    images = readImages()
    # first image:
    img = images[0]
    edges = cv2.GaussianBlur(img, (7, 7), 1)
    edges = cv2.Canny(edges, 50, 150)

    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    edges = cv2.erode(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )

    plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray')
    plt.title("edge image"), plt.xticks([]), plt.yticks([])


    plt.show()

