import glob
import numpy as np
import cv2
import scipy
from matplotlib import pyplot as plt
from scipy.signal import convolve2d



#TODO Flow:
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

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    gradient = np.arctan2(sobely, sobelx)

    plt.imshow(gradient, cmap='gray')
    plt.title("gradient")
    plt.show()
    plt.figure()
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(edges, cmap='gray')
    plt.title("edge image"), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3) , plt.imshow(sobely, cmap='gray')
    plt.title("sobel-y"), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4), plt.imshow(sobelx, cmap='gray')
    plt.title("sobel-x"), plt.xticks([]), plt.yticks([])



    plt.show()

