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


'''''
""" 1 """

def hough_centering(img,image_slops):
    """
    check centering in each iteration for all elements
    :param img: given image (dont need it )
    :param image_slops: slopes of every element in the image
    :return: an image of votes
    """
    voting_image=np.zeros((len(img),len(img[0])))
    for i in range(len(img)):
        for j in range(len(img[0])):
            for row in range(i,len(img)):
                for col  in range(j,len(img[0])):
                    # col represents y
                    # row represents x
                        p,q,z_p,z_q=(j,i),(col,row),image_slops[i][j],image_slops[row][col]
                        a,b,flag=centering(p,q,z_p,z_q)
                        vote(a,b,voting_image) if flag else None
    votes2=voting_image.ravel()

def centering(p,q,z_p,z_q):
    """ given two points ,and thier slopes ,return its TM line as two parameters
      :param p  an edge point
      :param q  an edge point
      :param z_q  slop of q
      :param z_p  slop of p
      :return a,b,flag  as in y=ax+b ,flag is true if a and b are relevant False otherwise
    """

    (x1,y1)=p[0],p[1]
    (x2,y2)=q[0],q[1]
    a=b=0
    flag=False
    if z_p!=z_q:
        t1 = (y1 - y2 - x1 * z_p + x2 * z_q) / (z_q - z_p)
        t2 = ((z_p * z_q)(x2 - x1) - y2 * z_p + y1 * z_q) / (z_q - z_p)
        m1 = (x1 + x2) / 2
        m2 = (y1 + y2) / 2
        a=(t2-m2)/(t1-m1)
        b=(m2*t1-m1*t2)/(t1-m1)
        flag=True
    return a,b,flag

def vote(a,b,votes):
    initial_x=False
    for i in range(len(votes)):
        initial_x= i if i*a+b else False
    for i in range(initial_x,len(votes)):
        votes[i][i*a+b]+=1

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

