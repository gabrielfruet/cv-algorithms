import cv2
from numba import njit
import numpy as np
from cv2.typing import MatLike, MatShape

img = cv2.imread('./data/samples/street.jpg')
img = cv2.resize(img, (0,0), None, 0.2, 0.2)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(gray_img: MatLike, hi_thresh: int, low_thresh: int) -> MatLike:
    """
    Apply the Canny edge detector algorithm to the `gray_img` argument, that
    should be a CV_8U image.

    Parameters:
        `gray_img` - A CV_8U image
        `hi_thresh` - the high bound threshold used on Canny
        `low_thresh` - the low bound (...)
    Return:
        The edges of the image
    """

    gray_img = cv2.GaussianBlur(gray_img, (3,3), 4)

    sobel_hor = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])
    sobel_ver = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    hor = cv2.filter2D(gray_img, cv2.CV_64F, sobel_hor)
    ver = cv2.filter2D(gray_img, cv2.CV_64F, sobel_ver)

    grad_mag = np.sqrt(hor**2 + ver**2)
    x = hor/(grad_mag + 1e-2)
    y = ver/(grad_mag + 1e-2)

    orient = np.arctan(y/(x + 1e-2)) * 180 / np.pi
    nms_img = non_maximum_supression(grad_mag, orient)
    thresh = threshold(nms_img, hi_thresh, low_thresh)
    hys = hysteresis(thresh)

    return hys

@njit
def non_maximum_supression(grad_mag: MatLike, orient: MatLike) -> MatLike:
    """
    Applies nms using a 3x3 kernel to the `grad_mag` matrix, in the direction
    of the `orient` matrix

    This function is JIT compiled for perfomance purposes

    Return:
        NMS of `grad_mag` at the direction of `orient`

    """
    assert grad_mag.shape == orient.shape
    nms_img = np.zeros_like(orient)
    shape: MatShape = orient.shape

    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            theta: np.float32 = orient[i,j]
            anchor: np.float32 = grad_mag[i,j]
            minv = np.nan

            if (theta > 0 and theta < 22.5) or (theta < 180 and theta > 157.5):
                minv = max([grad_mag[i,j-1], anchor, grad_mag[i,j+1]])

            elif theta > 22.5 and theta < 67.5:
                minv = max([grad_mag[i-1,j+1], anchor, grad_mag[i+1,j-1]])

            elif theta > 67.5 and theta < 112.5:
                minv = max([grad_mag[i-1,j], anchor, grad_mag[i+1,j]])

            elif theta > 112.5 and theta < 157.5:
                minv = max([grad_mag[i-1,j-1], anchor, grad_mag[i+1,j+1]])

            if anchor == minv:
                nms_img[i,j] = 0
            else:
                nms_img[i,j] = anchor

    return nms_img




def threshold(img: MatLike, low_thresh: int, hi_thresh: int) -> MatLike:
    g_nh = cv2.threshold(img, hi_thresh, 255, cv2.THRESH_BINARY)[1]
    g_nl = cv2.threshold(img, low_thresh, 25, cv2.THRESH_BINARY)[1]
    g_nl = cv2.add(g_nh, g_nl, dtype=cv2.CV_8U)

    return g_nl

@njit
def hysteresis(g_nl: MatLike, weak:int=25, strong:int=255) -> MatLike:
    """
    Turns the pixels of `g_nl` that does match the hysteresis
    to `strong` and the others to 0

    Return
        modified copy of `g_nl`
    """
    shape: MatShape = g_nl.shape
    g_nl = g_nl.copy()

    eight_connect = [
        (-1,-1), (-1, 0), (-1, 1), 
        (0, -1),          (0, 1), 
        (1, -1), (1, 0) , (1, 1),
    ]

    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            if g_nl[i,j] == weak:
                for di, dj in eight_connect:
                    if g_nl[i + di, j + dj] == strong:
                        g_nl[i, j] = strong
                        break

    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            if g_nl[i,j] < strong:
                g_nl[i, j] = 0

    return g_nl


def normalize(gray_img: MatLike) -> MatLike:
    ret = gray_img.copy()
    _ = cv2.normalize(gray_img, ret, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return ret



class ThreshManager:
    def __init__(self):
        self.low_thresh = 0
        self.hi_thresh = 0

    def change_low_to(self, val: int):
        self.low_thresh = val

    def change_hi_to(self, val: int):
        self.hi_thresh = val


def display():
    thresh_man = ThreshManager()
    cv2.namedWindow('canny')
    cv2.createTrackbar('change_hi_threshold', 'canny', 0, 255, thresh_man.change_hi_to)
    cv2.createTrackbar('change_low_threshold', 'canny', 255, 255, thresh_man.change_low_to)


    while True:
        res = canny(img_gray, thresh_man.hi_thresh, thresh_man.low_thresh)
        cv2.imshow('canny', res)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

display()
