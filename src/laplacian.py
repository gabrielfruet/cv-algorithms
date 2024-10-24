import cv2
import numpy as np

img = cv2.imread('./lena.png')

class FilterManager:
    def __init__(self):
        simple_laplacian_kernel = np.array([
            [0,1,0],
            [1,-4,1],
            [0,1,0],
        ])

        cpx_laplacian_kernel = np.array([
            [1,4,1],
            [4,-20,4],
            [1,4,1],
        ])

        self.current_filter_idx = 0
        self.filters = [simple_laplacian_kernel, cpx_laplacian_kernel]

    def change_to(self, val: int):
        self.current_filter_idx = val

    @property
    def filter(self):
        return self.filters[self.current_filter_idx]



def apply_laplacian(img_gray: cv2.typing.MatLike) -> cv2.typing.MatLike:
    filtered = cv2.filter2D(img_gray, cv2.CV_16S, filter_man.filter)
    return filtered

def normalize_laplacian(laplacian: cv2.typing.MatLike) -> cv2.typing.MatLike:
    return cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    

def detect_zero_crossing(laplacian: cv2.typing.MatLike) -> cv2.typing.MatLike:
    # computes the minimun on the kernel neighborhood
    l_erode = cv2.erode(laplacian, np.ones((3,3)))
    # computes the maximum on the kernel neighborhood
    l_dilate = cv2.dilate(laplacian, np.ones((3,3)))

    # zero crossing is basically when l_erode(i,j) is negative and l_dilate(i,j) is positive
    # that means that in the neighborhood of (i,j) it had a positive and a negative value,
    # meaning a zero crossing
    middle_value = 0

    zero_cross = (((l_erode < middle_value) & (laplacian > middle_value)) | ((l_dilate > middle_value) & (laplacian < middle_value)))
    return zero_cross.astype(np.uint8) * 255

filter_man = FilterManager()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('laplacian_kernel')
cv2.createTrackbar('change_kernel', 'laplacian_kernel', 0, 1, filter_man.change_to)

while True:
    cv2.imshow('img', img)
    cv2.imshow('laplacian_kernel', normalize_laplacian(apply_laplacian(img_gray)))
    cv2.imshow('laplacian_kernel3', (apply_laplacian(img_gray)))
    cv2.imshow('laplacian_kernel2', detect_zero_crossing(apply_laplacian(img_gray)))
    cv2.imshow('laplace_original', detect_zero_crossing(cv2.Laplacian(img_gray, cv2.CV_64F)))
    
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

