import cv2
import numpy as np

DATA_PATH = './data/samples/'
img = cv2.imread(DATA_PATH + 'lena.png')

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
        self.epsilon = 1e-3

    def change_to(self, val: int):
        self.current_filter_idx = val

    @property
    def filter(self):
        return self.filters[self.current_filter_idx] * 6*self.epsilon**2



def apply_laplacian(img_gray: cv2.typing.MatLike) -> cv2.typing.MatLike:
    filtered = cv2.filter2D(img_gray, cv2.CV_64F, filter_man.filter)
    return filtered

def normalize_laplacian(laplacian: cv2.typing.MatLike) -> cv2.typing.MatLike:
    return cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    

def detect_zero_crossing(laplacian: cv2.typing.MatLike, middle_value: int = 0) -> cv2.typing.MatLike:
    # computes the minimun on the kernel neighborhood
    kernel = np.ones((2,2))
    min_l = cv2.erode(laplacian, kernel)
    # computes the maximum on the kernel neighborhood
    max_l = cv2.dilate(laplacian, kernel)

    # zero crossing is basically when l_erode(i,j) is negative and l_dilate(i,j) is positive
    # that means that in the neighborhood of (i,j) it had a positive and a negative value,
    # meaning a zero crossing

    zero_cross = (((min_l < middle_value) & (laplacian > middle_value)) | ((max_l > middle_value) & (laplacian < middle_value)))
    return zero_cross.astype(np.uint8) * 255

filter_man = FilterManager()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('norm_laplacian')
cv2.createTrackbar('change_kernel', 'norm_laplacian', 0, 1, filter_man.change_to)

while True:
    cv2.imshow('img', img)
    img_gray_gauss = cv2.GaussianBlur(img_gray, (7,7), 0)
    cv2.imshow('gauss', img_gray_gauss)
    norm_lapl = normalize_laplacian(apply_laplacian(img_gray_gauss))
    cv2.imshow('norm_laplacian', norm_lapl)
    zero_cross = detect_zero_crossing(norm_lapl, 127)
    cv2.imshow('zero_cross', zero_cross)

    
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

