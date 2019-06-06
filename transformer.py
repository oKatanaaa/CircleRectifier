from __future__ import print_function
from __future__ import division
import cv2 as cv
import argparse
import numpy as np

alpha_slider_max = 100
title_window = 'Linear Blend'

CM1 = 888.8
CM2 = 1740.8
CM3 = 924.8
CM4 = 1523.6

CM1_MAX = 1100
CM2_MAX = 1900
CM3_MAX = 1200
CM4_MAX = 1700

CM1_MIN = 700
CM2_MIN = 1600
CM3_MIN = 700
CM4_MIN = 1300

R1 = -0.2
R2 = 0.05
R3 = -0.0015
R4 = 0.002
R5 = -0.031

R1_MAX = 1
R2_MAX = 1
R3_MAX = 1
R4_MAX = 1
R5_MAX = 1

R1_MIN = -1
R2_MIN = -1
R3_MIN = -1
R4_MIN = -1
R5_MIN = -1

W = 0
H = 0

SCROLL_MAX = 1000
SCROLL_MIN = 0

IMAGE = None
camera_matrix = np.array([[CM1, 0., CM2],
                          [0., CM3, CM4],
                          [0., 0., 1.]])

dist_coefs = np.array([R1, R2, R3, R4])

def update_matrix():
    global camera_matrix
    global dist_coefs
    dist_coefs = np.array([R1, R2, R3, R4, R5])
    camera_matrix = np.array([[CM1, 0., CM2],
                              [0., CM3, CM4],
                              [0., 0., 1.]])

def get_new_value(val, min_, max_):

    percent = val / SCROLL_MAX
    print(percent)
    delta = max_ - min_
    print(min_ + delta * percent)
    return min_ + delta * percent


def on_cm1_trackbar(val):
    global CM1
    CM1 = get_new_value(val, CM1_MIN, CM1_MAX)
    update_everything()


def on_cm2_trackbar(val):
    global CM2
    CM2 = get_new_value(val, CM2_MIN, CM2_MAX)
    update_everything()


def on_cm3_trackbar(val):
    global CM3
    CM3 = get_new_value(val, CM3_MIN, CM3_MAX)
    update_everything()


def on_cm4_trackbar(val):
    global CM4
    CM4 = get_new_value(val, CM4_MIN, CM4_MAX)
    update_everything()


def on_r1_trackbar(val):
    global R1
    R1 = get_new_value(val, R1_MIN, R1_MAX)
    update_everything()


def on_r2_trackbar(val):
    global R2
    R2 = get_new_value(val, R2_MIN, R2_MAX)
    update_everything()


def on_r3_trackbar(val):
    global R3
    R3 = get_new_value(val, R3_MIN, R3_MAX)
    update_everything()


def on_r4_trackbar(val):
    global R4
    R4 = get_new_value(val, R4_MIN, R4_MAX)
    update_everything()


def on_r5_trackbar(val):
    global R5
    R5 = get_new_value(val, R5_MIN, R5_MAX)
    update_everything()


def update_everything():
    update_matrix()
    Knew = camera_matrix.copy()
    Knew[(0,1), (0,1)] = 0.4 * Knew[(0,1), (0,1)]
    D = np.array([0., 0., 0., 0.])
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (W, H), 1, (W, H))
    dst = cv.fisheye.undistortImage(IMAGE, camera_matrix, D=D, Knew=Knew)
    # dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
    cv.imshow(title_window, dst)


parser = argparse.ArgumentParser(description='Code for Adding a Trackbar to our applications tutorial.')
parser.add_argument('--input1', help='Path to the first input image.', default='LinuxLogo.jpg')
args = parser.parse_args()
IMAGE = cv.imread(cv.samples.findFile(args.input1))
H = IMAGE.shape[0]
W = IMAGE.shape[1]
if IMAGE is None:
    print('Could not open or find the image: ', args.input1)
    exit(0)

cv.namedWindow(title_window, cv.WINDOW_NORMAL)
cv.resizeWindow(title_window, 600, 600)
trackbar_cm1 = 'cm1'
value_cm1 = int(((CM1 - CM1_MIN) / (CM1_MAX - CM1_MIN))*SCROLL_MAX)
print(value_cm1)
cv.createTrackbar(trackbar_cm1, title_window, value_cm1, SCROLL_MAX, on_cm1_trackbar)
trackbar_cm2 = 'cm2'
value_cm2 = int(((CM2 - CM2_MIN) / (CM2_MAX - CM2_MIN))*SCROLL_MAX)
cv.createTrackbar(trackbar_cm2, title_window, value_cm2, SCROLL_MAX, on_cm2_trackbar)
trackbar_cm3 = 'cm3'
value_cm3 = int(((CM3 - CM3_MIN) / (CM3_MAX - CM3_MIN))*SCROLL_MAX)
cv.createTrackbar(trackbar_cm3, title_window, value_cm3, SCROLL_MAX, on_cm3_trackbar)
trackbar_cm4 = 'cm4'
value_cm4 = int(((CM4 - CM4_MIN) / (CM4_MAX - CM4_MIN))*SCROLL_MAX)
cv.createTrackbar(trackbar_cm4, title_window, value_cm4, SCROLL_MAX, on_cm4_trackbar)
trackbar_r1 = 'r1'
cv.createTrackbar(trackbar_r1, title_window, SCROLL_MIN, SCROLL_MAX, on_r1_trackbar)
trackbar_r2 = 'r2'
cv.createTrackbar(trackbar_r2, title_window, SCROLL_MIN, SCROLL_MAX, on_r2_trackbar)
trackbar_r3 = 'r3'
cv.createTrackbar(trackbar_r3, title_window, SCROLL_MIN, SCROLL_MAX, on_r3_trackbar)
trackbar_r4 = 'r4'
cv.createTrackbar(trackbar_r4, title_window, SCROLL_MIN, SCROLL_MAX, on_r4_trackbar)
trackbar_r5 = 'r5'
cv.createTrackbar(trackbar_r5, title_window, SCROLL_MIN, SCROLL_MAX, on_r5_trackbar)
on_cm4_trackbar(value_cm4)

# Show some stuff
#on_cm1_trackbar(0)
# Wait until user press some key
cv.waitKey()
