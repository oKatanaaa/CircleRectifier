import cv2
import numpy as np


def find_circles(img, verbose=True, inner=False):
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if inner:
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 2, param1=20, param2=120, minRadius=10, maxRadius=0)
    else:
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 2, param1=1, param2=170, minRadius=100, maxRadius=0)

    circles = np.uint16(np.around(circles))
    if verbose:
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    mean_values = np.uint16(np.around(np.mean(circles[0, :], axis=0)))
    cv2.circle(cimg, (mean_values[0], mean_values[1]), mean_values[2], (255, 0, 0), 2)
    cv2.circle(cimg, (mean_values[0], mean_values[1]), 2, (0, 0, 255), 3)
    # plt.imshow(cimg)
    # plt.show()

    return mean_values


def bboxes():
    img = cv2.imread('DSC00131.JPG', 0)
    img = cv2.medianBlur(img, 25)
    canny = cv2.Canny(img, 0, 250)

    _, contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 5)

    cv2.imwrite(f'01.png', img)


def build_map(Ws, Hs, Wd, Hd, R1, R2, Cx, Cy):
    map_x = np.zeros((Hd, Wd), np.float32)
    map_y = np.zeros((Hd, Wd), np.float32)
    for y in range(0, int(Hd - 1)):
        for x in range(0, int(Wd - 1)):
            r = (float(y) / float(Hd)) * (R2 - R1) + R1
            theta = (float(x) / float(Wd)) * 2.0 * np.pi
            xS = Cx + r * np.sin(theta)
            yS = Cy + r * np.cos(theta)
            map_x.itemset((y, x), int(xS))
            map_y.itemset((y, x), int(yS))
    return map_x, map_y


def unwarp(img, xmap, ymap):
    output = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
    return output


if __name__ == "__main__":
    img = cv2.imread('1.png', 0)
    original_img = cv2.imread('1.png', cv2.IMREAD_UNCHANGED)
    img = cv2.medianBlur(img, 15)

    x, y, r = find_circles(img, True, True)
    cx = x
    cy = y
    r1 = r
    r2 = 352 / 2

    Ws = 352.0
    Hs = 352.0
    Wd = int(2.0 * ((r1 + r2) / 2) * np.pi)
    Hd = int(r2 - r1)

    xmap, ymap = build_map(Ws, Hs, Wd, Hd, r1, r2, cx, cy)
    result = unwarp(original_img, xmap, ymap)
    print(result.shape)
    cv2.imwrite('res.png', result)

    # cut circle
    # x, y, r = find_circles(img, True)
    # cutted_img = original_img[y - r: y + r, x - r: x + r]
    # cv2.imwrite('1.png', cutted_img)

    # cut all image
