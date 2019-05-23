import cv2
import matplotlib.pyplot as plt
import numpy as np


class CircleFinder:

    @staticmethod
    def find_outer_circle(img, verbose=False):
        outer = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 2, param1=1, param2=170, minRadius=100, maxRadius=0)
        outer = np.uint16(np.around(outer))
        mean_values_outer = np.uint16(np.around(np.mean(outer[0, :], axis=0)))

        if verbose:
            cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            for i in outer[0, :]:
                # draw the outer circle
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.circle(cimg, (mean_values_outer[0], mean_values_outer[1]), mean_values_outer[2], (255, 0, 0), 2)
            cv2.circle(cimg, (mean_values_outer[0], mean_values_outer[1]), 2, (0, 0, 255), 3)
            plt.imshow(cimg)
            plt.show()

        return mean_values_outer

    @staticmethod
    def find_inner_circle(img, verbose=True, param=30):
        inner = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 2, param1=param, param2=120, minRadius=10, maxRadius=0)

        if (inner is None or np.all(inner == 0) or inner.shape[1] > 1) and param > 0:
            mean_values_inner = CircleFinder.find_inner_circle(img, True, param - 1)
            return mean_values_inner

        inner = np.uint16(np.around(inner))
        mean_values_inner = np.uint16(np.around(np.mean(inner[0, :], axis=0)))

        if verbose:
            cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            for i in inner[0, :]:
                # draw the outer circle
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.circle(cimg, (mean_values_inner[0], mean_values_inner[1]), mean_values_inner[2], (255, 0, 0), 2)
            cv2.circle(cimg, (mean_values_inner[0], mean_values_inner[1]), 2, (0, 0, 255), 3)
            plt.imshow(cimg)
            plt.show()

        return mean_values_inner

    @staticmethod
    def cut_circle(img, mean_values_outer):
        x = mean_values_outer[0]
        y = mean_values_outer[1]
        r = mean_values_outer[2]

        cutted_img = img[y - r: y + r, x - r: x + r]
        return cutted_img

    def setup_marker(self):
        pass


class ImageTransformer:

    def build_transformation_map(self, Ws, Hs, Wd, Hd, R1, R2, Cx, Cy):
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

    def unwarp(self, img, xmap, ymap):
        output = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
        return output

    def transform_image(self, img_name):
        original_img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(img_name, 0)
        img = cv2.medianBlur(img, 15)

        outer = CircleFinder.find_outer_circle(img)

        original_img = CircleFinder.cut_circle(original_img, outer)
        img = CircleFinder.cut_circle(img, outer)

        inner = CircleFinder.find_inner_circle(img)

        Cx = inner[0]
        Cy = inner[1]
        Ri = inner[2]
        Ro = outer[2]
        Ws = img.shape[1]
        Hs = img.shape[0]
        Wd = int(2 * np.pi * ((Ri + Ro) / 2))
        Hd = Ro - Ri

        xmap, ymap = self.build_transformation_map(Ws, Hs, Wd, Hd, Ri, Ro, Cx, Cy)
        result = self.unwarp(original_img, xmap, ymap)
        return result


if __name__ == "__main__":
    transformer = ImageTransformer()
    res = transformer.transform_image('DSC00131.JPG')
    cv2.imwrite('res1.png', res)

    # cut circle
    # x, y, r = find_circles(img, True)
    # cutted_img = original_img[y - r: y + r, x - r: x + r]
    # cv2.imwrite('1.png', cutted_img)

    # cut all image
