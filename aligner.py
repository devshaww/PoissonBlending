import cv2
import numpy as np

source_name = "./images/source2_1.jpg"
target_name = "./images/bg1.jpg"
output_source_name = "aligned_mask.jpg"
aligned_mask_name = "new_s.jpg"


class Aligner:
    lsPointsChoose = []
    tpPointsChoose = []
    pointsCount = 0
    source_im = None
    target_im = None
    mask = None

    def __init__(self, source, target, out_source, out_mask):
        self.out_source = out_source
        self.out_mask = out_mask
        self.source_im = cv2.imread(source)
        self.source_copy = self.source_im.copy()
        self.target_im = cv2.imread(target)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pointsCount = self.pointsCount + 1
            print('pointsCount:', self.pointsCount)
            print(x, y)
            self.lsPointsChoose.append([x, y])
            self.tpPointsChoose.append((x, y))
            for i in range(len(self.tpPointsChoose) - 1):
                cv2.line(self.source_im, self.tpPointsChoose[i], self.tpPointsChoose[i + 1], (0, 0, 255), 2)
            cv2.imshow('src', self.source_im)

        if event == cv2.EVENT_RBUTTONDOWN:
            unfilled_mask = np.zeros(self.source_im.shape, np.uint8)
            pts = np.array([self.lsPointsChoose], np.int32)
            unfilled_mask = cv2.polylines(unfilled_mask, [pts], True, (255, 255, 255))
            self.mask = cv2.fillPoly(unfilled_mask, [pts], (255, 255, 255))
            cv2.imshow('src', self.target_im)

        if event == cv2.EVENT_MBUTTONDOWN:
            tx = x
            ty = y

            nonzero = np.nonzero(self.mask)
            x = nonzero[1]
            y = nonzero[0]
            y1 = np.amin(y) - 1
            y2 = np.amax(y) + 1
            x1 = np.amin(x) - 1
            x2 = np.amax(x) + 1

            yind = (np.arange(y1, y2+1))
            yind2 = yind - np.amax(y) + np.round(ty)
            xind = (np.arange(x1, x2+1))
            xind2 = xind - np.round(np.mean(x)) + np.round(tx)

            y = y - np.amax(y) + np.round(ty)
            x = x - np.round(np.mean(x)) + np.round(tx)
            self.mask = np.zeros(self.target_im.shape)
            for i, j in zip(y, x):
                self.mask[int(i)][int(j)] = 255
            cv2.imwrite(self.out_mask, self.mask, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            new_s = np.zeros(self.target_im.shape)
            i = 0
            while i < yind2.shape[0]:
                j = 0
                while j < xind2.shape[0]:
                    new_s[int(yind2[i])][int(xind2[j])] = self.source_copy[int(yind[i])][int(xind[j])]
                    j += 1
                i += 1

            cv2.imwrite(self.out_source, new_s, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print("Close the window to quit.")

    def draw_aligned_mask_and_source(self):
        cv2.namedWindow('src')
        cv2.setMouseCallback('src', self.on_mouse)
        cv2.imshow('src', self.source_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    am = Aligner("./images/source1.jpg", "./images/target1.jpg", "aligned_source.jpg", "aligned_mask.jpg")
    cv2.namedWindow('src')
    cv2.setMouseCallback('src', am.on_mouse)
    cv2.imshow('src', am.source_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



