import cv2
import numpy as np

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


video_filepath = r"/workspace/datasets/UCF101/all/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"


def extract_uio_optflow_caption_feature(video_filepath):
    """让uio模型根据optflow找到对应重要的部位"""
    cap = cv.VideoCapture(video_filepath)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame1 = cap.read()

    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    counter = 0
    length_snippets = 16  # 一个snippet中包含的帧的数量
    start_frame = max(int(length_snippets / 2 - 1), 0)-1
    # frame_indexes = [i for i in range(start_frame, total_frame, length_snippets)]
    index1 = [i for i in range(start_frame, total_frame, length_snippets)]
    index2 = [i for i in range(start_frame - 1, total_frame, length_snippets)]
    index3 = [i for i in range(start_frame - 2, total_frame, length_snippets)]
    index4 = [i for i in range(start_frame - 3, total_frame, length_snippets)]
    while True:
        ret, frame2 = cap.read()
        if frame2 is None:
            break
        if counter in index4:
            pre_frame_4 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            # frames4.append(next)
        if counter in index3:
            pre_frame_3 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            # frames3.append(next)
        if counter in index2:
            pre_frame_2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            # frames2.append(next)
        if counter in index1:
            gray = None
            pre_frame_1 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            # frames1.append(next)
            for pre_frame in [pre_frame_2, pre_frame_3, pre_frame_4]:
                # 返回一个两通道的光流向量，实际上是每个点的像素位移值
                flow = cv.calcOpticalFlowFarneback(pre_frame, pre_frame_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # 笛卡尔坐标转换为极坐标，获得极轴和极角
                mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

                hsv[..., 0] = ang * 180 / np.pi / 2  # 角度
                hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
                bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                #  移动的位置和移动的速度都有参考价值，直接用二值化可能不太好，但是mask必须二值化
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                th2, gray = cv.threshold(gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
                # th2, gray = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)
                # assert np.sum(gray)
                print(np.sum(gray))

                plt.imshow(gray, cmap="gray")
                plt.show()
                # plt.imshow(frame2)
                # plt.show()
                if np.sum(gray) != 0:
                    break
            if gray is None:
                raise RuntimeError("Can cal opt-flow")
            # cv.imshow('frame2',bgr)
            # cv.imshow("frame1", frame2)
        # k = cv.waitKey(30) & 0xff
        counter += 1
        # if k == 27:
        #     break
        # elif k == ord('s'):
        #     cv.imwrite('opticalfb.png',frame2)
        #     cv.imwrite('opticalhsv.png',bgr)
    cap.release()

if __name__ == '__main__':
    extract_uio_optflow_caption_feature(video_filepath)