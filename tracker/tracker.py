from typing import Any
import cv2
from ultralytics import YOLO
import numpy as np
from util import *


class Tracker:

    """
        构造
        :param model yolov8模型
        :param video_path 视频路径
    """
    def __init__(self, model: YOLO, video_path: str = '', **kwargs) -> None:
        self.__model = model
        self.__video_path = video_path
        self.__cap = cv2.VideoCapture(video_path)
        self.__statistics = target_statistics.Statistics()
        self.__statistics.set_frame_rate(self.__cap.get(cv2.CAP_PROP_FPS))

        self.__classes = 0
        if 'classes' in kwargs:
            self.__classes = kwargs['classes']
        self.__conf = 0.3
        if 'conf' in kwargs:
            self.__conf = kwargs['conf']
        self.__persist = True
        if 'persist' in kwargs:
            self.__persist = kwargs['persist']

    def read(self) -> None:
        counter = 0
        while self.__cap.isOpened():
            # 读取一帧图像
            success, frame = self.__cap.read()
            if success:
                files = pic_util.frame_to_pic(frame)
                file_url = None
                track_ids, results = self.track(frame, counter)
                # 遍历该帧的所有目标
                for track_id, box in zip(track_ids, results[0].boxes.data):
                    # 判断是否有记录，没有则新增
                    if track_id not in self.__statistics.target_time():
                        self.__statistics.target_time()[track_id] = counter
                    if track_id not in self.__statistics.target_total():
                        self.__statistics.target_total()[track_id] = 0
                        # if not file_url:
                        #     file_url = http_client.file_upload(Constants.URI + Constants.URI_PIC_UPLOAD,
                        #                                        files=files)
                        # http_client.data_post(Constants.URI + Constants.URI_PERSON_FIRST,
                        #                       data={'fileUrl': file_url, 'deviceId': device_id, 'personId': track_id})

                    # 绘制该目标的矩形框
                    self.box_label(frame, box, '#' + str(track_id) + ' person', (167, 146, 11))
                    # 绘制追踪线
                    if self.__persist:
                        # 得到该目标矩形框的中心点坐标(x, y)
                        x1, y1, x2, y2 = box[:4]
                        x = (x1 + x2) / 2
                        y = (y1 + y2) / 2
                        # 提取出该ID的以前所有帧的目标坐标，当该ID是第一次出现时，则创建该ID的字典
                        track = self.__statistics.track_history()[track_id]
                        track.append((float(x), float(y)))  # 追加当前目标ID的坐标
                        # 只有当track中包括两帧以上的情况时，才能够比较前后坐标的先后位置关系
                        if len(track) > 30:  # 在90帧中保留90个追踪点
                            track.pop(0)
                        # 绘制追踪线
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                cv2.imshow("YOLOv8", frame)  # 显示标记好的当前帧图像
                counter += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):  # 'q'按下时，终止运行
                    break

            else:  # 视频播放结束时退出循环
                break

    def track(self, frame: cv2.UMat, counter: int) -> tuple[list[int], Any]:
        """
        在帧上运行YOLOv8跟踪，持续追踪帧间的物体
        classes=0 只检测人体，可选: classes=[0,2,3]检测多种目标
        conf为0.3表示只检测置信值大于0.3的目标。
        persist为True表示保留跟踪信息
        """
        results = self.__model.track(frame, classes=self.__classes, conf=self.__conf, persist=self.__persist)
        track_ids = results[0].boxes.id.int().cpu().tolist()
        self.__statistics.difference(track_ids, counter)
        return track_ids, results

    """
        在帧中绘制矩形框及标记内容
        :param s_frame 视频帧
        :param p_box yolov8追踪box
        :param label 标记内容
        :param color 矩形框颜色 默认#000 灰色
        :param label_color 文本颜色 默认#FFF 白色
    """

    def box_label(self, s_frame, p_box, label='', color=(0, 0, 0), label_color=(255, 255, 255)) -> None:
        # 目标矩形框的左上角和右下角坐标
        p1, p2 = (int(p_box[0]), int(p_box[1])), (int(p_box[2]), int(p_box[3]))
        # 绘制矩形框
        cv2.rectangle(s_frame, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
        if label:
            # 得到要书写的文本的宽和长，用于给文本绘制背景色
            w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
            # 确保显示的文本不会超出图片范围
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(s_frame, p1, p2, color, -1, cv2.LINE_AA)  # 填充颜色
            # 书写文本
            cv2.putText(s_frame,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        2 / 3,
                        label_color,
                        thickness=1,
                        lineType=cv2.LINE_AA)
