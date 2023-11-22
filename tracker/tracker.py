from typing import Any
import cv2
from ultralytics import YOLO
import numpy as np
from util import *
from persistence import *
import keyboard

bool_esc = False


def on_press(event):
    if event.name == 'q':
        global bool_esc
        bool_esc = True


def box_label(s_frame, p_box, label='', color=(0, 0, 0), label_color=(255, 255, 255)) -> None:
    """
        在视频帧中绘制矩形框及标记文本
        :param s_frame 视频帧
        :param p_box 由yolov8解析的当前视频帧中目标的box信息
        :param label 标记文本内容
        :param color 矩形框的颜色
        :param label_color 标记文本颜色
    """
    # 目标矩形框的左上角和右下角坐标
    p1, p2 = (int(p_box[0]), int(p_box[1])), (int(p_box[2]), int(p_box[3]))
    # 绘制矩形框
    cv2.rectangle(s_frame, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
    if label:
        # 得到要书写的文本的宽和长，用于给文本绘制背景色
        w, h = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        # 确保显示的文本不会超出图片范围
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(s_frame, p1, p2, color, -1, cv2.LINE_AA)  # 填充颜色
        # 书写文本
        cv2.putText(s_frame,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    label_color,
                    thickness=1,
                    lineType=cv2.LINE_AA)


class Tracker:

    def __init__(self, model: YOLO, video_path: str = '', imshow: bool = True, **kwargs) -> None:
        """
            构造
            :param model yolov8模型
            :param video_path 视频路径
            :param imshow 是否开启窗口实时展示视频
            :param kwargs 使用yolov8目标追踪时，部分必要的参数
        """
        self.__model = model
        self.__video_path = video_path
        self.__imshow = imshow
        self.__cap = cv2.VideoCapture(video_path)
        self.__statistics = target_statistics.Statistics()
        self.__statistics.set_frame_rate(self.__cap.get(cv2.CAP_PROP_FPS))
        # classes 检测对象类别 =0只检测人体，可选: classes=[0,2,3]检测多种目标
        self.__classes = kwargs.get('classes') or 0
        # conf 置信值 为0.3表示只检测置信值大于0.3的目标。
        self.__conf = kwargs.get('conf') or 0.3
        # file_mapper 检测到目标时，会将此帧画面转为图片流传递到FileMapper类并调用save_file方法，可通过继承并重写的方式实现自定义处理
        self.__file_mapper = kwargs.get('file_mapper') or file_persistence.DefaultFileMapper()
        # data_mapper 检测到目标与目标消失时，会将部分信息传递到DataMapper下并调用相关方法，可通过继承并重写做自定义处理
        self.__data_mapper = kwargs.get('data_mapper') or data_persistence.DefaultDataMapper()
        # data_mapper透传参数
        self.__dt = kwargs.get('dt') or None
        # 是否监听按键q退出，当且仅当不做实时画面展示时，此参数为True,则在控制台监听按键q，展示画面时默认监听按键q
        self.__key_esc = kwargs.get('key_esc') or False

    def read(self) -> None:
        """
            核心功能入口
            读取视频并进行目标追踪
        """
        counter = 0
        while self.__cap.isOpened():
            # 读取一帧图像
            success, frame = self.__cap.read()
            if success:
                file = pic_util.frame_to_pic(frame)
                file_res = None
                track_ids, results = self.track(frame, counter)
                if track_ids is None or not track_ids:
                    counter += 1
                    is_exit = self.show_exit(frame)
                    if is_exit:
                        break
                    continue

                # 遍历该帧的所有目标
                for track_id, box in zip(track_ids, results[0].boxes.data):
                    # 判断是否有记录，没有则新增
                    if track_id not in self.__statistics.target_time():
                        self.__statistics.target_time()[track_id] = counter
                    if track_id not in self.__statistics.target_total():
                        self.__statistics.target_total()[track_id] = 0
                    if not file_res:
                        file_res = self.__file_mapper.save_file(file)
                        self.__data_mapper.save_first_frame(track_id, counter, self.__statistics.frame_rate(),
                                                            file_res,
                                                            self.__dt)
                    # 绘制追踪线
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

                    if self.__imshow:
                        # 绘制该目标的矩形框color=BGR
                        box_label(frame, box, '#' + str(track_id) + ' person', (0, 102, 255))
                        # 绘制追踪线
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                counter += 1
                is_exit = self.show_exit(frame)
                if is_exit:
                    break

            else:  # 视频播放结束时退出循环
                break
        self.__cap.release()
        cv2.destroyAllWindows()

    def track(self, frame: cv2.UMat, counter: int) -> tuple[list[int], Any]:
        """
        在帧上运行YOLOv8跟踪，持续追踪帧间的物体
        classes 检测对象类别 =0只检测人体，可选: classes=[0,2,3]检测多种目标
        conf 置信值 为0.3表示只检测置信值大于0.3的目标。
        persist 为True表示保留跟踪信息
        :param frame 视频帧
        :param counter 当前帧的帧数
        :return 所有追踪id, 追踪结果
        """
        results = self.__model.track(frame, classes=self.__classes, conf=self.__conf, persist=True)
        track_ids = None
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            diff = self.__statistics.difference(track_ids, counter)
            if len(diff) != 0:
                self.__data_mapper.save_second_frame(diff, counter, self.__statistics.frame_rate(),
                                                     self.__statistics.target_total(), self.__dt)
        return track_ids, results

    def show_exit(self, frame: cv2.UMat) -> bool:
        """
        视频流实时展示与监听退出
        :param frame: 帧
        :return: bool 是否退出
        """
        if self.__imshow:
            cv2.imshow("YOLOv8", frame)  # 显示标记好的当前帧图像
            if cv2.waitKey(1) & 0xFF == ord("q"):  # 'q'按下时，终止运行
                print('exit process.')
                return True
        elif self.__key_esc:
            keyboard.on_press(on_press)
            global bool_esc
            if bool_esc:
                print('exit process.')
                return True
        return False
