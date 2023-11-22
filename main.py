from ultralytics import YOLO
from tracker import tracker
from persistence import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 加载YOLOv8模型
    model = YOLO('yolov8s.pt')
    # http远程推送结果并持久化 接口配置：@see also http_constants
    # tracker.Tracker(model, 'demo.mp4', persist=True, file_mapper=remote_file_mapper.RemoteFileMapper(),
    #                data_mapper=remote_data_mapper.RemoteDataMapper()).read()
    # 不推送结果
    tracker.Tracker(model, 'demo_like_dog.mp4', imshow=False, file_mapper=file_persistence.DefaultFileMapper(),
                    data_mapper=data_persistence.DefaultDataMapper()).read()
