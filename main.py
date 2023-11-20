from ultralytics import YOLO
from tracker import tracker
from persistence import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 加载YOLOv8模型
    model = YOLO('yolov8s.pt')
    tracker.Tracker(model, 'demo.mp4', persist=True, file_mapper=remote_file_mapper.RemoteFileMapper(),
                    data_mapper=remote_data_mapper.RemoteDataMapper()).read()
