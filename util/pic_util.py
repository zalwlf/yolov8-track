from io import BytesIO
from io import BufferedReader
import time
import cv2


def frame_to_pic(frame: cv2.UMat):
    ret, frame_new = cv2.imencode('.jpg', frame)
    str_encode = frame_new.tobytes()  # 将array转化为二进制类型
    temp = BytesIO(str_encode)  # 转化为_io.BytesIO类型
    now = int(time.time())
    temp.name = f'{now}.jpg'
    file = BufferedReader(temp)  # 转化为_io.BufferedReader类型
    files = {'file': file}
    return files
