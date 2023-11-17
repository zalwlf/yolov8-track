import time
from collections import defaultdict


class Statistics:

    def __init__(self) -> None:
        self.__frame_rate = 1
        # 按先后顺序存储每个检测目标的各帧的目标位置坐标
        self.__track_history = defaultdict(lambda: [])
        # 记录每个检测目标的第一帧, 最后一帧时删除
        self.__target_time = dict()
        # 记录每个检测目标的驻留时长，如果目标丢失后再次出现，则累加时长
        self.__target_total = dict()

    def difference(self, track_ids: list[int], counter: int) -> set[int]:
        # 求差集，获取消失的目标并统计驻留时长
        target_t = set(self.__target_time)
        target_i = set(track_ids)
        diff = target_t - target_i
        for di in diff:
            value = self.__target_time.pop(di, None)
            if self.__target_total[di] != 0:
                self.__target_total[di] += (counter + 1 - value) / self.__frame_rate
                print(f'person {di} update total time : ({counter} + 1 - {value}) / {self.__frame_rate} = {self.__target_total[di]:.2f}s.')
            else:
                self.__target_total[di] = (counter + 1 - value) / self.__frame_rate
                print(f'person {di} total time : ({counter} + 1 - {value}) / {self.__frame_rate} = {self.__target_total[di]:.2f}s.')
        return diff

    def target_time(self) -> dict[int, int]:
        return self.__target_time

    def target_total(self) -> dict[int, int]:
        return self.__target_total

    def track_history(self):
        return self.__track_history

    def frame_rate(self):
        return self.__frame_rate

    def set_frame_rate(self, frame_rate: float) -> None:
        self.__frame_rate = frame_rate
