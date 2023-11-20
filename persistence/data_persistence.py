from typing import Any


class DataMapper:

    def save_first_frame(self, tracker_id: int, counter: int, frame_rate: float, file_res: Any, dt: Any) -> Any:
        """
            追踪到新增的目标信息持久化方法
            :param tracker_id 追踪到的新增的目标id
            :param counter 当前追踪到该新增目标时的帧数
            :param frame_rate 当前视频的帧率
            :param file_res 当前帧转换的图片信息
            :param dt 透传参数
        """
        pass

    def save_second_frame(self, tracker_ids: set[int], counter: int, frame_rate: float, duration: [int, int],
                          dt: Any) -> Any:
        """
            追踪目标消失时的信息持久化方法
            :param tracker_ids 目标id
            :param counter 目标消失时的帧数
            :param frame_rate 当前视频的帧率
            :param duration 驻留时长(毫秒)
            :param dt 透传参数
        """
        pass


class DefaultDataMapper(DataMapper):

    def save_first_frame(self, tracker_id: int, counter: int, frame_rate: float, file_res: Any, dt: Any) -> Any:
        return super().save_first_frame(tracker_id, counter, frame_rate, file_res, dt)

    def save_second_frame(self, tracker_ids: set[int], counter: int, frame_rate: float, duration: [int, int],
                          dt: Any) -> Any:
        return super().save_second_frame(tracker_ids, counter, frame_rate, duration, dt)
