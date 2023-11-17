from io import BufferedReader
from typing import Any


class FileMapper:

    def save_file(self, file: BufferedReader) -> Any:
        """
            文件流持久化方法
            :param file 由视频帧转换成的文件流
                   参见 @see also util.pic_util.frame_to_pic
        """
        pass


class DefaultFileMapper(FileMapper):

    def save_file(self, file: BufferedReader) -> Any:
        return super().save_file(file)
