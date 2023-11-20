from io import BufferedReader
from typing import Any
from persistence.file_persistence import FileMapper
from constants.http_constants import *
from util import http_client


class RemoteFileMapper(FileMapper):

    def save_file(self, file: BufferedReader) -> Any:
        return http_client.file_upload(URI + URI_PIC_UPLOAD, files={'file': file})
