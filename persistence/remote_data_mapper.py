from typing import Any
from persistence.data_persistence import DataMapper
from util import http_client
from constants.http_constants import *


class RemoteDataMapper(DataMapper):

    def save_first_frame(self, tracker_id: int, counter: int, frame_rate: float, file_res: Any, dt: Any) -> Any:
        return http_client.data_post(URI + URI_PERSON_FIRST,
                                     data={'fileUrl': file_res, 'counter': counter, 'frameRate': frame_rate,
                                           'personId': tracker_id})

    def save_second_frame(self, tracker_ids: set[int], counter: int, frame_rate: float, duration: [int, int],
                          dt: Any) -> Any:
        return http_client.data_post(URI + URI_PERSON_MORE,
                                     data={'personIds': tracker_ids, 'counter': counter, 'frameRate': frame_rate,
                                           'duration': duration, 'dt': dt})
