# -*- coding: utf-8 -*-
"""
Created on 06 Nov 2020 22:03:35
@author: jiahuei
"""
import logging
import cv2
import numpy as np
from imutils import video as video_utils

logger = logging.getLogger(__name__)


class WebCam:
    def __init__(self, cam_index=0):
        # start the video stream thread
        logger.info("Starting video stream")
        self.vs = video_utils.VideoStream(src=cam_index)

    def vid_stream(self):
        webcam = self.vs.start()
        while webcam.stopped is False:
            yield webcam.read()

            # cv2.waitKey() returns a 32 bit integer value (might be dependent on the platform).
            # The key input is in ASCII which is an 8 bit integer value
            # So you only care about these 8 bits and want all other bits to be 0
            # https://stackoverflow.com/a/39203128
            key = cv2.waitKey(delay=1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        logger.info("Ending video stream")
        cv2.destroyAllWindows()
        webcam.stop()


def read_video_file(path):
    vid = video_utils.FileVideoStream(path=path).start()
    all_frames = []
    while vid.running():
        frame = vid.read()
        if frame is None:
            continue
        all_frames.append(frame)
    vid.stop()
    return all_frames


def write_video_file(path, frame_list):
    assert isinstance(frame_list, (list, tuple)), \
        f"Expected `frame_list` of type `list` or `tuple`, saw {type(frame_list)}"
    if len(frame_list) == 0:
        return
    assert all(isinstance(_, np.ndarray) for _ in frame_list), \
        f"`frame_list` must contain only `np.ndarray`"
    assert len(set(_.shape for _ in frame_list)) == 1, \
        f"All array in `frame_list` must have the same shape."
    assert frame_list[0].ndim == 3 and frame_list[0].shape[2] == 3, \
        f"All array in `frame_list` must have shape (M, N, 3), saw {frame_list[0].shape}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid = cv2.VideoWriter(path, fourcc, round(30), frame_list[0].shape[:2][::-1])
    for frame in frame_list:
        vid.write(frame)
    vid.release()
