# -*- coding: utf-8 -*-
r"""
Created on 18 Mar 2021 18:06:08
@author: jiahuei

python -m tests.benchmark_speed_ws

"""
import logging
import os
import websocket
import json
import numpy as np
from statistics import median, mean
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from time import perf_counter, sleep
from scanner.utils.image import read_image, resize_image, ndarray_to_base64str
from scanner import utils

logger = logging.getLogger(__name__)
result_list = []


def websocket_task(ws_url: str, payload: str, num_frames: int):
    ws = websocket.WebSocket()
    ws.connect(ws_url)
    timings = []
    for i in range(num_frames):
        tic = perf_counter()
        ws.send(payload)
        response = ws.recv()
        assert isinstance(response, str), (
            f"Expected `response` to be of dtype `str`, saw `{type(response)}`"
        )
        assert "doc_points" in response, (
            f"Expected `response` to contain key `doc_points`, saw `{response}`"
        )
        timings.append((tic, perf_counter()))
    ws.close()
    return timings


def log_time(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.extend(result)


def main(
        ws_url,
        use_multiprocessing,
        num_connection=10, num_imgs_per_connection=20,
        max_image_side=480,
        write_to_file=True,
):
    if use_multiprocessing and num_connection > 100:
        print(f"WARNING: Spawning {num_connection} processes !!!")
    configs = [
        ("Concurrency method", f"{'multiprocessing' if use_multiprocessing else 'threading'}"),
        ("Num connections", f"{num_connection}"),
        ("Num images per connection", f"{num_imgs_per_connection}"),
        ("Max image side", f"{max_image_side}"),
        ("Num WebSocket workers", f"{os.getenv('WEBSOCKET_NUM_WORKERS')}"),
    ]
    print("---------------------------------------------------------")
    for x, y in configs:
        print(f"{x}: {y}")
    print("---------------------------------------------------------")

    global result_list
    result_list = []

    # Prepare payload
    test_image = read_image(os.path.join(utils.misc.REPO_DIR, "tests", "data", "test_w8_ben.jpg"))
    test_image, _ = resize_image(test_image, max_side=max_image_side)
    payload = {
        "image": ndarray_to_base64str(test_image),
        "state": [[None, None, True, False, False, True], [None, None, True, False, False, True]] * 4,
    }
    payload = json.dumps(payload)

    if use_multiprocessing:
        pool = Pool(processes=num_connection)
    else:
        pool = ThreadPool(processes=num_connection)

    tic = perf_counter()
    for i in range(num_connection):
        pool.apply_async(
            websocket_task,
            args=(ws_url, payload, num_imgs_per_connection),
            callback=log_time,
        )
    pool.close()
    pool.join()

    elapsed_time = perf_counter() - tic
    total_frames = num_connection * num_imgs_per_connection
    timings = [toc - tic for tic, toc in result_list]
    logs = [
        ("Total images sent", f"{total_frames:,d}"),
        ("Elapsed time (sec)", f"{elapsed_time:.5f}"),
        ("Frames per second (avg)", f"{total_frames / elapsed_time:.3f}"),
        ("Time taken per frame (avg)", f"{elapsed_time / total_frames:.5f}"),
        ("Latency (min)", f"{min(timings):.5f}"),
        ("Latency (max)", f"{max(timings):.5f}"),
        ("Latency (mean)", f"{mean(timings):.5f}"),
        ("Latency (q = 50 %)", f"{median(timings):.5f}"),
        ("Latency (q = 90 %)", f"{np.quantile(timings, .90):.5f}"),
        ("Latency (q = 99 %)", f"{np.quantile(timings, .99):.5f}"),
    ]
    for x, y in logs:
        print(f"{x}: {y}")
    print("---------------------------------------------------------")

    # Write to file
    if write_to_file:
        with open("benchmark_speed_ws.tsv", "a", newline="\n") as f:
            f.write("\n")
            x, y = zip(*configs)
            f.write("\t".join(x) + "\n")
            f.write("\t".join(y) + "\n")
            f.write("\n")
            x, y = zip(*logs)
            f.write("\t".join(x) + "\n")
            f.write("\t".join(y) + "\n")
            f.write("\n")
    sleep(10)


if __name__ == '__main__':
    # websocket.enableTrace(True)
    url = "ws://scanner:5000/detect_doc/real"
    max_connections = 48
    num_imgs_per_connect = 100
    num_workers = int(os.getenv("WEBSOCKET_NUM_WORKERS"))  # 6, 12, 18, 24
    total_imgs = max_connections * num_imgs_per_connect

    # Warm-up ?
    main(url, True, 5, 100, write_to_file=False)

    with open("benchmark_speed_ws.tsv", "a", newline="\n") as f:
        f.write("\n---------------------------------------------------------\n")
    for c in range(6, max_connections + 1, 6):
        # if c // num_workers > 3 or num_workers // c > 2:
        #     continue
        num_imgs = total_imgs // c
        main(url, True, c, num_imgs)
        main(url, False, c, num_imgs)

    # main(url, True, 5, 100)
    # main(url, True, 10, 50)
