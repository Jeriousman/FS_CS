from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import metrics.globals


def multi_process(path1, path2, process, update=None):
    results = []
    with ThreadPoolExecutor(max_workers=metrics.globals.execution_threads) as executor:
        futures = []
        queue_source = create_queue(path1)
        queue_target = create_queue(path2)
        queue_source_per_future = max(
            len(path1) // metrics.globals.execution_threads, 1
        )
        queue_target_per_future = max(
            len(path2) // metrics.globals.execution_threads, 1
        )

        while not queue_target.empty():
            future = executor.submit(
                process,
                pick_queue(queue_source, queue_source_per_future),
                pick_queue(queue_target, queue_target_per_future),
                update,
            )
            futures.append(future)
        for future in as_completed(futures):
            results.append(future.result())
    return results


def create_queue(paths):
    queue = Queue()
    for frame_path in paths:
        queue.put(frame_path)
    return queue


def pick_queue(queue, queue_per_future):
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues
