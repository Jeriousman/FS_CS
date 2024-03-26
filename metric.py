import metrics.globals
import torch, threading, cv2, sys

from tqdm import tqdm
from metrics.common import cals_euclidean_avg
from metrics.utilites import update_status, check_file_exists, update_progress, get_image_paths, update_status
from metrics.multiprocess import multi_process

from metrics import common


def start():
    pass


def run(source, target, swapped, processors):
    for metric in common.get_metric_modules(processors):
        if not metric.pre_start():
            return
        
    result = {}
    for metric in common.get_metric_modules(processors):
        update_status("Progressing...", metric.NAME)
        metric_score = metric.start_process(source, target, swapped)
        result[metric.NAME] = (metric_score)
        metric.clear_process()
        pass

    return result