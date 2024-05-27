import metrics.globals
import torch, threading, dlib, cv2, sys
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm

from metrics.common import cals_euclidean_avg
from metrics.utilites import update_status, check_file_exists, update_progress
from metrics.multiprocess import multi_process
import numpy as np

NAME = "metrics.EXPRESSION"
DLIB_DETECTOR = None
DLIB_PREDICTOR = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()


def get_detector():
    global DLIB_DETECTOR

    with THREAD_LOCK:
        if DLIB_DETECTOR is None:
            DLIB_DETECTOR = dlib.get_frontal_face_detector()

    return DLIB_DETECTOR


def get_predictor():
    global DLIB_PREDICTOR

    with THREAD_LOCK:
        if DLIB_PREDICTOR is None:
            DLIB_PREDICTOR = dlib.shape_predictor(metrics.globals.dlib)

    return DLIB_PREDICTOR


def clear_process():
    global DLIB_DETECTOR
    global DLIB_PREDICTOR

    DLIB_DETECTOR = None
    DLIB_PREDICTOR = None


def pre_start():
    if not check_file_exists(metrics.globals.dlib):
        update_status("Model Path is not Exsit", NAME)
    return True


def detect_face(image):
    model = get_detector()
    faces = model(image)
    return faces


def extract_lmrk(image):
    faces = detect_face(image)
    if len(faces) > 1:
        update_status("More than one face has been extracted.", NAME)
        sys.exit()
    model = get_predictor()
    lmrk = model(image, faces[0])
    lmrk = [[points.x, points.y] for points in lmrk.parts()]
    return lmrk


def calc_expression_score(target_face, swapped_face):
    with THREAD_SEMAPHORE:
        lmrk1 = extract_lmrk(target_face)
        lmrk1 = torch.tensor(lmrk1, dtype=torch.float32)

        lmrk2 = extract_lmrk(swapped_face)
        lmrk2 = torch.tensor(lmrk2, dtype=torch.float32)
        euclidean_avg = cals_euclidean_avg(lmrk1, lmrk2, dim=1).cpu().numpy()
        return euclidean_avg


def calc_expression_scores(target_faces, swapped_faces, update=None):
    total_score = None
    
    for target_face, swapped_face in zip(target_faces, swapped_faces):
        
        # tensor to numpy
        target = (target_face + 1) / 2
        swapped = (swapped_face + 1) /2
        topil = ToPILImage()
        targettopilimage = topil(target)
        swappedtopilimage = topil(swapped)
        
        # target_face_np = (target_face.cpu().numpy() * 255).astype(np.uint32)
        # swapped_face_np = (swapped_face.cpu().numpy() * 255).astype(np.uint32)

        # target_face_np = np.transpose(target_face_np, (1, 2, 0))
        # swapped_face_np = np.transpose(swapped_face_np, (1, 2, 0))

        target_face_np = np.array(targettopilimage)
        swapped_face_np = np.array(swappedtopilimage)
        # cv2.imwrite("/workspace/testtar.jpg", target_face_np)
        # cv2.imwrite("/workspace/testswa.jpg", swapped_face_np)


        # cv2.imwrite("/workspace/testtar_rgb.jpg", cv2.cvtColor(target_face_np, cv2.COLOR_RGB2BGR))

        
        score = calc_expression_score(target_face_np, swapped_face_np)
        if not total_score:
            total_score = score
        else:
            total_score += score
    if update:
        update(len(target_faces))
    return total_score


def start_process(_, target_faces, swapped_faces):
    progress_bar_format = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    total = len(target_faces)
    
    with tqdm(
        total=total,
        desc="Processing",
        unit="frame",
        dynamic_ncols=True,
        bar_format=progress_bar_format,
    ) as progress:

        def call_back_func(num):
            update_progress(progress, num)
        total_scores = multi_process(
            target_faces, swapped_faces, calc_expression_scores, call_back_func
        )
    avg_score = sum(total_scores) / total
    # print("EXPRESSION SCORE: ", avg_score)
    return avg_score