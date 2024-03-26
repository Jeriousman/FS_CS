import torch
import threading

from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import metrics.globals
from metrics.utilites import update_status, check_file_exists, update_progress
from metrics.multiprocess import multi_process
from .model import sphere

NAME = "metrics.ID"
COSFACE = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()


def get_cosface(device):
    global COSFACE

    with THREAD_LOCK:
        if COSFACE is None:
            model = sphere()
            model.load_state_dict(torch.load(metrics.globals.cosface))
            model.eval()
            model.to(device)
            COSFACE = model

    return COSFACE


def clear_process():
    global COSFACE
    COSFACE = None


def pre_start():
    if not check_file_exists(metrics.globals.cosface):
        update_status("Model Path is not Exsit", NAME)
    return True


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


def cosface(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_cosface(device)
    transform = transforms.Compose(
        [
            transforms.Resize((112, 92)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_vector = model(input_tensor)
    return feature_vector


def calc_sim_score(source_face, swapped_face):
    with THREAD_SEMAPHORE:
        feature_vector1 = cosface(source_face)
        feature_vector2 = cosface(swapped_face)
        similarity_score = (
            cosine_sim(feature_vector1, feature_vector2).cpu().numpy()[0][0]
        )
        return similarity_score


def calc_sim_scores(source_faces, swapped_faces, update=None):
    total_score = None

    for source_face, swapped_face in zip(source_faces, swapped_faces):
        score = calc_sim_score(source_face, swapped_face)
        if not total_score:
            total_score = score
        else:
            total_score += score
    if update:
        update(len(source_faces))
    return total_score


def start_process(source_faces, _, swapped_faces):
    progress_bar_format = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    total = len(swapped_faces)
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
            source_faces, swapped_faces, calc_sim_scores, call_back_func
        )
    avg_score = sum(total_scores) / total
    # print("ID SCORE: ", avg_score)
    return avg_score