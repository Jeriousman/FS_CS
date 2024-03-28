import os
import psutil
import metrics.globals

from pathlib import Path

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
# {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}


def has_image_extension(image_path):
    return image_path.lower().endswith(("png", "jpg", "jpeg", "webp"))


def check_file_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)


def update_status(message, scope="metrics.CORE"):
    print(f"[{scope}] {message}")


def get_image_paths(path):
    path = Path(path)
    return sorted(
        [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
    )


def update_progress(progress, num=1):
    for i in range(num):
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
        progress.set_postfix(
            {
                "memory_usage": "{:.2f}".format(memory_usage).zfill(5) + "GB",
                # "execution_threads": metrics.globals.execution_threads,
            }
        )
        progress.refresh()
        progress.update()
