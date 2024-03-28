import metrics.globals

from metrics.common import get_metric_modules
from omegaconf import OmegaConf

from .utilites import update_status, get_image_paths

arg = OmegaConf.load("config/config.yaml")

# models
metrics.globals.cosface = arg.models.ID.cosface.model_path
metrics.globals.hopenet = arg.models.POSE.hopenet.model_path
metrics.globals.dlib = arg.models.EXPRESSION.dlib.model_path
metrics.globals.inception = arg.models.FID.inception.model_path
metrics.globals.inception_dims = arg.models.FID.inception.dims
metrics.globals.inception_batch_size = arg.models.FID.inception.batch_size

# processor
metrics.globals.processors = arg.processors

# params
metrics.globals.target_path = arg.params.target_path
metrics.globals.source_path = arg.params.source_path
metrics.globals.swapped_path = arg.params.swapped_path
metrics.globals.execution_threads = arg.params.execution_threads


def start():
    pass


def run():
    # images
    for metric in get_metric_modules(metrics.globals.processors):
        if not metric.pre_start():
            return

    target_paths = get_image_paths(metrics.globals.target_path)
    source_paths = get_image_paths(metrics.globals.source_path)
    swapped_paths = get_image_paths(metrics.globals.swapped_path)

    for metric in get_metric_modules(metrics.globals.processors):
        update_status("Progressing...", metric.NAME)
        metric.start_process(source_paths, target_paths, swapped_paths)
        metric.clear_process()
        pass
