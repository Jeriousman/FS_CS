# models
root_dir = "/datasets/pretrained"
cosface = f"{root_dir}/ACC9928.pth"
hopenet = f"{root_dir}/hopenet_robust_alpha1.pkl"
processors = None
dlib = f"{root_dir}/shape_predictor_68_face_landmarks.dat"
inception = f"{root_dir}/pt_inception-2015-12-05-6726825d.pth"
inception_dims = 2048
inception_batch_size = 1

# processor
# processors = ["ID", "FID", "POSE"]

# params
# target_path = None
# source_path = None
# swapped_path = None
execution_threads_fid = 0
execution_threads = 4

