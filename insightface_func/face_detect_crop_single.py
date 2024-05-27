from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
import cv2
from insightface.model_zoo import model_zoo
from insightface.utils import face_align

__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        self.models = {}
        root = os.path.expanduser(root)  ##expanduser 로 전체절대경로를 뽑아낸다
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))  ##경로안의 onnx파일을 모두 찾는다
        onnx_files = sorted(onnx_files)  ##onnx파일을 순서대로나열한다
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:  ##onnx_file selfgen 파일을 찾으면 다음 for loop으로 넘어간다 (처음으로)
                #print('ignore:', onnx_file)
                continue
            model = model_zoo.get_model(onnx_file) ##아니면 모델주에서 모델 fetch해와라 
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size) ##what is ctx_id?
            else:
                model.prepare(ctx_id)

    def get(self, img, crop_size, max_num=0):
        bboxes, kpss = self.det_model.detect(img,    ##kpss = landmarks keypoints
                                             threshold=self.det_thresh,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return None
        # ret = []
        # for i in range(bboxes.shape[0]):
        #     bbox = bboxes[i, 0:4]
        #     det_score = bboxes[i, 4]
        #     kps = None
        #     if kpss is not None:
        #         kps = kpss[i]
        #     M, _ = face_align.estimate_norm(kps, crop_size, mode ='None') 
        #     align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
        # for i in range(bboxes.shape[0]):
        #     kps = None
        #     if kpss is not None:
        #         kps = kpss[i]
        #     M, _ = face_align.estimate_norm(kps, crop_size, mode ='None') 
        #     align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)

        det_score = bboxes[..., 4]  ##4번째 값이 detection score이다.

        # select the face with the hightest detection score
        best_index = np.argmax(det_score)

        kps = None
        if kpss is not None:
            kps = kpss[best_index]
        M, _ = face_align.estimate_norm(kps, crop_size, mode ='None')   ##M = translation matrix
        align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)  ##https://deep-learning-study.tistory.com/175 M은 transformation을 줄 affine matrix이다.
        
        return [align_img], [M]


