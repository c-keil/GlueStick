import numpy as np
import torch

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
# from gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from gluestick.models.two_view_pipeline import TwoViewPipeline

def numpy_to_batch(batch, device):
    for key in batch.keys():
        d = batch[key]
        if not type(d) == list:
            try :
                d = torch.from_numpy(d).unsqueeze_(0).to(device)
            except TypeError:
                d = torch.tensor([d], dtype = torch.int64)
        else:
            d = torch.Size(d)
        batch[key] = d
    return batch

class SLAM_Pipeline():
    '''thin wrapper around gluestick to make integration to c++ more convenient'''
    def __init__(self):

        MAX_N_POINTS, MAX_N_LINES = 1000, 300

        # Evaluation config
        conf = {
            'name': 'two_view_pipeline',
            'use_lines': True,
            'extractor': {
                'name': 'wireframe',
                'sp_params': {
                    'force_num_keypoints': False,
                    'max_num_keypoints': MAX_N_POINTS,
                },
                'wireframe_params': {
                    'merge_points': True,
                    'merge_line_endpoints': True,
                },
                'max_n_lines': MAX_N_LINES,
            },
            'matcher': {
                'name': 'gluestick',
                'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
                'trainable': False,
            },
            'ground_truth': {
                'from_pose_depth': False,
            }
        }

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.pipeline = TwoViewPipeline(conf).to(self.device).eval()
    
    def detect_extract(self, image):
        '''expects a numpy array image'''
        #convert to torch 
        torch_image = numpy_image_to_torch(image)
        #run get kps, lines, descs
        pred = self.pipeline._extract_single_forward(torch_image)
        #get numpy versions of everything
        pred = batch_to_np(pred)
        return pred
    
    def match(self, image_data1, image_data2):
        #put back on the gpu
        image_data1 = numpy_to_batch(image_data1, self.device)
        image_data2 = numpy_to_batch(image_data2, self.device)
        #merge image batches
        pred = self.pipeline._merge_kp_pred(image_data1, image_data2)
        #match
        pred = self.pipeline._match_forward(pred)

        return pred 

