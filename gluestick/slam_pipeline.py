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
        torch_image = torch_image.to(self.device)[None]
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
        pred = batch_to_np(self.pipeline._match_forward(pred))
        
        return pred 

if __name__ == "__main__":
    
    import cv2

    slam_pipe = SLAM_Pipeline()

    
    img_path0 = "/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/1689805041743999958.png"
    img_path1 = "/media/colin/box_data/ir_data/nuance_data/kri_night/cam_3/matlab_clahe2/1689819945923000097.png"
    gray0 = cv2.imread(img_path0, cv2.IMREAD_GRAYSCALE)
    gray1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)

    # import matplotlib.pyplot as plt
    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(gray0, cmap='gray')
    # axarr[1].imshow(gray1, cmap='gray')
    # plt.show()

    # torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1)
    # torch_gray0, torch_gray1 = torch_gray0.to(slam_pipe.device)[None], torch_gray1.to(slam_pipe.device)[None]

    pred0 = slam_pipe.detect_extract(gray0)
    pred1 = slam_pipe.detect_extract(gray1)
    print("kp detection keys")
    for k, v in pred0.items():
        print("key : {} , type : {}, shape {}".format(k, type(v), v.shape if type(v) == np.ndarray else -1))
    print(pred0["keypoints"][0])
    print("descriptor row:\n{}".format(pred0["descriptors"][0,:10]))

    pred = slam_pipe.match(pred0, pred1)
    print("descriptor row:\n{}".format(pred0["descriptors"][0,0,:10]))
    print("descriptor row:\n{}".format(pred["descriptors0"][0,:10]))
    print("match result keys")
    print(pred.keys())
    for k, v in pred.items():
        print("key : {} , type : {}, shape {}".format(k, type(v), v.shape if type(v) == np.ndarray else -1))
    print(pred["matches0"])