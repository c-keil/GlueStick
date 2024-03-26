import numpy as np
import torch
import cv2
import pickle

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
        # print("Python Side -- __init__ SLAM_Pipeline()")
        MAX_N_POINTS, MAX_N_LINES = 1000, 300
        self.image_counter = 0 #debug
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

        self.frames = {}
        self.reloaclization_frames = {}
        self.next_frame_id = 0
        self.debug_counter = 0
    
    def detect_extract(self, image, transpose = True):
        '''expects a numpy array image'''
        print("Python side -- detect_extract()")
        #convert to torch 
        torch_image = numpy_image_to_torch(image)
        torch_image = torch_image.to(self.device)[None]
        #run get kps, lines, descs
        pred = self.pipeline._extract_single_forward(torch_image)
        #get numpy versions of everything
        pred = batch_to_np(pred)
        if transpose:
            pred["descriptors"] = pred["descriptors"].T.copy(order = 'C')
        return pred
    
    def copy_numpy(tensor):
        return tensor.clone().detach().numpy()[0]
    
    def detect_extract2(self, image):
        '''expects a numpy array image'''
        #convert to torch 
        torch_image = numpy_image_to_torch(image)
        torch_image = torch_image.to(self.device)[None]
        #run get kps, lines, descs
        pred = self.pipeline._extract_single_forward(torch_image)

        #get numpy versions of everything
        descs = np.ascontiguousarray(pred["descriptors"].clone().detach().cpu().numpy()[0].T)
        kps = np.ascontiguousarray(pred["keypoints"].clone().detach().cpu().numpy()[0])
        prediction = {"gs_frame_id":self.next_frame_id, "descriptors":descs, "keypoints":kps}

        #store frame
        self.frames[self.next_frame_id] = pred
        self.next_frame_id += 1

        return prediction
    
    def match2(self, frame_id1, frame_id2, reloc1 = False, reloc2 = False, return_full_batch = False):
        '''runs matching against frames stored in python dict. reloc# attempts to use frames stored in the relocalization frames'''
        if not reloc1:
            pred1 = self.frames[frame_id1]
        else:
            pred1 = self.reloaclization_frames[frame_id1]
        if not reloc2:
            pred2 = self.frames[frame_id2]
        else:   
            pred2 = self.reloaclization_frames[frame_id2]
        pred = self.pipeline._match_forward(self.pipeline._merge_kp_pred(pred1, pred2))
        matches = pred["matches0"].clone().detach()[0].tolist()
        match_scores = pred["match_scores0"].clone().detach()[0].cpu().numpy()

        match_distances = 1 - match_scores

        if return_full_batch:
            return batch_to_np(pred)
        return matches, match_distances.tolist()
    
    def draw_debug_match_image(self, pred,i,j,a,b):
        im0 = pred["image_debug0"]
        im1 = pred["image_debug1"]
        
        pred = batch_to_np(pred)

        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0 = pred["matches0"]

        valid_matches = m0 != -1
        match_indices = m0[valid_matches]
        matched_kps0 = kp0[valid_matches]
        matched_kps1 = kp1[match_indices]
        keypoints0 = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in kp0]
        keypoints1 = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in kp1]

        good = [cv2.DMatch(i,j,0.0) for i,j in enumerate(m0) if j != -1]

        match_im = cv2.drawMatches(im0, keypoints0, im1, keypoints1, good, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        name = '/home/colin/Research/ir/slam_ws/mapping_results_tmp/debug_ims/' + "{}_{}_{}_{}_{}.png".format(self.debug_counter,i,j,a,b)
        cv2.imwrite(name,match_im)
        self.debug_counter += 1


    def match(self, image_data1, image_data2):
        #put back on the gpu
        image_data1 = numpy_to_batch(image_data1, self.device)
        image_data2 = numpy_to_batch(image_data2, self.device)
        #merge image batches
        pred = self.pipeline._merge_kp_pred(image_data1, image_data2)
        #match
        pred = batch_to_np(self.pipeline._match_forward(pred))
        return pred
    
    def save_frames(self, fname="/home/colin/Research/ir/slam_ws/mapping_results_tmp/gs_database.p"):
        '''saves a pickle of the frames database'''
        with open(fname,'bw') as pf:
            pickle.dump(self.frames, pf)
    
    def load_relocalization_frames(self, fname="/home/colin/Research/ir/slam_ws/mapping_results_tmp/gs_database.p"):
        '''reads pickled db'''
        with open(fname,'br') as pf:
            self.reloaclization_frames = pickle.load(pf)