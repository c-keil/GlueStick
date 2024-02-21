import numpy as np
import torch
import cv2

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
        print("Python Side -- __init__ SLAM_Pipeline()")
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
        self.next_frame_id = 0
    
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
        if np.any(pred["keypoints"] < 0) or np.any(pred["keypoints"] > np.array((639,511)).reshape(1,2)):
            print("python -- BAD KEYPOINT")
            p = "/home/colin/Research/ir/slam_ws/tmp"
            cv2.imwrite(p + "/debug_image_{}.png".format(self.image_counter), image)
            print(pred["keypoints"][pred["keypoints"] > np.array((639,511)).reshape(1,2)] )
        # print("Python: descriptors shape : {}".format(pred["descriptors"].shape))
        # print("Python: extracted {} descriptors".format(len(pred["descriptors"])))
        # print("Desc first row: {}...".format(pred["descriptors"][0,:10]))
        # print("Desc last row: {}...".format(pred["descriptors"][-1,:10]))
        # # print("Flags:")
        # # print(pred["descriptors"].flags)
        # #debug stuff
        # p = "/home/colin/Research/ir/slam_ws/tmp"
        # cv2.imwrite(p + "/debug_image_{}.png".format(self.image_counter), image)
        # self.image_counter += 1 
        return pred
    
    def copy_numpy(tensor):
        return tensor.clone().detach().numpy()[0]
    
    def detect_extract2(self, image):
        '''expects a numpy array image'''
        print("Python side -- detect_extract2()")
        #convert to torch 
        torch_image = numpy_image_to_torch(image)
        torch_image = torch_image.to(self.device)[None]
        #run get kps, lines, descs
        pred = self.pipeline._extract_single_forward(torch_image)
        #get numpy versions of everything
        # pred = batch_to_np(pred)
        descs = np.ascontiguousarray(pred["descriptors"].clone().detach().cpu().numpy()[0].T)
        print(descs.shape)
        kps = np.ascontiguousarray(pred["keypoints"].clone().detach().cpu().numpy()[0])
        prediction = {"gs_frame_id":self.next_frame_id, "descriptors":descs, "keypoints":kps}

        #store frame
        self.frames[self.next_frame_id] = pred
        self.next_frame_id += 1

        return prediction
    
    def match2(self, frame_id1, frame_id2):
        '''runs matching against frames stored in python object'''
        print("Python Side -- match2: frame {} against frame {}".format(frame_id1, frame_id2))
        pred1 = self.frames[frame_id1]
        pred2 = self.frames[frame_id2]
        pred = self.pipeline._match_forward(self.pipeline._merge_kp_pred(pred1, pred2))
        matches = pred["matches0"].clone().detach()[0].tolist()
        match_scores = pred["match_scores0"].clone().detach()[0].cpu().numpy()
        # #quick filter
        # c = 0
        # for i,s in enumerate(match_scores):
        #     if s < 0.9 and (not matches[i] == -1):
        #         matches[i] = -1
        #         c += 1
        # print("removed {} matches".format(c))
        match_distances = 1 - match_scores
        print("python side -- match2: first matches: {}".format(matches[:10]))
        return matches, match_distances.tolist()
    
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

    # pred0 = slam_pipe.detect_extract(gray0)
    # pred1 = slam_pipe.detect_extract(gray1)
    # print("kp detection keys")
    # for k, v in pred0.items():
    #     print("key : {} , type : {}, shape {}".format(k, type(v), v.shape if type(v) == np.ndarray else -1))
    # print(pred0["keypoints"][0])
    # print("descriptor row:\n{}".format(pred0["descriptors"][0,:10]))

    # pred = slam_pipe.match(pred0, pred1)
    # print("descriptor row:\n{}".format(pred0["descriptors"][0,0,:10]))
    # print("descriptor row:\n{}".format(pred["descriptors0"][0,:10]))
    # print("match result keys")
    # print(pred.keys())
    # for k, v in pred.items():
    #     print("key : {} , type : {}, shape {}".format(k, type(v), v.shape if type(v) == np.ndarray else -1))
    # print(pred["matches0"])
    
    pred0 = slam_pipe.detect_extract2(gray0)
    pred1 = slam_pipe.detect_extract2(gray1)
    matches, match_scores = slam_pipe.match2(pred0["gs_frame_id"], pred1["gs_frame_id"])
    # print(matches[:10])
    # print(match_scores[:10])
    
    # har = cv2.cornerHarris(gray0,10,5,0.001)
    # print("min :", har.min())
    # print("max :", har.max())
    # print("mean: ", har.mean())
    # har = cv2.dilate(har,None)
    # # har[har<0.01] = 0.0

    # import matplotlib.pyplot as plt
    # rbgim = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR)
    # rbgim[har > 0.01] = [0,0,255]
    # fig, ax = plt.subplots()
    # ax.imshow(rbgim)
    # plt.show()

