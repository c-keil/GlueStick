import os.path
import cv2
import numpy as np
import glob
import torch
import argparse
from matplotlib import pyplot as plt
import pickle
import warnings

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from gluestick.models.two_view_pipeline import TwoViewPipeline

from tqdm import tqdm

def get_pipeline():
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipeline_model = TwoViewPipeline(conf).to(device).eval()
    return pipeline_model

def get_line_descriptors(descriptors, line_junc_idx):
    '''return array of line descriptors'''
    line_descs = np.empty((line_junc_idx.shape[0], line_junc_idx.shape[1], 256), dtype = np.float32)
    for i, line in enumerate(line_junc_idx):
        line_descs[i,0,:] = descriptors[line[0]]
        line_descs[i,1,:] = descriptors[line[1]]
    return line_descs

def rank_descriptors(descriptors, rank):
    pass

if __name__ == "__main__":
    # print("Skipping every other image")
    freq = 1
    error_dir = "/home/colin/Desktop"
    # np.seterr(all='warn')
    # parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description='save descriptors for bow training/evaluation')
    # parser.add_argument('image_dir', type = str)
    # parser.add_argument('frequency',
    #                      nargs = '?',
    #                      type=int,
    #                      default=freq,
    #                      help = "frequency of images to match over implemented as images[::frequency], default={}".format(freq))
    # parser.add_argument('-dp','--disable-progress',
    #                     action='store_true',
    #                     default=False,
    #                     help='disable progress bar')
    # parser.add_argument('-sb','--save-batch',
    #                      action='store_true',
    #                      default = False,
    #                      help = "Save the full prediction dict")
    # parser.add_argument('--flatten-lines',
    #                     default = False,
    #                     help = "flattens line structire from [n,2,256] to [2*n, 256]",
    #                     action = "store_true")
    # parser.add_argument('save-dir', type=str)
    # args = parser.parse_args()
    # freq = args.frequency
    disable_pbar = False
    save_batch = False
    flatten_lines = False
    print("processing every {} images".format(freq))
    image_dir1 = "/media/colin/box_data/kri_day_stereo/cam_2/"
    image_dir2 = "/media/colin/box_data/kri_day_stereo/cam_3/"
    
    start_stamp = 1689805123
    stop_stamp = 1689805137

    save_dir1 = image_dir1
    save_dir2 = image_dir2
    # img_size = (args.W, args.H)
    # save_dir = args.save_dir

    # assert(os.path.isdir(image_dir))
    # image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    image_files1 = sorted(glob.glob(os.path.join(image_dir1, "*.png")))
    image_files2 = sorted(glob.glob(os.path.join(image_dir2, "*.png")))
    stamps1 = [int(os.path.basename(f)[:10]) for f in image_files1]
    stamps2 = [int(os.path.basename(f)[:10]) for f in image_files2]
    image_files1 = [f for s,f in zip(stamps1, image_files1) if start_stamp < s < stop_stamp]
    image_files2 = [f for s,f in zip(stamps2, image_files2) if start_stamp < s < stop_stamp]

    # image_files1 = image_files1[::freq] #skip every other

    print("Running on {} images".format(len(image_files1)))
    assert(len(image_files2)>0)
    assert(len(image_files1) == len(image_files2))

    features_name = "gluestick_freq{}".format(freq)
    # kp_dir = os.path.join(save_dir1,features_name,'keypoints')
    # desc_dir = os.path.join(save_dir1,features_name,'descriptors')

    match_dir1 = os.path.join(save_dir1,features_name,'matches')
    matched_desc_dir1 = os.path.join(save_dir1,features_name,'matched_descriptors')
    matched_kp_dir1 = os.path.join(save_dir1,features_name,'matched_keypoints')
    match_dir2 = os.path.join(save_dir2,features_name,'matches')
    matched_desc_dir2 = os.path.join(save_dir2,features_name,'matched_descriptors')
    matched_kp_dir2 = os.path.join(save_dir2,features_name,'matched_keypoints')

    # # matched_line_scores_dir = 
    # line_dir = os.path.join(save_dir1,features_name,'lines')
    # matched_lines_dir = os.path.join(save_dir1,features_name,'matched_lines')
    # matched_line_indices_dir = os.path.join(save_dir1,features_name,'matched_line_indices')
    # matched_line_descs_dir = os.path.join(save_dir1,features_name,'matched_line_descs')

    # pred_dir = os.path.join(save_dir1,features_name,'batch')
    # mean_descs_dir = os.path.join(save_dir1,features_name,'mean_descriptors')
    # mean_line_descs_dir = os.path.join(save_dir1,features_name,'mean_line_descriptors')
    # match_scores_dir = os.path.join(save_dir1,features_name,'match_scores')
    # match_line_scores_dir = os.path.join(save_dir1,features_name,'match_line_scores')
    # keys = ['keypoints0', 'keypoint_scores0', 'descriptors0', 'pl_associativity0', 'num_junctions0', 'lines0', 'orig_lines0', 'lines_junc_idx0', 'line_scores0', 'valid_lines0', 'keypoints1', 'keypoint_scores1', 'descriptors1', 'pl_associativity1', 'num_junctions1', 'lines1', 'orig_lines1', 'lines_junc_idx1', 'line_scores1', 'valid_lines1', 'log_assignment', 'matches0', 'matches1', 'match_scores0', 'match_scores1', 'line_log_assignment', 'line_matches0', 'line_matches1', 'line_match_scores0', 'line_match_scores1', 'raw_line_scores']


    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # if not os.path.isdir(kp_dir):
    #     os.makedirs(kp_dir)
    # if not os.path.isdir(desc_dir):
    #     os.makedirs(desc_dir)
    # if not os.path.isdir(match_dir):
    #     os.makedirs(match_dir)
    # os.makedirs(line_dir, exist_ok=True)
    # os.makedirs(matched_line_indices_dir, exist_ok=True)
    os.makedirs(matched_desc_dir1, exist_ok=True)
    os.makedirs(matched_kp_dir1, exist_ok=True)
    os.makedirs(matched_desc_dir2, exist_ok=True)
    os.makedirs(matched_kp_dir2, exist_ok=True)
    # os.makedirs(matched_lines_dir, exist_ok=True)
    # os.makedirs(matched_line_descs_dir, exist_ok=True)
    # os.makedirs(pred_dir, exist_ok=True)
    # os.makedirs(mean_descs_dir, exist_ok=True)
    # os.makedirs(mean_line_descs_dir, exist_ok=True)
    # os.makedirs(match_scores_dir, exist_ok=True)
    # os.makedirs(match_line_scores_dir, exist_ok=True)

    pipeline = get_pipeline()

    # img0_fname1 = image_files1[0]
    # img0_name1 = os.path.basename(img0_fname1)[:-4]
    # img01 = cv2.imread(img0_fname1, 0)
    # torch_gray01 = numpy_image_to_torch(img01).to('cuda')[None]

    # img0_fname2 = image_files2[0]
    # img0_name2 = os.path.basename(img0_fname2)[:-4]
    # img02 = cv2.imread(img0_fname2, 0)
    # torch_gray02 = numpy_image_to_torch(img02).to('cuda')[None]

    for img1_fname, img2_fname in tqdm(zip(image_files1, image_files2), disable = disable_pbar):

        img1_name = os.path.basename(img1_fname)[:-4]
        img1 = cv2.imread(img1_fname, 0)
        img2_name = os.path.basename(img2_fname)[:-4]
        img2 = cv2.imread(img2_fname, 0)

        torch_gray1 = numpy_image_to_torch(img1).to('cuda')[None]
        torch_gray2 = numpy_image_to_torch(img2).to('cuda')[None]
        #inference
        x = {'image0': torch_gray1, 'image1': torch_gray2}
        pred = pipeline(x)
        #convert to numpy
        pred = batch_to_np(pred)
        #extract keypoints
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        desc0, desc1 = pred["descriptors0"].T, pred["descriptors1"].T
        line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
        lines_junc_idx0, lines_junc_idx1 = pred['lines_junc_idx0'], pred['lines_junc_idx1']
        
        #get matches
        m0 = pred["matches0"]
        m0_scores = pred['match_scores0']
        line_matches = pred["line_matches0"]
        line_scores = pred['line_scores0']

        #get matched keypoints and descriptors
        valid_matches = m0 != -1
        match_scores = m0_scores[valid_matches]
        match_indices = m0[valid_matches]
        matched_kps0 = kp0[valid_matches]
        matched_kps1 = kp1[match_indices]
        matched_desc0 = desc0[valid_matches]
        matched_desc1 = desc1[match_indices]

        #sort kps descs according to match confidence
        sort_idxs = np.argsort(-match_scores) #negative for high to low sort
        matched_kps0 = matched_kps0[sort_idxs]
        matched_kps1 = matched_kps1[sort_idxs]
        matched_desc0 = matched_desc0[sort_idxs]
        matched_desc1 = matched_desc1[sort_idxs]
        matched_scores_sorted = match_scores[sort_idxs]
        #remove edge case where descriptors are off the edge of the image (may only affect lines)
        good_descriptors = np.logical_or(np.all(matched_desc0 != 0, axis = 1),
                          np.all(matched_desc1 != 0, axis = 1))
        if not np.all(good_descriptors):
            print("fround overflow keypoint in images \n{}\n{}".format(img0_fname1,img1_fname))
            matched_kps0 = matched_kps0[good_descriptors]
            matched_kps1 = matched_kps1[good_descriptors]
            matched_desc0 = matched_desc0[good_descriptors]
            matched_desc1 = matched_desc1[good_descriptors]
            matched_scores_sorted = matched_scores_sorted[good_descriptors]

        # #compute mean descriptor across match
        # mean_descs = (matched_desc0+matched_desc1)/2
        # #normalize average line descriptors
        # mean_descs /= np.linalg.norm(mean_descs, axis = -1).reshape(-1,1)


        # #get matched lines
        # valid_line_matches = line_matches != -1
        # match_line_indices = line_matches[valid_line_matches]
        # matched_lines0 = line_seg0[valid_line_matches]
        # matched_lines1 = line_seg1[match_line_indices]
        # matched_lines_junc_idx0 = lines_junc_idx0[valid_line_matches]
        # matched_lines_junc_idx1 = lines_junc_idx1[match_line_indices]
        # matched_line_descs0 = get_line_descriptors(desc0, matched_lines_junc_idx0)
        # matched_line_descs1 = get_line_descriptors(desc0, matched_lines_junc_idx1)
        # matched_line_scores = line_scores[valid_line_matches]

        # #sort matched lines
        # line_sort_idxs = np.argsort(-matched_line_scores)
        # matched_lines0 = matched_lines0[line_sort_idxs]
        # matched_lines1 = matched_lines1[line_sort_idxs]
        # matched_line_descs0 = matched_line_descs0[line_sort_idxs]
        # matched_line_descs1 = matched_line_descs1[line_sort_idxs]
        # matched_line_scores_sorted = matched_line_scores[line_sort_idxs]
        
        # #remove lines that overflow image_border (very rare?, results in desc = 0)
        # good_line_descriptors = np.all(np.logical_or( np.all(matched_line_descs0 != 0, axis = -1),
        #                         np.all(matched_line_descs0 != 0, axis = -1)), axis = -1)
        # if not np.all(good_line_descriptors):
        #     print("fround overflow lines in images \n{}\n{}".format(img0_fname1,img1_fname))
        #     matched_lines0 = matched_lines0[good_line_descriptors]
        #     matched_lines1 = matched_lines1[good_line_descriptors]
        #     matched_line_descs0 = matched_line_descs0[good_line_descriptors]
        #     matched_line_descs1 = matched_line_descs1[good_line_descriptors]
        #     matched_line_scores_sorted = matched_line_scores_sorted[good_line_descriptors]

        # # compute mean line desc
        # mean_line_descs = (matched_line_descs0+matched_line_descs1)/2
        # # #normalize average line descriptors
        # mean_line_descs /= np.linalg.norm(mean_line_descs, axis = -1).reshape(-1,2,1)
        
        # #save batch
        # if save_batch:
        #     with open(os.path.join(pred_dir, img0_name1 + "_" + img1_name), 'wb') as p:
        #         pickle.dump(pred, p)
        
        #Old saved things
        # np.save(os.path.join(kp_dir, img0_name), kp0)
        # np.save(os.path.join(kp_dir, img1_name + "_"), kp1)
        # np.save(os.path.join(desc_dir, img0_name), desc0)
        # np.save(os.path.join(desc_dir, img1_name + "_"), desc1)
        # np.save(os.path.join(match_dir, img0_name + "_" + img1_name), m0)
        # np.save(os.path.join(line_dir, img0_name), line_seg0)
        # np.save(os.path.join(line_dir, img1_name + "_"), line_seg1)
        # np.save(os.path.join(matched_line_indices_dir, img0_name), matched_lines_junc_idx0)
        # if flatten_lines:
        #     matched_lines0 = matched_lines0.reshape(-1, 2)
        #     matched_lines1 = matched_lines1.reshape(-1, 2)
        #     matched_line_descs0 = matched_line_descs0.reshape(-1, 256)
        #     matched_line_descs1 = matched_line_descs1.reshape(-1, 256)
        #     matched_line_scores_sorted = matched_line_scores_sorted.reshape(-1)
        #     mean_line_descs = mean_line_descs.reshape(-1, 256)

        #save matched lines and kps
        np.save(os.path.join(matched_kp_dir1, img1_name), matched_kps0)
        np.save(os.path.join(matched_desc_dir1, img1_name), matched_desc0)
        # np.save(os.path.join(matched_lines_dir, img0_name1), matched_lines0)
        # np.save(os.path.join(matched_line_descs_dir, img0_name1), matched_line_descs0)
        np.save(os.path.join(matched_kp_dir2, img2_name), matched_kps1)
        np.save(os.path.join(matched_desc_dir2, img2_name), matched_desc1)
        # np.save(os.path.join(matched_lines_dir, img1_name+"_"), matched_lines1)
        # np.save(os.path.join(matched_line_descs_dir, img1_name+"_"), matched_line_descs1)
        #save mean descs
        # np.save(os.path.join(mean_descs_dir, img0_name1 + "_" + img1_name), mean_descs)
        # np.save(os.path.join(mean_line_descs_dir, img0_name1 + "_" + img1_name), mean_line_descs)
        #save scores
        # np.save(os.path.join(match_scores_dir, img0_name1), matched_scores_sorted)
        # np.save(os.path.join(match_line_scores_dir, img0_name1), matched_line_scores_sorted)
        

        # #book keeping
        # img0_fname1 = img1_fname
        # img0_name1 = img1_name
        # img01 = img1
        # torch_gray01 = torch_gray1
        # quit()
    print("completed image set {} with freq {}".format(image_dir1,freq))
