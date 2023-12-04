import os.path as osp
import os
import glob
from itertools import combinations
import cv2
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from gluestick.models.two_view_pipeline import TwoViewPipeline

from save_descriptors_batch import get_pipeline
from util.viz_utils import draw_match_image_colors

pix_thresh = 3
dataset_root = "/media/colin/box_data/ir_data/timelapse_datasets/"
clahe_type = "matlab_clahe2"

def get_timelapse_images(path, skip = 1):
    image_paths = sorted(glob.glob(osp.join(path,clahe_type,"*.png")))[::skip]
    # image_combinations = combinations(image_paths, 2)
    # return list(image_combinations)
    # #strip pairs that are too close 6 images per hour x n hour gap
    # image_combinations = [pair for pair in image_combinations if 
    #                         6*6 < np.abs(int(osp.basename(pair[0])[:-4]) - int(osp.basename(pair[1])[:-4]))]
    # image_combinations = [pair for i, pair in enumerate(image_combinations) if i % 2 == 0] #get rid of half of pairs
    return image_paths

def get_timelapse_dataset(skip = 6):
    stub = "ir_timelapse_*/"
    paths = sorted(glob.glob(osp.join(dataset_root,stub)))

    image_fnames = [get_timelapse_images(path, skip = skip) for path in paths]

    images = [[cv2.imread(fname, cv2.IMREAD_GRAYSCALE) for fname in fnames] for fnames in image_fnames]
    return images

def match_experiment(images):
    pipeline = get_pipeline()

    querry_index = 10
    torch_images = [numpy_image_to_torch(im).to('cuda')[None] for im in images]
    querry_image = torch_images[querry_index]

    results = []
    for i, im in enumerate(torch_images):
        x = {'image0': querry_image, 'image1': im}
        pred = pipeline(x)
        pred = batch_to_np(pred)
        pred["image0_npy"] = images[querry_index]
        pred["image1_npy"] = images[i]
        pred["image_indices"] = (querry_index, i)
        print((querry_index, i))
        results.append(deepcopy(pred))
    return results

def plot_results(results, ax = None):
    if ax is None :
        fig, ax = plt.subplots()

    n_matches = []
    n_good_matches = []
    

    for result in results:
        n_matches.append(np.sum(result["matches0"] > 0))
        kp0, kp1 = result["keypoints0"], result["keypoints1"]
        # print(np.min(pixel_distdist))

        m0 = result["matches0"]
        valid_matches = m0 != -1
        match_indices = m0[valid_matches]
        matched_kps0 = kp0[valid_matches]
        matched_kps1 = kp1[match_indices]
        pixel_dist = np.linalg.norm(matched_kps0-matched_kps1, axis = 1)
        n_good_matches.append(np.sum(pixel_dist < pix_thresh))
        

    ax.plot(n_matches, 'r', label = 'matches')
    ax.plot(n_good_matches, 'b', label = "filtered matches")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.text(1, 50, "Minimum matches : {}".format(np.min(n_good_matches)), fontsize=12)
    return ax

def draw_match_images(results, save_path):
    
    for result in results:
        kp0, kp1 = result["keypoints0"], result["keypoints1"]
        # print(np.min(pixel_distdist))

        m0 = result["matches0"]
        valid_matches = m0 != -1
        match_indices = m0[valid_matches]
        matched_kps0 = kp0[valid_matches]
        matched_kps1 = kp1[match_indices]


        pixel_dist = np.linalg.norm(matched_kps0-matched_kps1, axis = 1)
        
        good = [0,255,0]
        bad = [0,0,255]
        colors = [good if a else bad for a in pixel_dist<pix_thresh]
        im0 = result["image0_npy"]
        im1 = result["image1_npy"]
        match_im = draw_match_image_colors(im0, im1, matched_kps0.astype(int), matched_kps1.astype(int), colors)
        idxs = result["image_indices"]
        save_name = "match_{:03}_{:03}.png".format(idxs[0], idxs[1])
        cv2.imwrite(osp.join(save_path, save_name), match_im)

if __name__ == "__main__":
    dataset = get_timelapse_dataset()
    save_dir = "/home/colin/Research/ir/GlueStick/scripts/tmp_results"
    fig, axes = plt.subplots(3,3)
    for i, ax in enumerate(axes.reshape(-1)):
        results = match_experiment(dataset[i])
        plot_results(results, ax=ax)
        save_path = osp.join(save_dir,"timelapse_{}".format(i))
        # os.makedirs(save_path , exist_ok=True)
        # draw_match_images(results, save_path)
        fig.savefig(osp.join(save_dir,"{}_pix_thresh.png".format(pix_thresh)))
    plt.show()
    
    # fig, axes = plt.subplots()
    # results = match_experiment(dataset[0])
    # plot_results(results, ax=axes)
    # plt.show()
    
    # #save_results
    # save_name = "tmp_results.p"
    # with open(save_name, 'wb') as file:
    #     print(type(file))
    #     print(type(results))
    #     pickle.dump(results, file)