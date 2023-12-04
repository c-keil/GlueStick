import numpy as np
import glob
import os 
'''Filters saved descriptors to save just mached descriptors and keypoints
- 
should not be needed further'''

# image_dir = "/media/colin/box_data/ir_data/nuance_data/kri_night/cam_3/matlab_clahe2/"
# desc_dir = "/media/colin/box_data/ir_data/nuance_data/kri_night/cam_3/matlab_clahe2/gluestick/descriptors"
# kp_dir = "/media/colin/box_data/ir_data/nuance_data/kri_night/cam_3/matlab_clahe2/gluestick/keypoints"
# match_dir = "/media/colin/box_data/ir_data/nuance_data/kri_night/cam_3/matlab_clahe2/gluestick/matches"

# image_dir = "/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/"
# desc_dir = "/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/gluestick/descriptors"
# kp_dir = "/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/gluestick/keypoints"
# match_dir = "/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2/gluestick/matches"

image_dir = "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_night/cam_3/matlab_clahe2"
desc_dir = "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_night/cam_3/matlab_clahe2/gluestick_skip2/descriptors"
kp_dir = "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_night/cam_3/matlab_clahe2/gluestick_skip2/keypoints"
match_dir = "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_night/cam_3/matlab_clahe2/gluestick_skip2/matches"

image_dir = "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_day2/cam_3/matlab_clahe2"
desc_dir = "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_day2/cam_3/matlab_clahe2/gluestick_skip2/descriptors"
kp_dir = "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_day2/cam_3/matlab_clahe2/gluestick_skip2/keypoints"
match_dir = "/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_day2/cam_3/matlab_clahe2/gluestick_skip2/matches"


match_desc_dir = os.path.join(os.path.dirname(desc_dir), "matched_descriptors")
match_kp_dir = os.path.join(os.path.dirname(desc_dir), "matched_keypoints")
os.makedirs(match_desc_dir, exist_ok=True)
os.makedirs(match_kp_dir, exist_ok=True)

desc_files0 = sorted(glob.glob(os.path.join(desc_dir, "*[0-9].npy")))
desc_files1 = sorted(glob.glob(os.path.join(desc_dir, "*_.npy")))
kp_files0 = sorted(glob.glob(os.path.join(kp_dir, "*[0-9].npy")))
kp_files1 = sorted(glob.glob(os.path.join(kp_dir, "*_.npy")))
match_files = sorted(glob.glob(os.path.join(match_dir,"*.npy")))

match_sets = [np.load(m) for m in match_files]
match_pairs = [os.path.basename(a)[:-4].split("_") for a in match_files ]

desc_strs0 = [os.path.basename(a)[:-4] for a in desc_files0]
desc_strs1 = [os.path.basename(a)[:-5] for a in desc_files1]

for i in range(len(match_pairs)):
    matches = match_sets[i]
    desc0 = np.load(desc_files0[i])
    desc1 = np.load(desc_files1[i])
    kp0 = np.load(kp_files0[i])
    kp1 = np.load(kp_files1[i])

    match_str0 = match_pairs[i][0]
    match_str1 = match_pairs[i][1]
    desc_str0 = desc_strs0[i]
    desc_str1 = desc_strs1[i]


    if desc_str0 != match_str0 or desc_str1 != match_str1:
        print(match_str0 = match_pairs[i][0], match_str1 = match_pairs[i][1], desc_str0 = desc_strs0[i], desc_str1 = desc_strs1[i])
        raise ValueError

    valid_matches = matches != -1
    # print("{} valid matches".format(np.sum(valid_matches)))
    match_indices = matches[valid_matches]
    matched_desc0 = desc0[valid_matches]
    matched_desc1 = desc1[match_indices]
    matched_kp0 = kp0[valid_matches]
    matched_kp1 = kp1[match_indices]

    # np.save(os.path.join(match_desc_dir, desc_str0), matched_desc0)
    np.save(os.path.join(match_kp_dir, desc_str0), matched_kp0) 