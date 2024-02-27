#!/bin/bash

paths=("/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_day1/cam_3/matlab_clahe2/1689283361099999905.png"
"/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_day2/cam_3/matlab_clahe2"
"/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_night/cam_3/matlab_clahe2"
"/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_night/cam_3/clahe"
"/media/colin/box_data/ir_data/nuance_data/cater_isec_day_night/carter_isec_alley_day2/cam_3/clahe_arvind")

# paths=("/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul14_7pm/cam_3/matlab_clahe2" \
# "/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul14_10pm/cam_3/matlab_clahe2" \
# "/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul17_5pm_1/cam_3/matlab_clahe2" \
# "/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul17_5pm_2/cam_3/matlab_clahe2" \
# "/media/colin/box_data/ir_data/nuance_data/columbus_garrage_day_night/garrage_roof_nuance_jul17_5pm_2/cam_3/matlab_clahe2_rect")

# paths=("/media/colin/box_data/ir_data/nuance_data/kri_day_2/cam_3/matlab_clahe2"
# "/media/colin/box_data/ir_data/nuance_data/kri_night/cam_3/matlab_clahe2")

python=/home/colin/Research/ir/GlueStick/venv/bin/python
script=/home/colin/Research/ir/GlueStick/scripts/save_descriptors_batch_v2.py
freq=2

for path in ${paths[@]}
do
    # parallel
    # $python $script $path $freq -dp &
    # sequential
    $python $script $path $freq
    
done
wait
echo "DONE"
# echo $paths