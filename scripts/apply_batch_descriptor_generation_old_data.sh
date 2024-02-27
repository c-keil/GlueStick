#!/bin/bash

paths=(/media/colin/DATA/data/snel_steps/images/clahe
/media/colin/DATA/data/bricks/images/clahe
/media/colin/DATA/data/box_gravel/images/clahe
/media/colin/DATA/data/memorial_loop/images/clahe
/media/colin/DATA/data/blob_sculpture1/images/clahe
/media/colin/DATA/data/alleyway_planter/images/clahe
/media/colin/DATA/data/courtyard_wall/images/clahe
/media/colin/DATA/data/blob_sculpture2/images/clahe
/media/colin/DATA/data/power_plant_close/images/clahe
/media/colin/DATA/data/power_plant_alley/images/clahe
/media/colin/DATA/data/power_plant_rotate/images/clahe
/media/colin/DATA/data/blob_sculpture_glare/images/clahe
/media/colin/DATA/data/power_plant_far/images/clahe
/media/colin/DATA/data/power_plant_wall/images/clahe
/media/colin/DATA/data/New_Data/coventry_street/images/clahe
/media/colin/DATA/data/New_Data/playground1/images/clahe
/media/colin/DATA/data/New_Data/playground2/images/clahe
/media/colin/DATA/data/west_villiage_wall/images/clahe
/media/colin/DATA/data/power_plant_close2/images/clahe
/media/colin/DATA/data/adam_sculpture/images/clahe)

python=/home/colin/Research/ir/GlueStick/venv/bin/python
script=/home/colin/Research/ir/GlueStick/scripts/save_descriptors_batch_v2.py
freq=5

for path in ${paths[@]}
do
    # parallel
    # $python $script $path $freq -dp &
    # sequential
    $python $script $path $freq --flatten-lines
    
done
wait
echo "DONE"
# echo $paths