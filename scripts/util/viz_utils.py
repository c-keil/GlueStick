import cv2
import numpy as np
from copy import deepcopy
import matplotlib.cm as cm
# import matplotlib.col


def draw_match_image_confidence(im1, im2, kp1, kp2, confidence):
    '''Expects confidence values in range 0 to 1 '''
    #colors
    # colors = (255 * cm.winter(confidence)[:,:3]).astype(int).tolist() #normalizes to min max range
    colormap = cm.get_cmap("viridis")
    colors = [np.flip((255*np.array(colormap(c)[:3])).astype(int)).tolist() for c in confidence]

    return draw_match_image_colors(im1, im2, kp1, kp2, colors)

def draw_match_image_dist(im1, im2, kp1, kp2, distances):
    raise NotImplementedError

def draw_match_image_colors(im1, im2, kp1, kp2, colors):
    '''Draws keypoints, similar to cv2 method, but with confidence encoded as color,
    expects a simple numpy array of kp coordinates, not opencv kp objects'''
    if len(colors) == 3:
        colors = [colors for _ in range(len(kp1))]
    assert(len(colors) == len(kp1) == len(kp2))
        
    #convert images to rgb
    if not len(im1.shape) == 3:
        im1_rgb = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
    else:
        im1_rgb = deepcopy(im1)
    if not len(im2.shape) == 3:
        im2_rgb = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    else:
        im2_rgb = deepcopy(im2)
    
    #concatenate images
    im = np.concatenate((im1_rgb, im2_rgb), axis = 1)

    #draw kps
    kp1 = deepcopy(kp1).astype(int)
    kp2 = deepcopy(kp2).astype(int)
    kp2[:,0] += im1.shape[1]
    for k1, k2, color in zip(kp1, kp2, colors):
        # color.reverse() #convert rgb to bgr
        cv2.circle(im, tuple(k1), 3, color, 2)
        cv2.circle(im, tuple(k2), 3, color, 2)
        #draw lines
        cv2.line(im, tuple(k1), tuple(k2), color, 1)

    return im

def draw_lines(im, lines):
    '''todo colors'''
    
    #convert images to rgb
    if not len(im.shape) == 3:
        im_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    else:
        im_rgb = deepcopy(im)

    #check lines
    lines = np.array(lines, dtype = int) #makes a copy
    if len(lines.shape) == 2:
        lines = lines.reshape(-1,2,2)
    
    #draw lines
    color = (255,0,0)
    for line in lines:
        cv2.line(im_rgb, tuple(line[0]), tuple(line[1]), color, 1)
    
    return im_rgb

def draw_line_match_im(im0, im1, lines0, lines1):

    line_im0 = draw_lines(im0, lines0)
    line_im1 = draw_lines(im1, lines1)

    return np.concatenate((line_im0, line_im1), axis = 1)

if __name__ == "__main__":
    im1 = 128 * np.ones((512, 640), dtype = np.uint8)
    im2 = 180 * np.ones((512, 640), dtype = np.uint8)

    kps1 = np.array([[120,120],[300, 300],[300,500]])
    kps2 = np.array([[120,120],[300, 300],[300,400]])
    conf = [0.1, 0.5 ,0.9]
    # im = draw_match_image_confidence(im1, im2, kps1, kps2, conf)

    # cv2.imshow("test", im)
    # cv2.waitKey()

    lines1 = "/media/colin/box_data/ir_data/nuance_data/kri_night/cam_3/matlab_clahe2/gluestick_skip2/matched_lines/1689819868072999954.npy"
    lines1 = np.load(lines1)
    im1 = "/media/colin/box_data/ir_data/nuance_data/kri_night/cam_3/matlab_clahe2/1689819868072999954.png"
    im1 = cv2.imread(im1, 0)
    line_im = draw_lines(im1, lines1)
    cv2.imshow("test", line_im)
    cv2.waitKey()
