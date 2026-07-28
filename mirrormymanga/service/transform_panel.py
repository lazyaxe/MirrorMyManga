import cv2
import numpy as np
import time
from mirrormymanga.ocr import _apply_ocr

def transform_panel(ocr, panel, show_logs=False, verbose=False):
    """
        transform_panel calls the apply_ocr method for ROI and bbox detection, flips the page, pastes the roi in the flipped page, returns the roi
    """
    
    start = time.perf_counter()
    panel = np.asarray(panel)
    height, width = panel.shape[:2]
    new_witdh, new_width = height, width
    aspect_ratio = 1

    #Max limit for the OG image
    #This max_limit is used for the final image whreas the other one is for the transformation operations
    MAX_LIMIT = 89478485
    if height * width >= MAX_LIMIT:
        height = int(height * 0.8)
        width = int(width * 0.8)
        panel = cv2.resize(panel, dsize=(width, height))

    #if the width is too large than simply get a scaled down copy of the 
    #OG panel and perform operations on it
    MAX_LIMIT = 1500
    if width > MAX_LIMIT:
        new_width = MAX_LIMIT
        aspect_ratio = new_width / width
        new_witdh = int(height * aspect_ratio)
        if verbose:
            print("length, old_width, aspect_ratio: ", height, width, aspect_ratio)
            print("new_length, new_width, aspect_ratio : ", new_witdh, new_width, aspect_ratio)
    panel_small = cv2.resize(panel, dsize=(new_width, new_witdh))
    
    #detected text and the boundary boxes
    _, bboxes = _apply_ocr(ocr, panel=panel_small)
    
    #scaling back the bboxes to the OG resolution format
    bboxes = (bboxes / aspect_ratio).astype(np.uint16)
    flipped_panel = np.fliplr(panel)
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        mirror_x_max, mirror_x_min = (width - x_min, width - x_max)
        if verbose:
                print("bbox = ", bbox)
                print(f"x_min, x_max = {x_min}, {x_max}")
                print(f"y_min, y_max = {y_min}, {y_max}")
                print(f"mirror_x_min, mirror_x_max = {mirror_x_min}, {mirror_x_max}")
                print("roi", panel[y_min: y_max, x_min: x_max])
        flipped_panel[y_min: y_max, mirror_x_min: mirror_x_max] = panel[y_min: y_max, x_min: x_max]
    if show_logs:
        print(f"LOG: image transformed in {time.perf_counter() - start}s")
    return flipped_panel