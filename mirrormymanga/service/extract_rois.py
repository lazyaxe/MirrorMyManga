import numpy as np
import cv2
from ..utils import is_cuda_available
from ..ocr import _apply_ocr

def _extract_ROI(ocr, panel, draw_bounding_boxes=False, show_logs=False, return_bbox=False, color=(0, 0, 255), lang='en', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False):
    """
        Returns the detected text as ROI(in form of NumPy array)
    """
    _, bboxes = _apply_ocr(ocr=ocr, panel=panel, use_doc_orientation_classify=use_doc_orientation_classify, use_doc_unwarping=use_doc_unwarping, use_textline_orientation=use_textline_orientation)
    ROI = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = list(map(int, bbox))
        ROI.append(panel[y_min: y_max, x_min: x_max])
        if show_logs:
            print("bbox = ", bbox)
        if draw_bounding_boxes:
            panel_copy = np.asarray(panel).copy()
            panel_copy = cv2.cvtColor(panel_copy, cv2.COLOR_BGR2RGB)
            for bbox in bboxes:
                if show_logs:
                    print("bbox = ", bbox)
                    print(f"x = ({x_min}, {x_max})\ny = ({y_min}, {y_max})")
                x_min, y_min, x_max, y_max = map(int, bbox)
                cv2.rectangle(panel_copy, pt1=(x_min, y_min), pt2=(x_max, y_max), color=color)
    if return_bbox:
        return bboxes, ROI 
    else:
        return ROI