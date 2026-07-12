import numpy as np

def apply_ocr(ocr, panel, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False):
    """
        Returns a tuple of detected text and boundary boxes in order: x_min, y_min, x_max, y_max
    """
    panel = np.asarray(panel)
    result = ocr.predict(panel, use_doc_orientation_classify=use_doc_orientation_classify, use_doc_unwarping=use_doc_unwarping, use_textline_orientation=use_textline_orientation)
    bboxes = dict(result[0])['rec_boxes']
    text = dict(result[0])['rec_texts']
    return text, bboxes
