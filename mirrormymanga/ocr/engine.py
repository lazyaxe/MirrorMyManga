import numpy as np
from paddleocr import PaddleOCR
from ..utils import is_cuda_available

def create_ocr(ocrSettings):
    lang = ocrSettings.lang
    device = ocrSettings.device
    use_doc_orientation_classify = ocrSettings.use_doc_orientation_classify
    use_doc_unwarping = ocrSettings.use_doc_unwarping
    use_textline_orientation = ocrSettings.use_textline_orientation
    enable_mkldnn = ocrSettings.enable_mkldnn
    if device=="auto":
        device = is_cuda_available()
    return PaddleOCR(lang = lang, device = device, use_doc_orientation_classify = use_doc_orientation_classify, use_doc_unwarping = use_doc_unwarping, use_textline_orientation = use_textline_orientation, enable_mkldnn = enable_mkldnn)