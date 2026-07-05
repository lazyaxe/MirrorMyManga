from zipfile import ZipFile
import time
import os
import shutil
import numpy as np
import cv2
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import fitz
import img2pdf

class MirrorMyManga:
    def __init__(self, lang="en", device="gpu", enable_mkldnn=False, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False) -> None:
        self.lang = lang
        self.device = device
        self.ROI = []
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.enable_mkldnn = enable_mkldnn
        self.ocr = PaddleOCR(lang='en', enable_mkldnn=self.enable_mkldnn, use_doc_orientation_classify=self.use_doc_orientation_classify, use_doc_unwarping=self.use_doc_unwarping, use_textline_orientation=self.use_textline_orientation, device=self.device)

    def apply_ocr(self, panel):
        """
            Returns a tuple of detected text and boundary boxes in order: x_min, y_min, x_max, y_max
        """
        panel = np.asarray(panel)
        result = self.ocr.predict(panel, use_doc_orientation_classify=self.use_doc_orientation_classify, use_doc_unwarping=self.use_doc_unwarping, use_textline_orientation=self.use_textline_orientation)
        bboxes = dict(result[0])['rec_boxes']
        text = dict(result[0])['rec_texts']
        return text, bboxes

    def extract_ROI(self, panel, draw_bounding_boxes=False, show_logs=False, return_bbox=False, color=(0, 0, 255)):
        """
            Returns the detected text as ROI(in form of NumPy array)
        """
        _, bboxes = self.apply_ocr(panel)
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
                #self.show(panel_copy)
        if return_bbox:
            return bboxes, ROI 
        else:
            return ROI

    def transform(self, panel, show_logs=False):
        """
            Applies the flipped ROI to the to the detect text ROIs, flips the whole panel then returns the panel
        """
        panel = np.asarray(panel)
        height, width = np.asarray(panel).shape[:2]
        new_height, new_width = height, width
        aspect_ratio = 1
        start = time.perf_counter()

        #if the width is too large than simply get a scaled down copy of the 
        #OG panel and perform operations on it
        if width > 1500:
            new_width = 1500
            aspect_ratio = new_width / width
            new_height = int(height * aspect_ratio)

        if show_logs:
            print("old: ", height, width, aspect_ratio)
            print("new: ", new_height, new_width, aspect_ratio)

        panel_small = cv2.resize(panel, dsize=(new_width, new_height))

        #detected text and 
        _, bboxes = self.apply_ocr(panel_small)
        

        #scaling back the bboxes to the OG resolution format
        bboxes = bboxes / aspect_ratio
        bboxes.astype(np.int32)

        #extract the ROIs with help of scaled bboxes
        ROI = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = map(int, bbox)
            if show_logs:
                    print("bbox = ", bbox)
                    print(f"x = ({x_min}, {x_max})\ny = ({y_min}, {y_max})")
            ROI.append(panel[y_min: y_max, x_min: x_max])

        #flip the detected ROIs
        ROI = list(map(np.fliplr, ROI))

        #then, assign the values of flipped ROIs to panel
        for roi, bbox in zip(ROI, bboxes):
            x_min, y_min, x_max, y_max = map(int, bbox)
            if show_logs:
                    print("bbox = ", bbox)
                    print(f"x = ({x_min}, {x_max})")
                    print(f"y = ({y_min}, {y_max})")
            panel[y_min: y_max, x_min: x_max] = roi
        
        #now flip the whole page, so the flipped text in unflipped panel becomes unflipped text in flipped panel
        panel = np.fliplr(panel)

        if show_logs:
            print("processing time: ", time.perf_counter() - start)

        return panel

    def save_imgs_as_pdf(self, input_path, output_path):
        # convert all files ending in .jpg inside a directory
        imgs_name_list = []
        for fname in os.listdir(input_path):
            path = os.path.join(input_path, fname)
            imgs_name_list.append(path)

        with open(output_path, "wb") as file:
            file.write(img2pdf.convert(imgs_name_list)) # type: ignore

    def save_imgs_as_cbz(self, input_path, output_path):
        with ZipFile(output_path, "w") as zfile:
            for fname in os.listdir(input_path):
                path = os.path.join(input_path, fname)
                zfile.write(path, arcname=fname)

    def transform_pdf(self, input_path: str, output_path: str, dpi=200, show_logs=False, output_as="pdf"):
        """
            Returns a PDF/CBZ of images transformed by transform method.
            Use it ONLY for PDF
            * Note: 
            1. Transformations in PDF are slower than transformations in CBZ,
            due to the extra work of extracting the images from the PDF
            2. If you already have the transformed images, just use the `save_imgs_as_pdf` for PDF and `save_imgs_as_cbz` for CBZ format.
        """
        file_dir, file_name = os.path.split(input_path.removesuffix('.pdf'))
        result_path = f"/home/vhvhs/MirrorMyManga/result_{file_name}"

        if not os.path.exists(result_path):
            if show_logs:
                print("DEBUG: result dir doesn't exist creating new one")
            os.mkdir(result_path)

        with fitz.open(input_path) as doc:
            #First, extract the images from the PDF as a NumPy arrays and then store them as a list of OpenCV objects.
            i = 0
            for page in doc:
                start = time.perf_counter()
                pix = page.get_pixmap(dpi=dpi, alpha=False) 
                panel = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                panel = self.transform(panel, show_logs)
                cv2.imwrite(filename=f"{result_path}/{i}.png", img=panel)
                if show_logs:
                    print(f"DEBUG: page{i} done in {time.perf_counter() - start}s")
                i += 1
            if output_as == "pdf":
                print("DEBUG: Starting PDF conversion")
                self.save_imgs_as_pdf(input_path=result_path, output_path=output_path)
            elif output_as == "cbz":
                if show_logs:
                    print("DEBUG: Starting CBZ conversion")
                self.save_imgs_as_cbz(input_path=result_path, output_path=output_path)
            else:
                raise Exception("Incorrect format of output_path")

    def transform_cbz(self, input_path: str, output_path: str, show_logs=False, output_as="cbz"):
        """
            Returns a PDF/CBZ of images transformed by transform method. Use it ONLY for CBZ files
            * Note: 
            If you already have the transformed images, just use the `save_imgs_as_pdf` for PDF and `save_imgs_as_cbz` for CBZ format.
        """
        if os.path.exists(input_path):
            #unzip the .cbz file
            with ZipFile(input_path, 'r') as zfile:
                #extract the zip file in the parent directory of result directory, not inside the result
                zfile.extractall(path=input_path.removesuffix('.cbz'), pwd=None)
                if show_logs:
                    print(f"DEBUG: Extracted {input_path} file at {input_path.removesuffix('.cbz')}")
                    print("DEBUG: namelist = ", zfile.namelist()[:10])

            file_dir, file_name = os.path.split(input_path.removesuffix('.cbz'))
            result_path = f"/home/vhvhs/MirrorMyManga/result_{file_name}"
            if not os.path.exists(result_path):
                if show_logs:
                    print("DEBUG: result directory not found, creating result dir")
                os.mkdir(result_path)

            if show_logs:
                print("DEBUG: List Dir", os.listdir(input_path.removesuffix('.cbz')))
            #Extracting, transforming, saving images from the unzipped directory to result directory
            for i, fname in enumerate(os.listdir(input_path.removesuffix('.cbz'))):
                if fname.endswith(".png") or fname.endswith(".jpg") or fname.endswith(".jpeg"):
                    panel = cv2.imread(f"/{input_path.removesuffix('.cbz')}/{fname}")
                    start = time.perf_counter()
                    panel = self.transform(panel, show_logs)
                    cv2.imwrite(f"{result_path}/{i}.png", img=panel)
                    if show_logs:
                        print(f"DEBUG: page {i} done in {time.perf_counter() - start}s")
                else:
                    if show_logs:
                        print(f"DEBUG skipped {fname}")

            shutil.rmtree(input_path.removesuffix('.cbz'))

            if output_as == "pdf":
                print("DEBUG: Starting PDF conversion")
                self.save_imgs_as_pdf(input_path=result_path, output_path=output_path)
            elif output_as == "cbz":
                if show_logs:
                    print("DEBUG: Starting CBZ conversion")
                self.save_imgs_as_cbz(input_path=result_path, output_path=output_path)
            else:
                raise Exception("Incorrect format of output_path")
        else:
            raise FileNotFoundError(f"{input_path} File not found")

if __name__ == "__main__":
    mmm = MirrorMyManga(lang="en", device="gpu")

    panel = cv2.imread('/home/vhvhs/MirrorMyManga/testFolder4/testImage4.png')

    start = time.perf_counter()
    panel = mmm.transform(panel, show_logs=True)
    #mmm.transform_pdf(input_path="/home/vhvhs/MirrorMyManga/testPDF.pdf", output_path="/home/vhvhs/MirrorMyManga/testPDF_result.pdf", output_as="pdf", dpi=200, show_logs=True)
    #mmm.transform_cbz(input_path="/home/vhvhs/MirrorMyManga/testPDF4.cbz", output_path="/home/vhvhs/MirrorMyManga/testPDF4_result.cbz", output_as="cbz", show_logs=True)

    end = time.perf_counter() - start
    print(f"Time = {end}s")

