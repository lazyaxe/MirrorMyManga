from zipfile import ZipFile
import time
import os
import shutil
import numpy as np
import cv2
from paddleocr import PaddleOCR
import fitz
import img2pdf
from pathlib import Path
#if you want to see the transformed image
import matplotlib.pyplot as plt

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

    def transform(self, panel, show_logs=False, verbose=False):
        """
            Applies the flipped ROI to the to the detect text ROIs, flips the whole panel then returns the panel
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
        text, bboxes = self.apply_ocr(panel_small)
        
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
            print("LOG: image transform time = ", time.perf_counter() - start)
        return flipped_panel

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

    def transform_pdf(self, input_path: str, output_path: str, dpi=200, show_logs=False, output_as="pdf", verbose=False):
        """
            Returns a PDF/CBZ of images transformed by transform method.
            Use it ONLY for PDF
            * Note: 
            1. Transformations in PDF are slower than transformations in CBZ,
            due to the extra work of extracting the images from the PDF
            2. If you already have the transformed images, just use the `save_imgs_as_pdf` for PDF and `save_imgs_as_cbz` for CBZ format.
        """
        file_dir, file_name = os.path.split(input_path.removesuffix('.pdf'))
        result_path = Path.cwd() / f"result_{file_name}"

        if not os.path.exists(result_path):
            if verbose:
                print("LOG: result dir doesn't exist creating new one")
            os.mkdir(result_path)

        with fitz.open(input_path) as doc:
            #First, extract the images from the PDF as a NumPy arrays and then store them as a list of OpenCV objects.
            i = 0
            for page in doc:
                #print("Print get images: ", page.get_images())
                for img in page.get_images():
                    xref = img[0]
                    image = doc.extract_image(xref)
                    panel = cv2.imdecode(np.frombuffer(image["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
                    print(panel.shape)
                    cv2.imwrite(f"debug_{xref}.png", panel)

                pix = page.get_pixmap(dpi=dpi, alpha=False)

                panel = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                panel = self.transform(panel, show_logs, verbose)

                start = time.perf_counter()
                cv2.imwrite(f"{result_path}/{i}.jpeg", panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if show_logs:
                    print(f"LOG: page{i} saved in {time.perf_counter() - start}s")
                i += 1

            if output_as == "pdf":
                if show_logs:
                    print("LOG: Starting PDF conversion")
                self.save_imgs_as_pdf(input_path=result_path, output_path=output_path)
            elif output_as == "cbz":
                if show_logs:
                    print("LOG: Starting CBZ conversion")
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
                    print(f"LOG: Extracted {input_path} file at {input_path.removesuffix('.cbz')}")
                    print("LOG: namelist = ", zfile.namelist()[:10])

            file_dir, file_name = os.path.split(input_path.removesuffix('.cbz'))
            result_path = Path.cwd() / f"result_{file_name}"
            if not os.path.exists(result_path):
                if show_logs:
                    print("LOG: result directory not found, creating result dir")
                os.mkdir(result_path)

            if show_logs:
                print("LOG: List Dir", os.listdir(input_path.removesuffix('.cbz')))
            #Extracting, transforming, saving images from the unzipped directory to result directory
            for i, fname in enumerate(os.listdir(input_path.removesuffix('.cbz'))):
                if fname.endswith(".png") or fname.endswith(".jpg") or fname.endswith(".jpeg"):
                    start = time.perf_counter()
                    panel = cv2.imread(f"/{input_path.removesuffix('.cbz')}/{fname}")
                    panel = self.transform(panel, show_logs)
                    cv2.imwrite(f"{result_path}/{i}.png", img=panel)
                    if show_logs:
                        print(f"LOG: page {i} saved in {time.perf_counter() - start}s")
                else:
                    if show_logs:
                        print(f"LOG skipped {fname}")

            shutil.rmtree(input_path.removesuffix('.cbz'))

            if output_as == "pdf":
                print("LOG: Starting PDF conversion")
                self.save_imgs_as_pdf(input_path=result_path, output_path=output_path)
            elif output_as == "cbz":
                if show_logs:
                    print("LOG: Starting CBZ conversion")
                self.save_imgs_as_cbz(input_path=result_path, output_path=output_path)
            else:
                raise Exception("Incorrect format of output_path")
        else:
            raise FileNotFoundError(f"{input_path} File not found")

if __name__ == "__main__":
    start = time.perf_counter()
    mmm = MirrorMyManga(lang="en", device="gpu")

    mmm.transform_pdf(input_path="/home/vhvhs/MirrorMyManga/testPDF.pdf", output_path="/home/vhvhs/MirrorMyManga/testPDF_result.pdf", output_as="pdf", dpi=200, show_logs=True, verbose=False)

    end = time.perf_counter() - start
    print(f"Program ended in {end}s")

