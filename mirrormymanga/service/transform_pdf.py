import time
import shutil
import os 
import numpy as np
import fitz
import cv2
from pathlib import Path
from .transform_panel import transform_panel
from mirrormymanga.utils import save_imgs_as_pdf, save_imgs_as_cbz

def transform_pdf(ocr, input_path: str, output_path: str, transformPDFSettings):
    """
        Returns a PDF/CBZ of images transformed by transform_panel method.
        Use it ONLY for PDF
        * Note: 
        1. Transformations in PDF are slower than transformations in CBZ,
        due to the extra work of extracting the images from the PDF
        2. If you already have the transformed images, just use the `save_imgs_as_pdf` for PDF and `save_imgs_as_cbz` for CBZ format.
    """
    verbose = transformPDFSettings.verbose
    show_logs = transformPDFSettings.show_logs
    dpi = transformPDFSettings.dpi

    _, file_name = os.path.split(input_path.removesuffix('.pdf'))
    result_path = Path.cwd() / f"result_{file_name}"
    if not os.path.exists(result_path):
        if verbose:
            print("LOG: result dir doesn't exist creating new one")
        os.mkdir(result_path)

    with fitz.open(input_path) as doc:
        #First, extract the images/pages from the PDF.
        i = 0
        start = time.perf_counter()
        for page in doc:
            start = time.perf_counter()
            #Change DPI if the resulted image will take too long to compute
            MAX_SIDE_LENGTH = 1988
            expected_width = int(page.rect.width * dpi / 72)
            if expected_width >= MAX_SIDE_LENGTH:
                if show_logs:
                    print(f"LOG: The page {i} is too big to be processed efficiently, reducding the dpi to {dpi}")
                dpi = 100
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            
            if verbose:
                print(f"VERBOSE: Page {i} height, width: {page.get_images()[0][3]}, {page.get_images()[0][2]}")
                print(f"VERBOSE: DPI of page {i} = {dpi}")
                print(f"VERBOSE: Pixmap for page {i} genrated in {time.perf_counter() - start}")

            panel = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            panel = transform_panel(ocr, panel=panel, show_logs=show_logs, verbose=verbose)
            cv2.imwrite(f"{result_path}/{i}.jpeg", panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if show_logs:
                print(f"LOG: page{i} saved in {time.perf_counter() - start}s")
            i += 1
        if output_path.endswith(".pdf"):
            if show_logs:
                print("LOG: Starting PDF conversion")
            save_imgs_as_pdf(input_path=result_path, output_path=output_path)
        elif output_path.endswith(".cbz"):
            if show_logs:
                print("LOG: Starting CBZ conversion")
            save_imgs_as_cbz(input_path=result_path, output_path=output_path)
        else:
            raise Exception("Incorrect format of output_path")
        shutil.rmtree(result_path)

        if show_logs:
            print("Transformed & Saved PDF in ", time.perf_counter() - start)