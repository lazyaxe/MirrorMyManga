import time
import shutil
import os 
import numpy as np
from zipfile import ZipFile
import cv2
from pathlib import Path
from .transform_panel import transform_panel
from mirrormymanga.utils import save_imgs_as_pdf, save_imgs_as_cbz

def transform_cbz(ocr, input_path: str, output_path: str, transformCBZSettings):
    """
        Returns a PDF/CBZ of images transformed by transform_panel method. Use it ONLY for CBZ files
        * Note: 
        If you already have the transformed images, just use the `save_imgs_as_pdf` for PDF and `save_imgs_as_cbz` for CBZ format.
    """
    show_logs = transformCBZSettings.show_logs
    verbose = transformCBZSettings.verbose

    if os.path.exists(input_path):
        #unzip the .cbz file
        with ZipFile(input_path, 'r') as zfile:
            #extract the zip file in the parent directory of result directory, not inside the result
            zfile.extractall(path=input_path.removesuffix('.cbz'), pwd=None)
            if show_logs:
                print(f"LOG: Extracted {input_path} file at {input_path.removesuffix('.cbz')}")
                print("LOG: namelist = ", zfile.namelist()[:10])
        _, file_name = os.path.split(input_path.removesuffix('.cbz'))
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
                panel = transform_panel(ocr=ocr, panel=panel, show_logs=show_logs, verbose=verbose, )

                cv2.imwrite(f"{result_path}/{i}.png", img=panel)
                if show_logs:
                    print(f"LOG: page {i} saved in {time.perf_counter() - start}s")
            else:
                if show_logs:
                    print(f"LOG skipped {fname}")
        shutil.rmtree(input_path.removesuffix('.cbz'))

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
