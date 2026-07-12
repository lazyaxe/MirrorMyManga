import os
from zipfile import ZipFile
import img2pdf

def save_imgs_as_pdf(input_path, output_path):
    # convert all files ending in .jpg inside a directory
    imgs_name_list = []
    for fname in os.listdir(input_path):
        path = os.path.join(input_path, fname)
        imgs_name_list.append(path)
    with open(output_path, "wb") as file:
        file.write(img2pdf.convert(imgs_name_list)) # type: ignore

def save_imgs_as_cbz(input_path, output_path):
    with ZipFile(output_path, "w") as zfile:
        for fname in os.listdir(input_path):
            path = os.path.join(input_path, fname)
            zfile.write(path, arcname=fname)