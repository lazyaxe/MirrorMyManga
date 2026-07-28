from .ocr import create_ocr
from .utils import is_cuda_available
from .service import transform_pdf, transform_cbz, cli
import time
import os, sys
from pathlib import Path
from .settings import OCRSettings, TransformPanelSettings, TransformPDFSettings, TransformCBZSettings

def main():
    start = time.perf_counter()
    cli()

    #USAGE EXAMPLE FOR A SCRIPT:
    #input_path = "/home/vhvhs/Test/testPDF4.pdf"
    #output_path = os.path.join(Path.cwd(), "testPDF4_result.cbz")
    #ocrSettings = OCRSettings(lang="")
    #ocr = create_ocr(ocrSettings=OCRSettings(lang="en"))
    #transform_pdf(ocr=ocr, input_path=input_path, output_path=output_path, transformPDFSettings=TransformPDFSettings(dpi=100))
    #print(f"Program ended in {time.perf_counter() - start}s")
if __name__ == "__main__":
    sys.exit(main())
