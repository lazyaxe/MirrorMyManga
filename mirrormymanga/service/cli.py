import argparse
from . import transform_pdf, transform_cbz
from mirrormymanga.settings import OCRSettings, TransformPDFSettings, TransformCBZSettings
from mirrormymanga.ocr.engine import create_ocr

def cli():
    """
        A simple Command Line Interface(CLI) of MirrorMyManga project.\n usage: $mirrormymanga input_path output_path --lang [-h | --help] | [-S | --showlogs] | [-V | --verbose]\n options:\n-h, --help     show this help message and exit\n-V, --verbose    show low level details like image size, width lenght, boundarries etc...\n-S, --showlogs            show logs of occuring processes\n\npositional arguments:\n  input_path               source file(PDF or CBZ) of the transformed manga/comic\n  output_path               destination file(PDF or CBZ) of the transformed manga/comic
    """
    parser = argparse.ArgumentParser(description="\nFlip the reading orientation of Manga/Comics\n", usage="$mirrormymanga input_path output_path [--lang] [--dpi] [-h] [-S] [-V]", add_help=True)

    parser.add_argument("input_path", type=str, help="source file(PDF or CBZ) of the manga/comic")
    parser.add_argument("output_path", type=str, help="destination file(PDF or CBZ) of the transformed manga/comic")
    parser.add_argument("--lang", type=str, help='the language that model should detect accurately, example english: "en" other supported languages, "ch", "chinese_cht", "en", "japan"', default="en")
    parser.add_argument("--dpi", type=int, help="the DPI to use for scanning PDF pages", default="200")
    parser.add_argument("-S", "--showlogs", dest="show_logs", help="show logs and keep track of occuring processes", action="store_true")
    parser.add_argument("-V", "--verbose", help="show low level details like image size, width lenght, boundarries etc...", action="store_true")

    args = parser.parse_args()

    if args.input_path and args.output_path:
        ocr = create_ocr(ocrSettings=OCRSettings(lang=args.lang))
        
        if args.input_path.endswith(".pdf"):
            transformPDFSettings = TransformPDFSettings(dpi=args.dpi, show_logs=args.show_logs, verbose=args.verbose)
            transform_pdf(ocr=ocr, input_path=args.input_path, output_path=args.output_path,transformPDFSettings= transformPDFSettings)
        
        elif args.input_path.endswith(".cbz"):
            transformCBZSettings = TransformCBZSettings(show_logs=args.show_logs, verbose=args.verbose)
            transform_cbz(ocr=ocr, input_path=args.input_path, output_path=args.output_path, transformCBZSettings=transformCBZSettings)
        else:
            parser.error("Input must be a PDF or CBZ file.")
