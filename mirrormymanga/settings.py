from dataclasses import dataclass

@dataclass(slots=True, kw_only=True)
class OCRSettings:
    lang: str = "en"
    device:str = "auto"
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    enable_mkldnn: bool = True

@dataclass(slots=True, kw_only=True)
class TransformPanelSettings:
    show_logs: bool = True
    verbose: bool = False

@dataclass(slots=True, kw_only=True)
class TransformPDFSettings:
    dpi: int = 200
    show_logs: bool = True
    verbose: bool = False

@dataclass(slots=True, kw_only=True)
class TransformCBZSettings:
    show_logs: bool = True
    verbose: bool = False