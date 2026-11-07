"""Synthetic PDFs for the offline corpus tests, written with pypdf (Courier: 600/1000 em per glyph).

Used by tests/corpus/test_pdf_text_recipe.py.
"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

CHAR_W = 7.2  # Courier at 12 pt


def _stream(writer: PdfWriter, data: bytes, extra: dict | None = None):
    obj = DecodedStreamObject()
    obj.set_data(data)
    for k, v in (extra or {}).items():
        obj[NameObject(k)] = v
    return writer._add_object(obj)


def make_pdf(path: Path, contents: list[bytes], with_image: bool = False) -> Path:
    """One US-letter page per content stream; font /F1 = Courier, optional image /Im1 (1x1 grey)."""
    w = PdfWriter()
    font = w._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Courier"), NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
    }))
    image = _stream(w, b"\x00", {
        "/Type": NameObject("/XObject"), "/Subtype": NameObject("/Image"), "/Width": NumberObject(1),
        "/Height": NumberObject(1), "/ColorSpace": NameObject("/DeviceGray"), "/BitsPerComponent": NumberObject(8),
    })
    for data in contents:
        page = w.add_blank_page(612, 792)
        res = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        if with_image:
            res[NameObject("/XObject")] = DictionaryObject({NameObject("/Im1"): image})
        page[NameObject("/Resources")] = res
        page[NameObject("/Contents")] = _stream(w, data)
    with path.open("wb") as fh:
        w.write(fh)
    return path


def text_op(x: float, y: float, s: str, rgb: str = "0 0 0") -> bytes:
    """Show ``s`` at (x, y) in 12 pt Courier with fill colour ``rgb`` (PDF string-literal escapes applied)."""
    lit = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"BT /F1 12 Tf {rgb} rg {x} {y} Td ({lit}) Tj ET\n".encode("latin-1")


def rule(x0: float, x1: float, y: float, rgb: str = "1 0 0") -> bytes:
    """A 0.8 pt stroked horizontal rule (an underline or strike-through when it sits on a text line)."""
    return f"{rgb} RG 0.8 w {x0} {y} m {x1} {y} l S\n".encode("latin-1")


def artifact(data: bytes) -> bytes:
    """Wrap content as a running header/footer the way Word tags it (/Artifact <</Type /Pagination>>)."""
    return b"/Artifact <</Type /Pagination>> BDC\n" + data + b"EMC\n"
