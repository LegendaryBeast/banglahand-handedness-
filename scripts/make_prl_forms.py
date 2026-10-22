#!/usr/bin/env python3
"""Fill PRL's mandatory "Authorship Confirmation" form (Elsevier patrec-authorship-and-formatting.pdf).

Writes manuscripts/prl/forms/authorship_confirmation.pdf with the form's first three pages:
  1  Authorship Confirmation (corresponding author, date; no preprints, no conference versions)
  2  Graphical abstract page (title, authors, manuscripts/prl/graphical_abstract/graphical_abstract.png, short text)
  3  Research highlights page (manuscripts/prl/highlights.txt; each at most 85 characters)
The signature line is left blank for the corresponding author. Upload as item type 'Cover Letter'.
"""
from pathlib import Path

import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PRL = ROOT / "manuscripts/prl"
FORM = PRL / "forms/patrec-authorship-and-formatting_official.pdf"
TITLE = "Writer-independent handedness identification from offline Bangla handwriting under controlled acquisition"
AUTHORS = "Tanzim Hasan Prappo, Sayeed Hassan, Sayem Ali, Sukhendra Nath Dhar, M. Shahidur Rahman"
CORRESPONDING, DATE = "Tanzim Hasan Prappo", "22 October 2026"
GA_TEXT = ("Handedness of unseen writers is identified from photographed Bangla paragraphs (77% balanced accuracy, "
           "p = 0.001). Part of the cue is darker stroke centres in right-handed writers; with darkness and brightness "
           "removed, the gradient descriptor keeps 65% while CNN features fall near chance.")
F, S = "tiro", 9.96


def box(page, rect, text, font=F):
    assert page.insert_textbox(rect, text, fontname=font, fontsize=S) >= 0, text[:40]


def main():
    d = fitz.open(); d.insert_pdf(fitz.open(FORM), from_page=0, to_page=2)
    p = d[0]
    p.insert_text((156, 231.5), CORRESPONDING, fontname=F, fontsize=S)
    p.insert_text((305, 418.6), DATE, fontname=F, fontsize=S)
    for y, t in [(485, "None."), (581, "None."), (653, "Not applicable.")]:
        p.insert_text((42.5, y), t, fontname=F, fontsize=S)

    p = d[1]                                              # replace the template placeholders inside the box
    p.add_redact_annot(fitz.Rect(44, 206, 544, 623)); p.apply_redactions()
    for a, b in [((42.5, 204.8), (546.2, 204.8)), ((42.7, 204.8), (42.7, 625.5)), ((546.0, 204.8), (546.0, 625.5)),
                 ((42.5, 625.5), (546.2, 625.5))]:
        p.draw_line(a, b, color=(0, 0, 0), width=0.4)
    box(p, fitz.Rect(54.4, 210, 536, 238), TITLE, "tibo")
    box(p, fitz.Rect(54.4, 238, 536, 256), AUTHORS)
    ga = PRL / "graphical_abstract/graphical_abstract.png"
    w, h = Image.open(ga).size
    r = fitz.Rect(52, 262, 536, 262 + 484 * h / w)
    p.insert_image(r, filename=str(ga))
    box(p, fitz.Rect(54.4, r.y1 + 10, 536, 620), GA_TEXT)

    p = d[2]
    hl = [l[2:].strip() for l in (PRL / "highlights.txt").read_text().splitlines() if l.strip()]
    assert 3 <= len(hl) <= 5
    for k, h in enumerate(hl):
        assert len(h) <= 85, h
        p.insert_text((71.0, 385.6 + 19.93 * k), h, fontname=F, fontsize=S)
    out = PRL / "forms/authorship_confirmation.pdf"
    d.save(out, garbage=3, deflate=True)
    print("written", out)


if __name__ == "__main__":
    main()
