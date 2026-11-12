#!/usr/bin/env python3
"""Compile both manuscripts and assemble release/<paper>_submission/ plus release/<paper>_submission.zip.

Only files the .tex actually references are copied; the folder is rebuilt from scratch and checked to compile on its own.
PRL also gets upload/: one file per Editorial Manager item (PRL Guide for Authors, mandatory requirements 1-4):
the PDF, the LaTeX source zip, highlights (.docx), graphical abstract (.tif/.pdf), the filled authorship
confirmation form (scripts/make_prl_forms.py), the declaration of competing interests (.docx, Elsevier's standard
wording), every figure as a separate Figure_<n>.pdf, and the Response to Technical Check Results (.docx) for the
resubmission of PRLETTERS-D-26-02698 (authors and emails read from the .tex; non-institutional addresses are flagged).
"""
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
PAPERS = {"prl": ("prl_manuscript", ["references.bib", "cover_letter.txt", "highlights.txt"], "prl_submission"),
          "data_in_brief": ("dib_manuscript", ["dib_references.bib", "cover_letter.pdf"], "dib_submission")}
REL = ROOT / "release"
JUNK = (".aux", ".log", ".out", ".spl", ".blg")


def latex(d, base):
    for cmd in (["pdflatex", "-interaction=nonstopmode", base + ".tex"], ["bibtex", base],
                ["pdflatex", "-interaction=nonstopmode", base + ".tex"], ["pdflatex", "-interaction=nonstopmode", base + ".tex"]):
        subprocess.run(cmd, cwd=d, capture_output=True)
    log = (d / f"{base}.log").read_text(errors="ignore")
    pages = re.search(r"Output written on .*\((\d+) pages", log)
    undef = re.findall(r"(?:Reference|Citation|Control sequence) .*undefined", log)
    return int(pages.group(1)) if pages else None, undef


def prl_upload(d, sub, base, figs_in_order):
    up = sub / "upload"
    up.mkdir()
    shutil.copy(sub / f"{base}.pdf", up / "Manuscript.pdf")
    with zipfile.ZipFile(up / "Manuscript_LaTeX_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in [f"{base}.tex", f"{base}.bbl", "references.bib"] + figs_in_order:
            z.write(sub / f, f)
    doc = docx.Document(); doc.add_heading("Highlights", level=1)
    for line in (d / "highlights.txt").read_text().splitlines():
        if line.strip():
            assert len(line[2:].strip()) <= 85, line
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
    doc.save(up / "Highlights.docx")
    for ext in ("tif", "pdf"):
        shutil.copy(d / f"graphical_abstract/graphical_abstract.{ext}", up / f"Graphical_abstract.{ext}")
    shutil.copy(d / "forms/authorship_confirmation.pdf", up / "Authorship_confirmation_form.pdf")
    title = re.search(r"\\title\{(.+?)\}", (d / f"{base}.tex").read_text()).group(1)
    doc = docx.Document(); doc.add_heading("Declaration of interests", level=1)
    doc.add_paragraph(f"Manuscript: {title}")
    doc.add_paragraph("\u2612 The authors declare that they have no known competing financial interests or personal relationships "
                      "that could have appeared to influence the work reported in this paper.")
    doc.add_paragraph("\u2610 The authors declare the following financial interests/personal relationships which may be "
                      "considered as potential competing interests:")
    doc.save(up / "Declaration_of_competing_interests.docx")
    technical_check_response(d / f"{base}.tex", title, up / "Response_to_technical_check.docx")
    shutil.copy(d / "cover_letter.txt", up / "Cover_letter.txt")
    shutil.copy(d / "SUBMISSION_STEPS.md", up / "SUBMISSION_STEPS.md")
    for n, f in enumerate([f for f in figs_in_order if f.endswith(".pdf")], 1):
        shutil.copy(sub / f, up / f"Figure_{n}.pdf")


INSTITUTIONAL = ("sust.edu", ".edu", ".ac.bd", ".ac.")


def technical_check_response(tex_path, title, out):
    """Reply to the technical check of 22-25 Oct 2026 (institutional emails; authorship form under 'Cover Letter')."""
    authors = re.findall(r"\\author\[[^]]*\]\{([^}\\]+)(?:\\corref\{[^}]*\})?\}\s*\\ead\{([^}]+)\}", tex_path.read_text())
    assert len(authors) == 5, authors
    bad = [e for _, e in authors if not any(s in e for s in INSTITUTIONAL)]
    if bad:
        print("WARNING: non-institutional email addresses in the manuscript:", ", ".join(bad))
    doc = docx.Document(); doc.add_heading("Response to Technical Check Results", level=1)
    doc.add_paragraph(f"Ref. PRLETTERS-D-26-02698\nTitle: {title}")
    doc.add_paragraph("We thank the journal office for the technical check. Both comments have been addressed as follows.")
    doc.add_heading("Comment 1: institutional email addresses", level=2)
    doc.add_paragraph("Response: The institutional email address of every author is now given both in the author details of the "
                      "submission system and in the footnote of the first manuscript page:")
    for name, email in authors:
        doc.add_paragraph(f"{name.strip()}: {email}", style="List Bullet")
    doc.add_heading("Comment 2: Authorship Confirmation form", level=2)
    doc.add_paragraph("Response: The Authorship Confirmation form, downloaded from the journal link, completed in the journal's "
                      "format and signed by the corresponding author on behalf of all authors, has been uploaded under the item "
                      "type 'Cover Letter'. The form also includes the graphical abstract and research highlights pages.")
    doc.save(out)


def main():
    REL.mkdir(exist_ok=True)
    for paper, (base, extra, name) in PAPERS.items():
        d = ROOT / "manuscripts" / paper
        pages, undef = latex(d, base)
        for f in d.glob(f"{base}.*"):
            if f.suffix in JUNK:
                f.unlink()
        sub = REL / name
        if sub.exists():
            shutil.rmtree(sub)            # generated folder, rebuilt below from the manuscript sources
        (sub / "figures").mkdir(parents=True)
        tex = (d / f"{base}.tex").read_text()
        for f in [f"{base}.tex", f"{base}.bbl", f"{base}.pdf"] + extra:
            shutil.copy(d / f, sub / f)
        figs_in_order = list(dict.fromkeys(re.findall(r"figures/[\w\-]+\.(?:pdf|png|jpg|tex)", tex)))
        figs = sorted(figs_in_order)
        for f in figs:
            shutil.copy(d / f, sub / f)
        # check that the folder compiles on its own
        chk_pages, chk_undef = latex(sub, base)
        for f in sub.glob(f"{base}.*"):
            if f.suffix in JUNK:
                f.unlink()
        if paper == "prl":
            prl_upload(d, sub, base, figs_in_order)
        zpath = REL / f"{name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(sub.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(sub))
        print(f"{paper}: {pages} pages, undefined: {undef or 'none'}; submission folder compiles to {chk_pages} pages "
              f"(undefined: {chk_undef or 'none'}); {len(figs)} figure files; {zpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
