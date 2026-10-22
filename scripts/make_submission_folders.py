#!/usr/bin/env python3
"""Compile both manuscripts and assemble release/<paper>_submission/ plus release/<paper>_submission.zip.

Only files the .tex actually references are copied; the folder is rebuilt from scratch and checked to compile on its own.
"""
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

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
        figs = sorted(set(re.findall(r"figures/[\w\-]+\.(?:pdf|png|jpg|tex)", tex)))
        for f in figs:
            shutil.copy(d / f, sub / f)
        # check that the folder compiles on its own
        chk_pages, chk_undef = latex(sub, base)
        for f in sub.glob(f"{base}.*"):
            if f.suffix in JUNK:
                f.unlink()
        zpath = REL / f"{name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(sub.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(sub))
        print(f"{paper}: {pages} pages, undefined: {undef or 'none'}; submission folder compiles to {chk_pages} pages "
              f"(undefined: {chk_undef or 'none'}); {len(figs)} figure files; {zpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
