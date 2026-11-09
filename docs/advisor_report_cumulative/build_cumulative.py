"""Assemble the cumulative advisor report: a cover, then one chapter per report, each preceded by a
full-page divider with its date. Chapters 1-2 embed the existing reports' <body> content verbatim
(their folders are read, never written). Chapter 3 is `chapter3_2026_11_09.html`.
Run from this folder: python build_cumulative.py   (then: node render_pdf.js)"""
import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
CHAPTER_1 = DOCS / "advisor_report" / "H2S-NN_progress_report.html"
CHAPTER_2 = DOCS / "advisor_report_2026_11_08" / "H2S-NN_progress_report_2026_11_08.html"
CHAPTER_3 = HERE / "chapter3_2026_11_09.html"
OUT = HERE / "H2S-NN_progress_report_cumulative.html"

# (date from each report's README version history, title, one-line scope).
CHAPTERS = [
    ("8 November 2026", "Scalability and path coupling",
     "Lighter restriction maps, meta-path faces coupled directly to nodes, fact storage, path "
     "sampling and computation, node gates, cost", CHAPTER_1),
    ("8 November 2026", "Encoder and geometry",
     "Data-pipeline check, frequency-selective filter, neighbourhood summary, known-answer test of "
     "the path mechanism, cost", CHAPTER_2),
    ("9 November 2026", "Heterophily of the evaluation datasets",
     "The two heterophily measures, a check against a published value, one table for all "
     "node-classification datasets", CHAPTER_3),
]

EXTRA_CSS = """
  .cover, .divider { break-after: page; min-height: 240mm; display: flex; flex-direction: column;
    justify-content: center; }
  .divider { break-before: page; }
  .cover h1 { font-size: 24pt; margin-bottom: 8pt; }
  .divider .chap { font-family: "Inter","DejaVu Sans",sans-serif; font-size: 11pt; color: var(--muted);
    letter-spacing: 1pt; text-transform: uppercase; margin: 0 0 6pt; }
  .divider .date { font-family: "Inter","DejaVu Sans",sans-serif; font-size: 26pt; font-weight: 700;
    color: var(--accent); margin: 0 0 10pt; }
  .divider .title { font-family: "Inter","DejaVu Sans",sans-serif; font-size: 16pt; font-weight: 650;
    margin: 0 0 8pt; }
  .divider .scope { font-size: 10.5pt; color: var(--muted); max-width: 140mm; }
  .toc { border-top: 1px solid var(--rule); margin-top: 14pt; padding-top: 8pt; }
  .toc td { border-bottom: none; padding: 3pt 5pt; }
"""


def body_of(path: Path) -> str:
    html = path.read_text(encoding="utf-8")
    match = re.search(r"<body[^>]*>(.*)</body>", html, flags=re.S)
    return (match.group(1) if match else html).strip("\n")


def style_of(path: Path) -> str:
    return re.search(r"<style>(.*?)</style>", path.read_text(encoding="utf-8"), flags=re.S).group(1)


def main() -> None:
    before = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in (CHAPTER_1, CHAPTER_2)}
    # Chapter 1's stylesheet is a superset of chapter 2's (chapter 2 lacks only the unused .compare
    # rules), so one stylesheet renders both unchanged.
    css = style_of(CHAPTER_1) + EXTRA_CSS
    toc_rows = "\n".join(
        f'<tr><td class="num">{i}</td><td>{date}</td><td><b>{title}</b><br>'
        f'<span class="small">{scope}</span></td></tr>'
        for i, (date, title, scope, _) in enumerate(CHAPTERS, start=1))
    parts = [
        "<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        "<title>H2S-NN Revision: Cumulative Progress Report</title>", f"<style>{css}</style>",
        "</head>", "<body>",
        '<section class="cover">',
        "<h1>H2S-NN Revision: Cumulative Progress Report</h1>",
        '<p class="subtitle">The progress reports of the revision in one document, one chapter per '
        "report, each opened by a page with its date</p>",
        '<div class="meta"><span>8–9 November 2026</span></div>',
        f'<table class="toc"><thead><tr><th class="num">Chapter</th><th>Date</th><th>Content</th>'
        f"</tr></thead><tbody>{toc_rows}</tbody></table>",
        "</section>",
    ]
    for i, (date, title, scope, path) in enumerate(CHAPTERS, start=1):
        parts += [
            '<section class="divider">', f'<p class="chap">Chapter {i}</p>',
            f'<p class="date">{date}</p>', f'<p class="title">{title}</p>',
            f'<p class="scope">{scope}</p>', "</section>",
            f'<section class="chapter" id="chapter-{i}">', body_of(path), "</section>",
        ]
    parts += ["</body>", "</html>", ""]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    after = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in (CHAPTER_1, CHAPTER_2)}
    assert before == after, "a source report changed"
    for path in (CHAPTER_1, CHAPTER_2):
        assert body_of(path) in OUT.read_text(encoding="utf-8"), f"{path.name} not embedded verbatim"
    print("wrote", OUT.name, "; source reports unchanged:", after)


if __name__ == "__main__":
    main()
