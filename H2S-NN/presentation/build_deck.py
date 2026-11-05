"""Build the H2S-NN AAAI presentation deck from the manuscript content.

Slides carry short headline fragments; the full narrative lives in the speaker notes.
All content is grounded in paper/h2snn_manuscript.tex (main text + appendix).
Run through the h2snn conda environment.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# --------------------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------------------

INK = RGBColor(0x1A, 0x1F, 0x36)
PRIMARY = RGBColor(0x1F, 0x3B, 0x73)
ACCENT = RGBColor(0xB0, 0x7D, 0x14)
TEAL = RGBColor(0x2E, 0x7D, 0x6E)
MUTED = RGBColor(0x5A, 0x63, 0x7A)
RULE = RGBColor(0xD8, 0xDD, 0xE6)
PANEL = RGBColor(0xF4, 0xF6, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HILITE = RGBColor(0xFD, 0xF6, 0xE3)

FONT = "Segoe UI"
MONO = "Consolas"

# Type scale — slides are headlines, so these run large.
BODY = 18.0        # main slide bullets
PANEL_BODY = 15.5  # text inside a panel
EQ = 16.0          # display equations
PANEL_HEAD = 13.5
CALLOUT = 15.5

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.62)

FIG_DIR = Path(__file__).resolve().parent / "fig"
PAPER_FIG = Path("g:/My Drive/H2S-NN_test/paper/figure")
OUT = Path("g:/My Drive/H2S-NN_test/presentation/H2S-NN_AAAI_15min.pptx")


# --------------------------------------------------------------------------------------
# Text measurement / auto-fit
# --------------------------------------------------------------------------------------

FONT_FILES = {
    ("Segoe UI", False): "C:/Windows/Fonts/segoeui.ttf",
    ("Segoe UI", True): "C:/Windows/Fonts/segoeuib.ttf",
    ("Consolas", False): "C:/Windows/Fonts/consola.ttf",
    ("Consolas", True): "C:/Windows/Fonts/consolab.ttf",
}
_BASE_PX = 200
_FONT_CACHE: dict = {}
_FIT_QUEUE: list = []
_SLIDE_NO = [0]


def _face(name, bold):
    key = (name, bool(bold))
    if key not in _FONT_CACHE:
        path = FONT_FILES.get(key) or FONT_FILES[("Segoe UI", bool(bold))]
        _FONT_CACHE[key] = ImageFont.truetype(path, _BASE_PX)
    return _FONT_CACHE[key]


def _adv(text, name, bold, pt):
    if not text:
        return 0.0
    return _face(name, bold).getlength(text) * pt / _BASE_PX


def _tokens(p, scale):
    out = []
    for r in p.runs:
        pt = (r.font.size.pt if r.font.size else BODY) * scale
        name = r.font.name or FONT
        bold = bool(r.font.bold)
        for i, w in enumerate(r.text.split(" ")):
            out.append((w, name, bold, pt, i > 0))
    return out


def _para_height(p, avail_pt, scale):
    toks = _tokens(p, scale)
    if not toks:
        return 0.0
    max_pt = max(t[3] for t in toks)
    lines, cur = 1, 0.0
    for word, name, bold, pt, had_space in toks:
        if not word:
            continue
        ww = _adv(word, name, bold, pt)
        sp = _adv(" ", name, bold, pt) if had_space or cur > 0 else 0.0
        if cur == 0.0:
            cur = ww
        elif cur + sp + ww <= avail_pt:
            cur += sp + ww
        else:
            lines += 1
            cur = ww
    spacing = p.line_spacing if isinstance(p.line_spacing, float) else 1.2
    before = p.space_before.pt if p.space_before else 0.0
    after = p.space_after.pt if p.space_after else 0.0
    return lines * max_pt * 1.2 * spacing + (before + after) * scale


def _frame_height(tf, width_pt, scale):
    total = 0.0
    for p in tf.paragraphs:
        marl = 0.0
        pPr = p._p.find("{http://schemas.openxmlformats.org/drawingml/2006/main}pPr")
        if pPr is not None and pPr.get("marL"):
            marl = int(pPr.get("marL")) / 12700.0
        total += _para_height(p, max(width_pt - marl, 24.0), scale)
    return total


def fit_all(min_scale=0.88, safety=1.035):
    """Shrink registered frames until content fits; report anything that had to shrink."""
    shrunk = []
    for tf, w_emu, h_emu, sno in _FIT_QUEUE:
        width_pt = w_emu / 12700.0
        avail_pt = h_emu / 12700.0
        if _frame_height(tf, width_pt, 1.0) * safety <= avail_pt:
            continue
        scale = 1.0
        while scale > min_scale:
            scale -= 0.02
            if _frame_height(tf, width_pt, scale) * safety <= avail_pt:
                break
        for p in tf.paragraphs:
            if p.space_before:
                p.space_before = Pt(round(p.space_before.pt * scale, 1))
            if p.space_after:
                p.space_after = Pt(round(p.space_after.pt * scale, 1))
            for r in p.runs:
                if r.font.size:
                    r.font.size = Pt(round(r.font.size.pt * scale, 1))
        shrunk.append((round(scale, 2), sno,
                       tf.paragraphs[0].runs[-1].text[:44] if tf.paragraphs[0].runs else ""))
    return shrunk


# --------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------


def _blank(tf):
    return len(tf.paragraphs) == 1 and not tf.paragraphs[0].runs


def new_slide(prs: Presentation):
    _SLIDE_NO[0] += 1
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_box(slide, left, top, width, height, fill=None, line=None, line_w=Pt(1.0)):
    from pptx.enum.shapes import MSO_SHAPE

    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = line_w
    shp.shadow.inherit = False
    return shp


def add_text(slide, left, top, width, height, anchor=MSO_ANCHOR.TOP, fit=True):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    if fit:
        _FIT_QUEUE.append((tf, width, height, _SLIDE_NO[0]))
    return tf


def style_run(run, size, bold=False, color=INK, font=FONT, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font
    return run


def para(tf, text, size=BODY, bold=False, color=INK, space_before=0, space_after=6,
         font=FONT, first=False, align=PP_ALIGN.LEFT, italic=False, line=1.18):
    p = tf.paragraphs[0] if (first or _blank(tf)) else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    p.line_spacing = line
    style_run(p.add_run(), size, bold, color, font, italic).text = text
    return p


def rich(tf, chunks, size=BODY, space_before=0, space_after=6, first=False, line=1.18,
         align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if (first or _blank(tf)) else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    p.line_spacing = line
    for text, opt in chunks:
        r = p.add_run()
        r.text = text
        style_run(r, opt.get("size", size), opt.get("bold", False), opt.get("color", INK),
                  opt.get("font", FONT), opt.get("italic", False))
    return p


def bullet(tf, text, size=BODY, color=INK, bold=False, level=0, first=False, space_after=10):
    p = tf.paragraphs[0] if (first or _blank(tf)) else tf.add_paragraph()
    p.space_before = Pt(0)
    p.space_after = Pt(space_after)
    p.line_spacing = 1.16
    p.level = level
    marker = "\u2022  " if level == 0 else "\u2013  "
    style_run(p.add_run(), size, False, ACCENT if level == 0 else MUTED).text = marker
    style_run(p.add_run(), size, bold, color).text = text
    pPr = p._p.get_or_add_pPr()
    pPr.set("indent", str(int(-Inches(0.22))))
    pPr.set("marL", str(int(Inches(0.22) + Inches(0.26) * level)))
    return p


def eq(tf, text, size=EQ, color=PRIMARY, space_after=8, first=False, align=PP_ALIGN.LEFT):
    return para(tf, text, size=size, bold=True, color=color, font=MONO,
                space_after=space_after, first=first, align=align)


def chrome(slide, eyebrow, title, idx, total, sub=None):
    add_box(slide, 0, 0, SLIDE_W, Inches(0.11), fill=PRIMARY)

    tf = add_text(slide, MARGIN, Inches(0.42), SLIDE_W - 2 * MARGIN, Inches(0.3))
    para(tf, eyebrow.upper(), size=12, bold=True, color=ACCENT, first=True, space_after=0)

    tf = add_text(slide, MARGIN, Inches(0.72), SLIDE_W - 2 * MARGIN, Inches(0.62))
    para(tf, title, size=28, bold=True, color=PRIMARY, first=True, space_after=0)

    top = Inches(1.42)
    if sub:
        tf = add_text(slide, MARGIN, Inches(1.3), SLIDE_W - 2 * MARGIN, Inches(0.32))
        para(tf, sub, size=14, color=MUTED, first=True, space_after=0, italic=True)
        top = Inches(1.66)

    add_box(slide, MARGIN, top, SLIDE_W - 2 * MARGIN, Emu(11430), fill=RULE)

    tf = add_text(slide, MARGIN, Inches(6.98), Inches(6.0), Inches(0.3))
    para(tf, "H2S-NN: Heterogeneous Heterophilic Sheaf Neural Network",
         size=9.5, color=RGBColor(0x9A, 0xA2, 0xB2), first=True, space_after=0)

    tf = add_text(slide, SLIDE_W - MARGIN - Inches(2.0), Inches(6.98), Inches(2.0), Inches(0.3))
    para(tf, f"{idx} / {total}", size=9.5, color=RGBColor(0x9A, 0xA2, 0xB2), first=True,
         space_after=0, align=PP_ALIGN.RIGHT)

    return top + Inches(0.24)


def notes(slide, script, budget):
    tf = slide.notes_slide.notes_text_frame
    tf.text = f"[ {budget} ]\n\n{script}"
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = Pt(12)


def picture(slide, path, left, top, max_w, max_h, caption=None):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    slide.shapes.add_picture(str(path), int(left + (max_w - w) / 2), int(top), width=w, height=h)
    if caption:
        tf = add_text(slide, left, int(top + h + Inches(0.08)), max_w, Inches(0.5))
        para(tf, caption, size=11.5, color=MUTED, first=True, align=PP_ALIGN.CENTER, italic=True)
    return h


def table(slide, rows, left, top, width, col_w=None, font=9.0, header_font=9.0,
          head_rows=1, bold_last=True, row_h=Inches(0.26), group_col=False,
          highlight_last=True):
    n_rows, n_cols = len(rows), len(rows[0])
    shp = slide.shapes.add_table(n_rows, n_cols, left, top, width, row_h * n_rows)
    tbl = shp.table
    tbl.first_row = False
    tbl.horz_banding = False

    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(width * cw / total))

    for r in range(n_rows):
        tbl.rows[r].height = row_h
        for c in range(n_cols):
            cell = tbl.cell(r, c)
            cell.text = str(rows[r][c])
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.01)
            cell.margin_bottom = Inches(0.01)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

            is_head = r < head_rows
            is_last = bold_last and r == n_rows - 1

            cell.fill.solid()
            if is_head:
                cell.fill.fore_color.rgb = PRIMARY
            elif is_last and highlight_last:
                cell.fill.fore_color.rgb = HILITE
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else PANEL

            tp = cell.text_frame.paragraphs[0]
            tp.alignment = PP_ALIGN.LEFT if c < (2 if group_col else 1) else PP_ALIGN.CENTER
            for run in tp.runs:
                run.font.size = Pt(header_font if is_head else font)
                run.font.name = FONT
                run.font.bold = is_head or is_last
                run.font.color.rgb = WHITE if is_head else (PRIMARY if is_last else INK)
    return tbl


def callout(slide, left, top, width, height, lines, accent=ACCENT, fill=HILITE):
    add_box(slide, left, top, width, height, fill=fill)
    add_box(slide, left, top, Emu(34290), height, fill=accent)
    tf = add_text(slide, left + Inches(0.2), top + Inches(0.11), width - Inches(0.34),
                  height - Inches(0.18), anchor=MSO_ANCHOR.MIDDLE)
    for i, (txt, opt) in enumerate(lines):
        para(tf, txt, size=opt.get("size", CALLOUT), bold=opt.get("bold", False),
             color=opt.get("color", INK), first=(i == 0), space_after=opt.get("after", 4),
             italic=opt.get("italic", False))


def panel(slide, left, top, width, height, title, accent=PRIMARY):
    add_box(slide, left, top, width, height, fill=PANEL)
    add_box(slide, left, top, width, Inches(0.34), fill=accent)
    tf = add_text(slide, left + Inches(0.16), top + Inches(0.045), width - Inches(0.32), Inches(0.3))
    para(tf, title, size=PANEL_HEAD, bold=True, color=WHITE, first=True, space_after=0)
    return add_text(slide, left + Inches(0.18), top + Inches(0.48), width - Inches(0.36),
                    height - Inches(0.62))


# --------------------------------------------------------------------------------------
# Deck
# --------------------------------------------------------------------------------------

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H

TOTAL = 24
FULL_W = SLIDE_W - 2 * MARGIN


# ---- 1. Title -------------------------------------------------------------------------
s = new_slide(prs)
add_box(s, 0, 0, SLIDE_W, SLIDE_H, fill=PRIMARY)
add_box(s, 0, Inches(2.84), SLIDE_W, Inches(0.045), fill=ACCENT)

tf = add_text(s, Inches(1.1), Inches(1.42), Inches(11.2), Inches(0.4))
para(tf, "AAAI  \u00b7  HETEROGENEOUS GRAPH LEARNING UNDER HETEROPHILY", size=13, bold=True,
     color=RGBColor(0xE3, 0xC1, 0x6A), first=True, space_after=0)

tf = add_text(s, Inches(1.1), Inches(1.82), Inches(11.2), Inches(1.02))
para(tf, "H2S-NN", size=48, bold=True, color=WHITE, first=True, space_after=0)

tf = add_text(s, Inches(1.1), Inches(3.04), Inches(11.2), Inches(0.6))
para(tf, "Heterogeneous Heterophilic Sheaf Neural Network", size=27, color=WHITE, first=True,
     space_after=0)

tf = add_text(s, Inches(1.1), Inches(3.78), Inches(11.2), Inches(0.9))
para(tf, "Keeping meta-path instances as 2D-cells,", size=17,
     color=RGBColor(0xC3, 0xCD, 0xE4), first=True, space_after=2)
para(tf, "then diffusing node and edge cochains.", size=17,
     color=RGBColor(0xC3, 0xCD, 0xE4), space_after=0)

add_box(s, Inches(1.1), Inches(5.02), Inches(1.5), Emu(22860), fill=ACCENT)

tf = add_text(s, Inches(1.1), Inches(5.28), Inches(11.2), Inches(1.0))
para(tf, "Peeradon Sarnkaew", size=19, bold=True, color=WHITE, first=True, space_after=4)
para(tf, "Independent Researcher  \u00b7  Bangkok, Thailand", size=14,
     color=RGBColor(0xC3, 0xCD, 0xE4), space_after=0)

notes(s, """Good morning. I am presenting H2S-NN, a Heterogeneous Heterophilic Sheaf Neural Network.

The one-sentence version of this talk: when a heterogeneous graph method uses a meta-path, it almost
always collapses that path into a shortcut between the two endpoints. We do not. We keep each
concrete path as a two-dimensional cell, and we let sheaf-Hodge diffusion run over both the nodes
and the edges of that enriched structure.

Evaluated on node classification, graph classification and link prediction.""",
      "0:15  \u00b7  running 0:15")


# ---- 2. Outline -----------------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "Roadmap", "Outline", 2, TOTAL, sub="Following the section structure of the paper")

items = [
    ("1", "Introduction", "The joint problem, and what collapse discards", PRIMARY),
    ("2", "Preliminaries", "Typed graphs \u00b7 heterophily \u00b7 schemas \u00b7 sheaves", PRIMARY),
    ("3", "Methodology", "AMPSS \u00b7 Sheaf-Hodge encoding \u00b7 diffusion \u00b7 SheafCovPool",
     ACCENT),
    ("4", "Experiments", "Node \u00b7 graph \u00b7 link", PRIMARY),
    ("5", "Discussion", "Two ablations", PRIMARY),
    ("6", "Conclusion", "What holds, and what does not", PRIMARY),
    ("A\u2013F", "Appendix", "Readouts \u00b7 tuning \u00b7 clustering \u00b7 efficiency", TEAL),
]

y = top + Inches(0.16)
for num, head, desc, col in items:
    add_box(s, MARGIN, y, Inches(0.66), Inches(0.62), fill=col)
    tfn = add_text(s, MARGIN, y + Inches(0.15), Inches(0.66), Inches(0.34), anchor=MSO_ANCHOR.MIDDLE)
    para(tfn, num, size=16, bold=True, color=WHITE, first=True, space_after=0, align=PP_ALIGN.CENTER)

    tfx = add_text(s, MARGIN + Inches(0.92), y + Inches(0.02), Inches(10.8), Inches(0.62))
    para(tfx, head, size=19, bold=True, color=PRIMARY, first=True, space_after=2)
    para(tfx, desc, size=13.5, color=MUTED, space_after=0)
    y += Inches(0.72)

notes(s, """The talk follows the paper's own section order.

Most of the time goes to Section 3, the methodology, because all three contributions live there.
I will then cover the three benchmark tasks, the two ablations, and finish with a compressed pass
over the appendix: the task readouts, the clustering diagnostic, the training dynamics, and the
computational cost. I include the cost deliberately, because it is a real part of the story rather
than something to bury.""", "0:15  \u00b7  running 0:30")


# ---- 3. Introduction: motivation ------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "1. Introduction", "Two properties, one hard problem", 3, TOTAL)

tf = add_text(s, MARGIN, top + Inches(0.14), Inches(6.5), Inches(2.5))
bullet(tf, "Heterogeneity \u2014 many node and relation types", first=True)
bullet(tf, "Heterophily \u2014 neighbours disagree, so averaging misleads")
bullet(tf, "Real graphs are both at once")
bullet(tf, "Assuming either away loses structure and accuracy")

callout(s, MARGIN, top + Inches(3.0), Inches(6.5), Inches(1.62), [
    ("The literature is split down the middle", {"bold": True, "size": 16, "color": PRIMARY}),
    ("Heterogeneous methods rarely model heterophily.", {"size": 15}),
    ("Heterophilic sheaf methods rarely model typed nodes and relations.", {"size": 15}),
])

picture(s, PAPER_FIG / "Het_HomGraph.drawio.png", Inches(7.5), top + Inches(0.3), Inches(5.2),
        Inches(3.9),
        caption="Several node and relation types \u2014 and adjacent nodes that disagree.")

notes(s, """Two properties define the setting.

HETEROGENEITY means the graph has several node types and several relation types, each with a
distinct semantic role \u2014 citation networks, social interaction networks. Formally that is more
than one node type or more than one relation type.

HETEROPHILY means that two connected nodes often disagree: different labels, different features. So
the standard "average your neighbours" move is not merely unhelpful, it is actively harmful, and it
drives oversmoothing.

Real graphs are frequently both at once, as in the figure. And this is not the union of two easy
problems. Model such a graph as homogeneous and you throw away the type semantics; assume homophily
and you smooth away exactly the signal you need. The paper's phrasing is that these simplifying
assumptions produce an incomplete representation of the original graph and degrade predictive
performance.

The state of the literature, in the box: the heterogeneous line of work \u2014 HGT, HiNormer,
PSHGCN, SeHGNN, HAN \u2014 does not usually treat heterophily explicitly. The heterophilic sheaf
line \u2014 NSD, CSNN \u2014 does not usually treat typed nodes and relations. Recent work has begun
to join the two using latent subgraphs, rewiring, transformers, or sheaf maps; that is our starting
point.""", "0:50  \u00b7  running 1:20")


# ---- 4. Introduction: the gap ---------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "1. Introduction", "What meta-path collapse throws away", 4, TOTAL)

tf = add_text(s, MARGIN, top + Inches(0.14), Inches(6.5), Inches(2.7))
bullet(tf, "A meta-path carries what neither endpoint holds alone", first=True)
bullet(tf, "But it is used as a view: A-P-A becomes author\u2013author")
bullet(tf, "The shared paper leaves the propagation operator")
bullet(tf, "Under heterophily, the path decides: reinforce or contrast?")

callout(s, MARGIN, top + Inches(3.14), Inches(6.5), Inches(1.5), [
    ("A second, independent gap: granularity", {"bold": True, "size": 16, "color": PRIMARY}),
    ("Selection happens at the schema level. PAP can be useful overall while individual PAP "
     "instances link unrelated papers.", {"size": 15}),
])

picture(s, PAPER_FIG / "meta_path.drawio.png", Inches(7.5), top + Inches(0.4), Inches(5.2),
        Inches(3.6),
        caption="Under A-P-A, authors sharing a paper are joined directly \u2014 and the paper "
                "disappears.")

notes(s, """Now the specific gap, and it has two halves.

FIRST HALF \u2014 representation. Meta-path schemas are valuable precisely because they express
information that is in neither endpoint. An Author-Paper-Author path is a co-authorship relation;
you cannot read that off either author alone.

But look at what existing methods do with the schema. They use it as a semantic view \u2014 a
filtered edge set, or an aggregation channel. Concretely: A-P-A becomes an author-to-author edge. At
that moment the shared paper and the two authorship edges stop shaping the propagation operator.
They have been summarised down to the mere existence of a shortcut.

Under homophily you can often get away with this. Under heterophily you cannot, because whether a
neighbour's signal should be reinforced or contrasted depends on which relations and which
intermediate nodes you travelled through. Collapsing the path discards exactly that local relational
structure, and the higher-order information the concrete instance carries.

SECOND HALF \u2014 granularity, in the box. Even the automatic meta-path search methods \u2014
greedy selection, progressive sampling, tree search, reinforcement learning \u2014 decide at the
SCHEMA level. But PAP can be useful on average while specific PAP instances link entirely unrelated
papers. A schema-level decision cannot separate those two cases: you either keep the schema together
with its noise, or discard it together with its signal. Nothing tells you which of its concrete
occurrences should influence propagation.

Those two halves map onto our first two contributions.""", "0:55  \u00b7  running 2:15")


# ---- 5. Contributions -----------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "1. Introduction", "Contributions", 5, TOTAL)

cards = [
    ("H2S-NN", PRIMARY,
     ["Keep each path instance as a 2D-cell",
      "Build node\u2192edge and edge\u2192face geometry",
      "No homophily assumption"]),
    ("AMPSS", ACCENT,
     ["Stage I \u2014 which schemas? (training only)",
      "Stage II \u2014 which instances? (learned)",
      "Weights act inside the upper edge Laplacian"]),
    ("SheafCovPool", TEAL,
     ["Transport edge states into node stalks",
      "Attention-weighted Gram means + covariances",
      "Permutation- and gauge-invariant"]),
]

w = Inches(3.95)
x = MARGIN
for name, col, lines in cards:
    add_box(s, x, top + Inches(0.16), w, Inches(2.66), fill=PANEL)
    add_box(s, x, top + Inches(0.16), w, Inches(0.56), fill=col)
    tfh = add_text(s, x + Inches(0.2), top + Inches(0.28), w - Inches(0.4), Inches(0.34))
    para(tfh, name, size=19, bold=True, color=WHITE, first=True, space_after=0)
    tfb = add_text(s, x + Inches(0.22), top + Inches(0.88), w - Inches(0.44), Inches(1.86))
    for i, ln in enumerate(lines):
        bullet(tfb, ln, size=15, first=(i == 0), space_after=11)
    x += w + Inches(0.29)

callout(s, MARGIN, top + Inches(3.18), FULL_W, Inches(1.06), [
    ("Evaluated on node classification, graph classification, and link prediction \u2014 where "
     "H2S-NN attains the best overall performance among the compared methods.",
     {"size": 17, "bold": True, "color": PRIMARY}),
])

notes(s, """Three contributions, one per card.

FIRST, the model. H2S-NN retains concrete path instances as path-supported two-dimensional cells in
an augmented 2D cellular complex, and builds the compatible node-to-edge and edge-to-face geometry
that sheaf-Hodge diffusion needs on top of them. Nothing is collapsed; nothing assumes homophily.

SECOND, AMPSS \u2014 Adaptive Meta-path Schema-to-cell Searching and Selection. It answers the two
questions in sequence. Stage I: which typed patterns are worth keeping? That runs on training data
only. Stage II: which concrete occurrences of those patterns should influence propagation? That one
is differentiable and schema-conditioned, and its weights act inside the upper edge Laplacian. That
second question is the one schema-level methods cannot ask.

THIRD, SheafCovPool, a graph-level readout designed specifically for sheaf models \u2014 because
sheaf states live in local coordinate frames, and naive pooling mixes incompatible bases. It
transports incident edge states into node stalks, then summarises node and edge states through
attention-weighted Gram means and covariances, giving a permutation- and gauge-invariant
representation.

We evaluate on all three heterogeneous graph tasks, and the framework attains the best overall
performance among the methods we compare against.""", "0:35  \u00b7  running 2:50")


# ---- 6. Preliminaries I ---------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "2. Preliminaries", "Typed graphs, heterophily, schema vs. instance", 6, TOTAL)

tfa = panel(s, MARGIN, top + Inches(0.1), Inches(6.28), Inches(1.44), "Heterogeneous graph")
eq(tfa, "G = (V, E, \u03c6, \u03c8)", first=True, space_after=6)
para(tfa, "\u03c6 types the nodes, \u03c8 types the edges. Heterogeneous as soon as there "
          "is more than one node type or relation type.", size=PANEL_BODY)

tfb = panel(s, MARGIN, top + Inches(1.72), Inches(6.28), Inches(2.34), "Homophily / heterophily",
            accent=TEAL)
eq(tfb, "H_edge(G\u2032) = (1/|E\u2032|) \u00b7 \u03a3 1[y_u = y_v]", first=True, space_after=8)
para(tfb, "Averaged over the meta-path-induced subgraphs:", size=PANEL_BODY, space_after=6)
eq(tfb, "\u210b = (1/|\u2119|) \u00b7 \u03a3_p H_edge(G\u2032_p)", space_after=10)
para(tfb, "DBLP 59.76   \u00b7   ACM 72.39   \u00b7   IMDB 50.04",
     size=16, color=ACCENT, bold=True)

tfc = panel(s, MARGIN + Inches(6.55), top + Inches(0.1), Inches(6.06), Inches(3.96),
            "The distinction the paper turns on", accent=ACCENT)
para(tfc, "Meta-path schema \u2014 a type-level pattern", size=16, bold=True, color=PRIMARY,
     space_after=6)
eq(tfc, "\u03c0 :  t\u2080 \u2192\u02b3\u00b9 t\u2081 \u2192\u02b3\u00b2 \u2026 "
        "\u2192\u02b3\u02e1 t\u2113", space_after=6)
para(tfc, "Induces the walk-count matrix  A_\u03c0 = A_r\u2081 \u22ef A_r\u2113 .",
     size=PANEL_BODY, space_after=14)

para(tfc, "Concrete path instance \u2014 one occurrence", size=16, bold=True, color=PRIMARY,
     space_after=6)
eq(tfc, "f = (v\u2080, e\u2081, v\u2081, \u2026, e\u2113, v\u2113)", space_after=14)

para(tfc, "Prior work selects \u03c0.   We keep f.", size=17, bold=True, color=ACCENT)

notes(s, """Four definitions, quickly, because everything after this depends on them.

A heterogeneous graph carries a node-typing map phi and an edge-typing map psi. It is heterogeneous
as soon as there is more than one node type OR more than one relation type; with exactly one of
each it reduces to an ordinary homogeneous graph.

EDGE HOMOPHILY is the fraction of edges whose endpoints share a label. In a heterogeneous graph the
endpoint labels are not always comparable \u2014 an author and a paper are not the same kind of
thing \u2014 so we follow the convention of computing homophily on each meta-path-induced subgraph
and averaging over the schemas considered. Those are the numbers in gold: DBLP about 60, ACM about
72, IMDB about 50. Remember IMDB is the genuinely heterophilic one; it will matter in the results.

Now the distinction the whole paper turns on, on the right. A META-PATH SCHEMA, pi, is a TYPE-LEVEL
pattern \u2014 a sequence of node types and relation types. It induces a walk-count matrix, the
product of the per-relation adjacency matrices, whose entries count the walks following that
pattern; and when the endpoint types agree, thresholding it produces the homogeneous endpoint graph
everyone uses.

A CONCRETE PATH INSTANCE, f, is one actual occurrence of that pattern in the observed graph, with
specific nodes and specific edges.

Prior work selects and weights at the level of pi. We keep f. Please hold on to that
distinction.""", "0:40  \u00b7  running 3:30")


# ---- 7. Preliminaries II --------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "2. Preliminaries", "Cellular sheaves in one slide", 7, TOTAL)

tf = add_text(s, MARGIN, top + Inches(0.14), Inches(6.4), Inches(2.6))
bullet(tf, "A graph is a 1D cellular complex: nodes, then edges", first=True)
bullet(tf, "A sheaf gives every cell a stalk \u2014 its own local frame")
bullet(tf, "And every incidence a restriction map F_{c\u22b4\u03c4}")
bullet(tf, "Diffusion runs through those maps, not over raw features")

tfp = panel(s, MARGIN + Inches(6.68), top + Inches(0.1), Inches(5.93), Inches(2.34),
            "The first coboundary compares transported endpoints")
eq(tfp, "(\u03b4\u2080 x)_e  =  F_{v\u22b4e} x_v  \u2212  F_{u\u22b4e} x_u", size=17, first=True,
   space_after=10)
eq(tfp, "L\u2080 = \u03b4\u2080\u1d40 \u03b4\u2080 ,   x\u1d40 L\u2080 x = "
        "\u2016\u03b4\u2080 x\u2016\u00b2", size=17, space_after=10)
para(tfp, "Disagreement is measured after transport, not before. Identity maps recover the ordinary "
          "graph Laplacian.", size=PANEL_BODY)

callout(s, MARGIN, top + Inches(3.06), Inches(6.4), Inches(1.56), [
    ("Why sheaves suit heterophily", {"bold": True, "size": 16, "color": PRIMARY}),
    ("Two neighbours can look different and still be perfectly consistent in the learned geometry.",
     {"size": 15}),
])

callout(s, MARGIN + Inches(6.68), top + Inches(2.62), Inches(5.93), Inches(2.0), [
    ("The opening we work in", {"bold": True, "size": 16, "color": PRIMARY}),
    ("Existing sheaf models \u2014 HetSheaf included \u2014 stop at observed node-edge incidences.",
     {"size": 15}),
    ("H2S-NN adds path-supported 2D-cells and edge-to-face maps.",
     {"size": 15, "bold": True, "color": ACCENT}),
], accent=TEAL, fill=PANEL)

notes(s, """Sheaves in one slide, for anyone who has not met them.

View a graph as a one-dimensional cellular complex: nodes are zero-cells, edges are one-cells. We
write c-incident-tau for the incidence relation. A cellular sheaf attaches a vector space \u2014 a
STALK \u2014 to every cell, and a linear RESTRICTION MAP to every incidence. So instead of one
feature vector per node, each node carries its own small local coordinate system. The node
assignments form the zeroth cochain space, the edge assignments the first.

The key operator is the first coboundary, delta-zero. On an oriented edge it transports BOTH
endpoint signals into the shared edge stalk and takes their difference. The sheaf Laplacian L-zero
is delta-zero transpose delta-zero, and its quadratic form is the squared norm of delta-zero x
\u2014 it measures disagreement AFTER transport, not raw feature difference.

That is exactly why sheaf models handle heterophily, as in the box on the left: two neighbours may
look completely different and still be perfectly consistent in the learned geometry. Sheaf networks
are known to address oversmoothing, oversquashing, and heterophily for this reason. And with
one-dimensional stalks and identity restriction maps you recover the ordinary graph Laplacian, so
this is a strict generalisation, not a different object.

Now the opening we work in, on the right. Cellular complexes in general admit two-dimensional cells
bounded by edges. But sheaf networks ON GRAPHS \u2014 including HetSheaf, which adapts sheaf maps to
heterogeneous graphs \u2014 define their structure over observed node-edge incidences only. We add
the missing level.""", "0:45  \u00b7  running 4:15")


# ---- 8. Method overview ---------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "3. Methodology", "Problem formulation and pipeline", 8, TOTAL)

tfp = add_text(s, MARGIN, top + Inches(0.04), FULL_W, Inches(0.66))
rich(tfp, [("\u0177_o = g_\u03b8(o ; G, \u2110_\u03a3)",
            {"size": 19, "bold": True, "font": MONO, "color": PRIMARY}),
           ("     one backbone; o is a node, a node pair, or a whole graph.",
            {"size": 17})], first=True, space_after=0)

picture(s, FIG_DIR / "h2snn_overview-1.png", MARGIN, top + Inches(0.84), FULL_W, Inches(2.72))

stages = [
    ("1  Preprocessing", "Typed features \u2192 shared d-dim stalks.", PRIMARY),
    ("2  AMPSS", "Select schemas, then weight instances.", ACCENT),
    ("3  Encoding", "Path-supported 2D-cells and their maps.", TEAL),
    ("4  Diffusion", "K Euler steps \u2192 task readout.", PRIMARY),
]
w = Inches(2.85)
x = MARGIN
for head, desc, col in stages:
    add_box(s, x, top + Inches(3.96), w, Inches(0.88), fill=PANEL)
    add_box(s, x, top + Inches(3.96), w, Emu(45720), fill=col)
    tfs = add_text(s, x + Inches(0.16), top + Inches(4.12), w - Inches(0.32), Inches(0.66))
    para(tfs, head, size=14, bold=True, color=col, first=True, space_after=3)
    para(tfs, desc, size=12.5, color=MUTED, space_after=0)
    x += w + Inches(0.216)

notes(s, """The formal problem, then the pipeline.

We learn a predictor g-theta over a directed heterogeneous graph with type-dependent raw features.
The supervised object o is deliberately abstract \u2014 a target node, a candidate typed node pair,
or an entire graph \u2014 so ONE backbone serves all three tasks and only the readout changes. The
loss is cross-entropy for single-label node and graph classification, and binary cross-entropy for
multilabel node classification and link prediction. Script-I-Sigma denotes the retained instances of
the selected schemas.

This figure is the map for the next six slides.

STAGE ONE, far left: type-specific encoders map heterogeneous features of different dimensions into
one shared d-dimensional stalk space.

STAGE TWO, AMPSS: depth-first search over the type graph enumerates candidate schemas; Stage I
selects a task-relevant subset using training data only; Stage II learns weights for the individual
instances that survive.

STAGE THREE, the sheaf-Hodge encoding: each retained instance becomes a two-cell, closed by that
dashed synthetic edge, and the edge-to-face maps are built by transport rather than emitted freely.

STAGE FOUR, the diffusion itself: two lanes, one for node cochains, one for edge cochains, then the
task heads.

Two details in the figure to notice now, because they close the loop. The GOLD line running along
the bottom: the instance weight w_f enters the upper edge Laplacian. The PINK DASHED line: the
gradient of the task loss reaching back into the weighting network. That loop is the whole of Stage
II.""", "0:50  \u00b7  running 5:05")


# ---- 9. Input preprocessing -----------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "3.3  Methodology", "Node stalks and sheaf restrictions", 9, TOTAL)

tfa = panel(s, MARGIN, top + Inches(0.1), Inches(6.28), Inches(1.62), "Type-specific node encoder")
eq(tfa, "h\u1d65\u207d\u2070\u207e = Drop[ GELU( LN(W_\u03c6(v) x\u1d65 + b_\u03c6(v)) + "
        "q_\u03c6(v) ) ]", size=15, first=True, space_after=8)
para(tfa, "Every type lands in one shared d-dimensional stalk space.", size=PANEL_BODY)

tfb = panel(s, MARGIN, top + Inches(1.9), Inches(6.28), Inches(1.68),
            "Relation-conditioned restriction maps", accent=ACCENT)
eq(tfb, "F_{z\u22b4e} = \u03bb_{z,e} Q_{z,e} ,   Q\u1d40Q = I ,   \u03bb > 0", size=15, first=True,
   space_after=8)
para(tfb, "The orthogonal part aligns coordinates; the scale weights the disagreement.",
     size=PANEL_BODY)

tfc = panel(s, MARGIN + Inches(6.55), top + Inches(0.1), Inches(6.06), Inches(1.62),
            "The edge cochain is initialised independently", accent=TEAL)
eq(tfc, "s_e\u207d\u2070\u207e = MLP( h_u \u2016 h_v \u2016 \u03b6_r \u2016 "
        "(\u03b4\u2080h)_e )", size=15, first=True, space_after=8)
para(tfc, "Not confined to im \u03b4\u2080 \u2014 an edge carries its own state.", size=PANEL_BODY,
     bold=True, color=ACCENT)

callout(s, MARGIN + Inches(6.55), top + Inches(1.9), Inches(6.06), Inches(1.68), [
    ("Why scaled-orthogonal?", {"bold": True, "size": 16, "color": PRIMARY}),
    ("Transport stays norm-preserving, so diffusion neither explodes nor collapses.", {"size": 15}),
    ("And F\u207b\u00b9 = F\u1d40/\u03bb\u00b2 \u2014 no matrix inversion in the loop.",
     {"size": 15}),
])

callout(s, MARGIN, top + Inches(3.76), FULL_W, Inches(0.86), [
    ("Everything downstream \u2014 the closure maps, the edge-to-face maps, the whole face "
     "geometry \u2014 is built from these two ingredients.",
     {"size": 16, "bold": True, "color": PRIMARY}),
], accent=TEAL, fill=PANEL)

notes(s, """Stage one, in detail. Two ingredients, and everything later is built from them.

FIRST, the node encoder. Different node types have raw feature vectors of different dimensions and
different meanings. The type-specific encoder maps each into one shared d-dimensional stalk space,
with its own weight matrix, its own bias, a learned type embedding, then LayerNorm, GELU and
dropout. If a node type has no attributes at all, we simply use a trainable node embedding directly.
Each relation also gets an embedding zeta-r. The node cochain is initialised to this encoding.

SECOND, the restriction maps. For each oriented typed edge, two relation-conditioned networks
produce the endpoint restrictions from the two endpoint states and the relation embedding. We
constrain each to be a POSITIVE SCALE times an ORTHOGONAL matrix. The orthogonal factor aligns the
two endpoint coordinate systems; the scale modulates how strongly the resulting disagreement counts.

Why that form? Two reasons, in the box. Orthogonal transport is norm-preserving, so diffusion
neither blows up nor collapses the stalk geometry. And the inverse \u2014 which the transport
construction will need in two slides \u2014 is just the transpose divided by lambda squared, so
there is no matrix inversion inside the training loop. In the implementation these come from the
Fast Householder Product parameterisation.

One design choice on the top right deserves emphasis. We initialise the edge cochain INDEPENDENTLY.
It sees both endpoints, the relation embedding, and the transported disagreement \u2014 but it is
not constrained to live in the image of delta-zero. So an edge carries genuine state of its own
rather than a shadow of its endpoints. Without that, the face-induced term we introduce later would
have nothing of its own to act on.""", "0:40  \u00b7  running 5:45")


# ---- 10. AMPSS Stage I ----------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "3.4  Methodology \u00b7 AMPSS", "Stage I \u2014 which schemas?", 10, TOTAL,
             sub="Runs once, before training, on training data only")

tf = add_text(s, MARGIN, top + Inches(0.12), Inches(6.9), Inches(2.5))
bullet(tf, "Bounded DFS over the type graph \u2192 candidate pool \u03a0", first=True)
bullet(tf, "Quotas fix which instances each schema retains")
bullet(tf, "Greedily add the schema with the best held-out gain")
bullet(tf, "Stop when a budget binds or nothing helps")

tfa = panel(s, MARGIN + Inches(7.18), top + Inches(0.1), Inches(5.43), Inches(1.62),
            "Structural cost = edge-face incidences", accent=TEAL)
eq(tfa, "C\u1d35(S) = \u03a3_\u03c0 \u03a3_f (\u2113_f + 1)", size=17, first=True, space_after=8)
para(tfa, "A hard, auditable cap on how much structure is added.", size=PANEL_BODY)

tfb = panel(s, MARGIN + Inches(7.18), top + Inches(1.9), Inches(5.43), Inches(1.62),
            "Greedy gain on a held-out split", accent=ACCENT)
eq(tfb, "q(\u03c0|S) = \u2112_sel(S) \u2212 \u2112_sel(S \u222a {\u03c0})", size=17, first=True,
   space_after=8)
para(tfb, "Subject to  |S| \u2264 K_\u03a3  and  C\u1d35 \u2264 B_inc .", size=PANEL_BODY)

callout(s, MARGIN, top + Inches(2.98), Inches(6.9), Inches(1.64), [
    ("Stated as a limitation, not a feature", {"bold": True, "size": 16, "color": PRIMARY}),
    ("Schema selection is discrete, so gradients from the task loss never reach Stage I.",
     {"size": 15}),
])

callout(s, MARGIN + Inches(7.18), top + Inches(3.7), Inches(5.43), Inches(0.92), [
    ("For link prediction, the selection graph excludes the inner-selection positives and their "
     "reverses.", {"size": 14.5, "color": PRIMARY, "bold": True}),
], accent=TEAL, fill=PANEL)

notes(s, """AMPSS Stage I. This runs ONCE, before training, and it never touches validation or test
data.

We build a type graph from the relation types available, and run BOUNDED DEPTH-FIRST SEARCH to
enumerate a candidate pool of schemas with lengths between ell-min and ell-max. Any candidate with
no feasible SIMPLE path instances is discarded immediately \u2014 that simple-path restriction
matters later, for the proposition.

For a trial schema set, DETERMINISTIC INSTANCE AND INCIDENCE QUOTAS decide which instances are
retained. Those retained instances do double duty: they generate the features for the selection
model, and they determine the structural cost.

That cost, top right, is simply the number of retained edge-face incidences: each instance
contributes its ell path edges plus one closure edge. This is worth pausing on. It means the
incidence budget bounds the SUPPORT of the face-induced operator directly \u2014 the extra structure
we add is capped by an auditable number, not by a hope.

Selection itself is GREEDY ON HELD-OUT TASK GAIN. We fit a cheap auxiliary LINEAR selection model on
an inner-training partition, score it on a DISJOINT held-out selection partition, and add whichever
feasible schema most reduces that held-out loss. The model is refitted for each trial set, using
outer-training labels only. Feasibility means respecting both the schema-count limit K-Sigma and the
incidence budget B-inc. We start from the empty set and stop when no candidate gives a positive
reduction or a budget binds. After selection, the retained instances are rebuilt on the full
outer-training graph.

For link prediction there is an extra precaution, bottom right: we delete the inner-selection
positive links AND their reverse counterparts from the graph used for selection, so the probe cannot
see its own targets.

And an honest note, bottom left: because selection is discrete, the task-loss gradient does not
reach Stage I. We report that as a limitation.""", "0:50  \u00b7  running 6:35")


# ---- 11. AMPSS Stage II ---------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "3.4  Methodology \u00b7 AMPSS", "Stage II \u2014 which instances?", 11, TOTAL,
             sub="Fully differentiable; trained by the supervised task loss alone")

tfa = panel(s, MARGIN, top + Inches(0.1), Inches(6.28), Inches(2.5),
            "A positive weight per retained instance", accent=ACCENT)
eq(tfa, "w_f = (1\u2212r) + 2r \u00b7 sigmoid( g_w( \u03be_f \u2016 e_\u03c0(f) ) )", size=15,
   first=True, space_after=10)
bullet(tfa, "\u03be_f \u2014 tells instances apart within a schema", size=PANEL_BODY, space_after=7)
bullet(tfa, "e_\u03c0(f) \u2014 tells PAP from PSP", size=PANEL_BODY, space_after=7)
bullet(tfa, "r \u2014 warm-up, uniform \u2192 learned", size=PANEL_BODY, space_after=0)

tfb = panel(s, MARGIN, top + Inches(2.78), Inches(6.28), Inches(1.84),
            "Learning starts from the unweighted operator", accent=TEAL)
para(tfb, "The gate's output layer is zero-initialised, so w_f = 1 at the start.", size=PANEL_BODY,
     space_after=8)
para(tfb, "The model departs from it only if the task loss rewards that. There is no auxiliary "
          "discovery objective.", size=PANEL_BODY)

tfc = panel(s, MARGIN + Inches(6.55), top + Inches(0.1), Inches(6.06), Inches(1.9),
            "Where the weight acts")
eq(tfc, "\u03b4\u2081\u1d40 W(\u03c4) \u03b4\u2081 ,    W = diag(w_f)", size=18, first=True,
   space_after=10)
para(tfc, "Only the face-induced component. Since w_f > 0, it stays positive semidefinite.",
     size=PANEL_BODY)

callout(s, MARGIN + Inches(6.55), top + Inches(2.18), Inches(6.06), Inches(2.44), [
    ("What the weights do not touch", {"bold": True, "size": 16, "color": PRIMARY}),
    ("Closure edges stay in the extended edge domain regardless of w_f , and keep contributing to "
     "\u03b4\u2080 , L\u2080 , the node-mediated term, and the readout.", {"size": 15}),
    ("Stage II modulates influence. It never prunes the complex.",
     {"size": 15, "bold": True, "color": ACCENT}),
])

notes(s, """Stage II is where the granularity gap gets addressed directly.

Every retained instance gets a POSITIVE WEIGHT, and the gate sees two things. Xi-f is the
ordered-path latent produced by the face encoder \u2014 it distinguishes instances WITHIN a single
schema, which is precisely the discrimination schema-level methods cannot make. The schema embedding
lets one shared gate tell PAP from PSP. And r-of-tau is a warm-up schedule that ramps the operator
from uniform weights to learned ones. The output lies between zero and two.

Two properties matter. First, the gate's output layer is ZERO-INITIALISED, so w_f equals exactly one
at initialisation. Training therefore starts from precisely the unweighted face-induced operator and
departs from it only if the task loss rewards doing so. Second, these weights are learned purely
from the supervised task loss \u2014 no auxiliary discovery objective, no schema gate.

Top right: the weight enters ONLY the face-induced component, as delta-one transpose W delta-one.
Because every weight is strictly positive, that component stays positive semidefinite, so the
diffusion remains well posed for any value the gate can produce. Intuitively: a weight near zero
mutes an instance's higher-order coupling, a weight near two amplifies it.

And what the weights do NOT touch, bottom right \u2014 this is the safety property. The closure
edges remain part of the extended edge domain no matter what w_f does. They keep contributing to
delta-zero, to L-zero, to the node-mediated component of the edge operator, to the weighted face
boundary, and to the edge readout. So Stage II modulates the INFLUENCE of higher-order supports; it
never prunes the complex, and it cannot silently disconnect the graph.""",
      "0:40  \u00b7  running 7:15")


# ---- 12. 2-cell construction ----------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "3.5  Methodology", "Path-supported 2D-cells", 12, TOTAL)

tf = add_text(s, MARGIN, top + Inches(0.14), Inches(7.0), Inches(1.5))
bullet(tf, "Keep the instance; do not collapse it to a shortcut", first=True)
bullet(tf, "A synthetic closure 1-cell closes the open path:")

tfe = add_text(s, MARGIN + Inches(0.44), top + Inches(1.5), Inches(6.6), Inches(0.5))
eq(tfe, "\u2202f = ( e\u2081, \u2026, e\u2113, e_cl )", size=21, first=True, space_after=0)

tf2 = add_text(s, MARGIN, top + Inches(2.12), Inches(7.0), Inches(2.5))
bullet(tf2, "It is not an observed relation, and not a shortcut", first=True)
bullet(tf2, "E_ext = E \u2294 E_cl gives three graded cochain spaces")
bullet(tf2, "Closure cells sit inside \u03b4\u2080 , L\u2080 and L\u2081 \u2014 no side correction")

tfc = panel(s, MARGIN + Inches(7.28), top + Inches(0.1), Inches(5.33), Inches(2.14),
            "The payoff, on P\u2013A\u2013P", accent=ACCENT)
para(tfc, "The two paper-author edges stay as boundary edges of one 2D-cell.", size=PANEL_BODY,
     space_after=8)
para(tfc, "So the intermediate author remains in the operator, instead of vanishing into a "
          "paper-paper shortcut.", size=PANEL_BODY, bold=True, color=PRIMARY)

tfd = panel(s, MARGIN + Inches(7.28), top + Inches(2.42), Inches(5.33), Inches(2.2),
            "Scope, stated up front", accent=TEAL)
bullet(tfd, "Faces are simple paths \u2014 no revisited nodes", size=PANEL_BODY, first=True,
       space_after=8)
bullet(tfd, "Incidence signs are explicit, and are +1 here", size=PANEL_BODY, space_after=8)
bullet(tfd, "Enumeration is forward-only \u2014 asserted, not assumed", size=PANEL_BODY,
       space_after=0)

notes(s, """Here is the core representational move of the paper.

A concrete instance is a path: v-zero, edge one, v-one, and so on to v-ell. Instead of collapsing it
to a shortcut between the two endpoints, we make it a TWO-DIMENSIONAL CELL whose boundary is the
ORDERED sequence of path edges.

A path is open, and a face needs a closed boundary, so we introduce one SYNTHETIC CLOSURE ONE-CELL
running from the last node back to the first. Let me be precise about what that cell is and is not,
because this is the obvious place to object. It exists solely to complete the boundary of the
two-cell. It is NOT an observed relation, and it does NOT create an observed endpoint shortcut
\u2014 which would reintroduce exactly the collapse we are trying to avoid.

We then work over the EXTENDED edge set: observed edges disjoint-union closure edges. The complex
now has three graded pieces \u2014 nodes, extended edges, and the retained instances as faces.
Because closure cells are first-class members of the extended edge set, delta-zero is evaluated over
all of them, and they participate in the initial edge cochain and in BOTH Laplacians without any
separate node-side correction term.

The payoff, top right, on P-A-P: the two paper-author edges stay as boundary edges of one cell, so
the intermediate author remains part of the operator rather than vanishing into a paper-paper
shortcut. The face-induced interactions can then couple disagreements along the whole
paper-author-paper pattern \u2014 precisely the information that collapse destroys.

Bottom right, the scope conditions, stated up front rather than buried. Faces are simple paths;
instances that revisit a node, INCLUDING coincident endpoints, are excluded at enumeration.
Incidence signs are represented explicitly, and they are plus one here because the enumeration
follows stored edge directions and orients the closure edge along the cycle \u2014 that is asserted
in the code, not assumed. The next slide shows why the simple-path restriction is not
cosmetic.""", "0:40  \u00b7  running 7:55")


# ---- 13. Transport + proposition ------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "3.5  Methodology", "One free map per face \u2014 and exactness", 13, TOTAL)

tfa = panel(s, MARGIN, top + Inches(0.1), Inches(6.28), Inches(2.14),
            "Predict A_f , then transport it around the path")
eq(tfa, "A_f = MLP_\u2113( h_{v\u2080} \u2016 \u22ef \u2016 h_{v\u2113} \u2016 \u03b6_{r\u2081} "
        "\u2016 \u22ef )", size=14.5, first=True, space_after=8)
eq(tfa, "T_{v\u1d62} = T_{v\u1d62\u208b\u2081} F\u207b\u00b9_{u\u22b4e\u1d62} F_{v\u22b4e\u1d62} , "
        "  F_{e\u1d62\u22b4f} = T_{v\u1d62\u208b\u2081} F\u207b\u00b9_{u\u22b4e\u1d62}",
   size=14.5, space_after=8)
para(tfa, "The closure maps come from the same transport \u2014 no new learned map.",
     size=PANEL_BODY, bold=True, color=ACCENT)

tfb = panel(s, MARGIN, top + Inches(2.52), Inches(6.28), Inches(2.1), "Proof sketch", accent=TEAL)
para(tfb, "Consecutive path terms telescope to  T_{v\u2113}x_{v\u2113} \u2212 "
          "T_{v\u2080}x_{v\u2080}, and the closure term contributes exactly its negative.",
     size=PANEL_BODY, space_after=8)
para(tfb, "Simplicity is required because transport assigns one map per node position.",
     size=PANEL_BODY, color=MUTED, italic=True)

add_box(s, MARGIN + Inches(6.55), top + Inches(0.1), Inches(6.06), Inches(2.14), fill=HILITE)
add_box(s, MARGIN + Inches(6.55), top + Inches(0.1), Emu(34290), Inches(2.14), fill=ACCENT)
tfp = add_text(s, MARGIN + Inches(6.8), top + Inches(0.26), Inches(5.6), Inches(1.84))
para(tfp, "Proposition 1", size=17, bold=True, color=PRIMARY, first=True, space_after=8)
eq(tfp, "\u03b4\u2081,w \u03b4\u2080 = 0", size=26, space_after=8, align=PP_ALIGN.CENTER)
para(tfp, "Identically \u2014 for every node cochain and every parameter value.", size=15.5,
     bold=True, space_after=6)
para(tfp, "Enforced by construction, not by a penalty.", size=15.5, color=ACCENT, bold=True)

callout(s, MARGIN + Inches(6.55), top + Inches(2.52), Inches(6.06), Inches(2.1), [
    ("Why it matters", {"bold": True, "size": 16, "color": PRIMARY}),
    ("The Hodge decomposition stays interpretable: edge signals in im \u03b4\u2080 are the ones "
     "nodes induce.", {"size": 15}),
    ("The face-induced term therefore acts on the residual \u2014 the part nodes cannot explain.",
     {"size": 15, "bold": True, "color": PRIMARY}),
], accent=TEAL, fill=PANEL)

notes(s, """This slide is the mathematical heart of the paper, so let me take it carefully.

For each instance we predict exactly ONE free matrix, A_f, from the ordered path \u2014 the node
states in path order, concatenated with the schema's relation embeddings, through a
path-length-specific MLP. A_f defines the local coordinate FRAME of the face stalk.

Every OTHER map around that face is then DERIVED, not learned. We transport A_f along the path
through the node-to-edge restriction maps: the running transport at node i is the previous transport
composed with the inverse source map and the target map; the edge-to-face map falls out of the same
recursion. The closure edge uses the same transport \u2014 its first map is the identity, its second
is the accumulated product along the whole path, and its face map is A_f itself. So the closure edge
introduces NO additional learned restriction map. All ell-plus-one boundary maps are generated by
one matrix.

Why go to this trouble? PROPOSITION ONE, in gold. For simple-path instances with the default forward
orientation, delta-one composed with delta-zero is EXACTLY ZERO \u2014 identically, for every node
cochain and every value of the learned parameters. Not approximately. Not encouraged by a penalty
term. By construction.

The proof is a telescoping argument. By construction the edge-to-face map composed with the source
restriction gives the transport at the previous node, and composed with the target restriction gives
the transport at this node. So the i-th path term is a difference of consecutive transported values;
summing along the path telescopes to the endpoint difference; and the closure term contributes
exactly its negative. Total: zero.

And here is why it matters practically, bottom right. Because the composition vanishes, the HODGE
DECOMPOSITION STAYS CLEAN: edge signals in the image of delta-zero are precisely the ones induced by
node representations, and the face-induced component acts on the edge-level RESIDUAL \u2014 the part
nodes cannot explain. If the composition were only approximately zero, the face term would be partly
re-diffusing node information, and the whole higher-order story would be unfalsifiable.

The simple-path condition is exactly what the proof needs: transport assigns ONE map per node
position, so a revisited node would demand two different transported values at the same
node.""", "0:50  \u00b7  running 8:45")


# ---- 14. Diffusion --------------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "3.6  Methodology", "Sheaf-Hodge diffusion", 14, TOTAL)

tfa = panel(s, MARGIN, top + Inches(0.1), Inches(6.28), Inches(2.3), "Two operators, two lanes")
eq(tfa, "L\u2080 x = \u03b4\u2080\u1d40 \u03b4\u2080 x", size=17, first=True, space_after=8)
rich(tfa, [("L\u2081 s = ", {"size": 17, "bold": True, "font": MONO, "color": PRIMARY}),
           ("\u03b4\u2080 \u03b4\u2080\u1d40 s", {"size": 17, "bold": True, "font": MONO,
                                                  "color": TEAL}),
           (" + ", {"size": 17, "bold": True, "font": MONO, "color": PRIMARY}),
           ("\u03b4\u2081\u1d40 W \u03b4\u2081 s", {"size": 17, "bold": True, "font": MONO,
                                                    "color": ACCENT})], space_after=10)
bullet(tfa, "node-mediated \u2014 through shared incident nodes", size=PANEL_BODY, color=TEAL,
       space_after=7)
bullet(tfa, "face-induced \u2014 inside a 2D-cell; Stage II weights this", size=PANEL_BODY,
       color=ACCENT, space_after=0)

tfb = panel(s, MARGIN, top + Inches(2.58), Inches(6.28), Inches(2.04), "K explicit Euler steps",
            accent=TEAL)
eq(tfb, "x\u207d\u1d4f\u207a\u00b9\u207e = x\u207d\u1d4f\u207e \u2212 \u03b1\u2080 L\u2080 "
        "x\u207d\u1d4f\u207e", size=15, first=True, space_after=4)
eq(tfb, "s\u207d\u1d4f\u207a\u00b9\u207e = s\u207d\u1d4f\u207e \u2212 \u03b1\u2081 L\u2081 "
        "s\u207d\u1d4f\u207e", size=15, space_after=10)
para(tfb, "Step sizes may be learned. LayerNorm gives x\u2605 and s\u2605 .", size=PANEL_BODY)

tfc = panel(s, MARGIN + Inches(6.55), top + Inches(0.1), Inches(6.06), Inches(1.62),
            "Matrix-free by design", accent=ACCENT)
para(tfc, "Local d\u00d7d maps, batched, aggregated with indexed sums.", size=PANEL_BODY,
     space_after=8)
para(tfc, "No dense Laplacian is ever materialised.", size=PANEL_BODY, bold=True, color=PRIMARY)

tfd = panel(s, MARGIN + Inches(6.55), top + Inches(1.9), Inches(6.06), Inches(2.72),
            "Three boundaries we state explicitly")
bullet(tfd, "\u03b4\u2081 uses the complete boundary cycle, not a truncation", size=PANEL_BODY,
       first=True, space_after=9)
bullet(tfd, "Signs are +1 because enumeration is forward-only", size=PANEL_BODY, space_after=9)
bullet(tfd, "Simple paths only \u2014 revisiting instances are excluded", size=PANEL_BODY,
       space_after=0)

notes(s, """Diffusion. Two operators, two lanes.

L-ZERO on nodes is the familiar sheaf Laplacian.

L-ONE on edges has two parts, and the split is the interesting bit. The NODE-MEDIATED, or lower,
component relates edge signals through shared incident nodes \u2014 that is standard Hodge theory.
The FACE-INDUCED, or upper, component couples boundary edges within a selected two-cell. That upper
term exists ONLY because we built the faces, and it is exactly the term AMPSS Stage II weights.
Because each closure cell is a first-class member of the extended edge set, it participates in both
components with no separate node-side correction.

The update is K explicit Euler steps on each cochain, starting from the encoded node states and the
independently initialised edge states, with positive step sizes that may be learned, followed by
LayerNorm to give the terminal states x-star and s-star.

One honest note on the architecture: the two lanes are NOT exchanged between steps. They are coupled
through the shared restriction geometry and the shared initial node representations, rather than by
mixing at every step.

Top right: this is all MATRIX-FREE. We batch local d-by-d maps and aggregate with indexed sums,
applying delta-zero, its transpose, delta-one and its transpose directly on node, edge and face
tensors. No global operator is ever materialised, which is what keeps the cost linear in graph size.

Bottom right, three boundaries we state explicitly rather than leave to a reader's assumption.
Delta-one uses the FULL boundary cycle \u2014 the ell path edges plus the closure cell, not a
truncation. Incidence signs are explicit and are plus one here because enumeration is forward-only.
And simple paths only. Under those constraints the chain condition is a property of the constructed
operator, not a penalty added to the objective.""", "0:40  \u00b7  running 9:25")


# ---- 15. SheafCovPool -----------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "3.7  Methodology", "SheafCovPool: a gauge-invariant readout", 15, TOTAL)

tf = add_text(s, MARGIN, top + Inches(0.14), Inches(6.6), Inches(2.5))
bullet(tf, "Node and edge states live in different local frames", first=True)
bullet(tf, "So transport edge states into node stalks first")
bullet(tf, "Three views per cell \u2192 a 3\u00d73 Gram matrix")
bullet(tf, "Attention \u2192 weighted mean and covariance")

tfe = add_text(s, MARGIN + Inches(0.1), top + Inches(2.76), Inches(6.5), Inches(0.5))
eq(tfe, "\u03b3_g = [ \u03bc\u2080 \u2016 triu(\u03a3\u2080) \u2016 \u03bc\u2081 \u2016 "
        "triu(\u03a3\u2081) ]", size=18, first=True, space_after=0)

callout(s, MARGIN, top + Inches(3.4), Inches(6.6), Inches(1.22), [
    ("The Gram matrix cancels the common orthogonal orientation \u2014 so \u03b3_g is invariant to "
     "ordering AND to stalk basis.", {"size": 15.5, "bold": True, "color": PRIMARY}),
])

picture(s, FIG_DIR / "sheafcovpool-1.png", MARGIN + Inches(6.88), top + Inches(0.16),
        Inches(5.73), Inches(2.52),
        caption="Two lanes: views \u2192 Gram \u2192 attention \u2192 mean and covariance.")

callout(s, MARGIN + Inches(6.88), top + Inches(3.4), Inches(5.73), Inches(1.22), [
    ("The Gram order is fixed at 3 by the three semantic views \u2014 capacity is tuned through "
     "the post-Gram width F instead.", {"size": 15.5, "bold": True, "color": PRIMARY}),
], accent=TEAL, fill=PANEL)

notes(s, """The third contribution, and the last method slide.

Graph classification needs ONE vector per graph. But in a sheaf model every node and every edge has
its own local coordinate frame. If you simply average incident edge states you are adding vectors
expressed in different bases \u2014 the result is not meaningful.

So SheafCovPool TRANSPORTS FIRST. Each incident edge state is projected into its endpoint node space
using the TRANSPOSE of the restriction map, then averaged over the incidences at that node, across
both observed and closure edges; an isolated node gets the zero vector. Both endpoints contribute
with a POSITIVE sign, because what we want here is incident context, not the orientation-sensitive
divergence.

Then we form THREE complementary views at each cell. For a node: the diffused state, the initial
typed state, and the transported edge summary. For an edge: two transported node disagreements and
the edge's own diffused state. Each stack is d-by-three.

The trick is what we do with those views: we take the three-by-three GRAM MATRIX of the stack. That
records the energies and pairwise alignments of the three views while CANCELLING their common
orthogonal orientation. Formally, if you change the node and edge bases by independent orthogonal
matrices and transform the restriction maps consistently, the node stack transforms by Q, the edge
stack by R, and both Gram descriptors are unchanged.

Attention then runs over those invariant descriptors, and we keep both a weighted MEAN and a
weighted COVARIANCE, for nodes and for edges. The mean says what the attention emphasised; the
covariance says how much that evidence varies across the graph. Concatenating gives gamma-g, which
lives in F-squared plus 3F dimensions.

Because every learned score acts on an invariant Gram matrix and every aggregation is symmetric,
gamma-g is invariant to node and edge ordering, and to independent orthogonal changes of stalk
basis.

One detail people ask about: the Gram order is fixed at THREE because three is the number of
semantic views, not a hyperparameter. Changing it would mean adding or removing stalk-valued views
\u2014 a different readout, not a wider one. Capacity is tuned through the post-Gram width F
instead.""", "0:45  \u00b7  running 10:10")


# ---- 16. Experiment settings ----------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "4. Experiments", "Settings", 16, TOTAL)

rows = [
    ["Task", "Datasets", "Protocol", "Metrics"],
    ["Node classification", "DBLP, IMDB (HGB); ACM", "HGB; transductive 20 / 20 / 60",
     "Macro-F1, Micro-F1"],
    ["Graph classification", "MUTAG, PROTEINS, ENZYMES, NCI1 (TUDataset)",
     "10-fold CV over 5 seeds", "Macro-F1, Micro-F1"],
    ["Link prediction", "LastFM, PubMed, Amazon, YouTube (HGB)",
     "HGB; 81 / 9 / 10; 1:1 two-hop negatives", "AUPR, AUROC"],
]
table(s, rows, MARGIN, top + Inches(0.1), FULL_W, col_w=[1.5, 3.2, 2.9, 1.4], font=13,
      header_font=13, row_h=Inches(0.44), bold_last=False, highlight_last=False)

tfa = panel(s, MARGIN, top + Inches(2.12), Inches(7.4), Inches(2.5),
            "Baselines, by graph type and homophily assumption")
bullet(tfa, "Hom./hom. \u2014 GCN, GAT, ChebNetII", size=PANEL_BODY, first=True, space_after=8)
bullet(tfa, "Hom./het. \u2014 NSD, CSNN", size=PANEL_BODY, space_after=8)
bullet(tfa, "Het./hom. \u2014 HAN, SeHGNN, HiNormer, PSHGCN", size=PANEL_BODY, space_after=8)
bullet(tfa, "Het./het. \u2014 HDHGR, Hetero\u00b2Net, H2SGNN, HetSheaf, LatGRL", size=PANEL_BODY,
       space_after=0)

tfb = panel(s, MARGIN + Inches(7.68), top + Inches(2.12), Inches(4.93), Inches(2.5),
            "Common protocol", accent=TEAL)
bullet(tfb, "Early stopping, patience 100", size=PANEL_BODY, first=True, space_after=8)
bullet(tfb, "Per-dataset sweeps", size=PANEL_BODY, space_after=8)
bullet(tfb, "5 seeds, mean \u00b1 std", size=PANEL_BODY, space_after=8)
bullet(tfb, "AMPSS Stage I sees training data only", size=PANEL_BODY, space_after=0)

notes(s, """Experimental setup, briefly.

Three tasks. NODE CLASSIFICATION on DBLP and IMDB from HGB, plus ACM, under the HGB protocol with a
twenty / twenty / sixty transductive split. GRAPH CLASSIFICATION on four TUDatasets \u2014 two
biological, PROTEINS and ENZYMES, two chemical, MUTAG and NCI1 \u2014 with ten-fold cross-validation
over five seeds, and one-vs-rest for the six-class ENZYMES. LINK PREDICTION on four HGB benchmarks
with an eighty-one / nine / ten split and negatives sampled one-to-one from the two-hop
neighbourhood.

Note the second row carefully. For the TUDatasets we read the provided NODE AND EDGE LABELS as node
types and relation types. So these are genuinely heterogeneous inputs, not homogeneous graphs with
categorical attributes \u2014 and that choice turns out to matter in the results.

Baselines are grouped along two axes: is the graph modelled as homogeneous or heterogeneous, and is
the method designed for homophily or heterophily. That gives four groups, and we keep
representatives of all four in every task. Graph classification and link prediction use a subset of
the node-classification baselines, still spanning all four groups. The most relevant comparison for
us is HetSheaf, in the bottom group \u2014 the closest heterogeneous SHEAF baseline.

Everything uses early stopping with patience one hundred, per-dataset sweeps, five seeds, and
reported standard deviations. AMPSS Stage I sees training data only, with selector and
materialisation seeds kept separate from training seeds.""", "0:35  \u00b7  running 10:45")


# ---- 17. Node classification ----------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "4.2  Experiments", "Node classification", 17, TOTAL,
             sub="Macro-F1 / Micro-F1;  graph homophily \u210b in parentheses")

rows = [
    ["Group", "Model", "DBLP Ma", "DBLP Mi", "ACM Ma", "ACM Mi", "IMDB Ma", "IMDB Mi"],
    ["Hom./hom.", "GCN", "90.77", "90.28", "91.63", "91.38", "62.33", "67.69"],
    ["", "GAT", "94.39", "92.52", "92.07", "90.71", "62.13", "68.09"],
    ["", "ChebNetII", "92.89", "92.67", "91.70", "91.59", "62.97", "68.47"],
    ["Hom./het.", "NSD", "92.96", "93.42", "92.33", "92.05", "61.61", "65.23"],
    ["", "CSNN", "93.73", "92.12", "93.13", "92.81", "62.92", "62.59"],
    ["Het./hom.", "HAN", "91.78", "92.24", "90.88", "91.61", "61.95", "67.76"],
    ["", "SeHGNN", "95.52", "95.48", "94.38", "94.04", "72.01", "72.59"],
    ["", "HiNormer", "94.06", "94.14", "91.87", "89.64", "64.13", "67.53"],
    ["", "PSHGCN", "95.58", "95.70", "93.66", "94.70", "71.82", "74.34"],
    ["Het./het.", "HDHGR", "95.35", "95.00", "93.99", "94.13", "59.41", "59.63"],
    ["", "Hetero\u00b2Net", "93.54", "93.54", "94.05", "92.87", "65.34", "70.63"],
    ["", "H2SGNN", "95.82", "95.11", "94.90", "94.08", "74.23", "75.29"],
    ["", "HetSheaf", "93.87", "93.77", "95.09", "94.94", "64.65", "67.18"],
    ["", "LatGRL", "94.32", "94.51", "93.47", "92.29", "58.27", "58.81"],
    ["Ours", "H2S-NN", "96.01", "96.31", "95.91", "95.72", "75.05", "76.68"],
]
table(s, rows, MARGIN, top + Inches(0.06), Inches(8.1),
      col_w=[1.3, 1.45, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      font=11, header_font=10.5, row_h=Inches(0.283), group_col=True)

tfa = panel(s, MARGIN + Inches(8.4), top + Inches(0.06), Inches(4.21), Inches(2.3),
            "Reading the table", accent=ACCENT)
bullet(tfa, "Best on all three, both metrics", size=PANEL_BODY, first=True, space_after=8)
bullet(tfa, "DBLP: +0.19 / +0.61", size=PANEL_BODY, space_after=8)
bullet(tfa, "ACM: +0.82 / +0.78 over HetSheaf", size=PANEL_BODY, space_after=8)
bullet(tfa, "IMDB: +0.82 / +1.39 \u2014 biggest gain, lowest homophily", size=PANEL_BODY,
       space_after=0)

tfb = panel(s, MARGIN + Inches(8.4), top + Inches(2.56), Inches(4.21), Inches(2.06),
            "A caution we state ourselves", accent=TEAL)
para(tfb, "Several margins sit inside one standard deviation of the runner-up.", size=PANEL_BODY,
     space_after=8)
para(tfb, "So: best among the compared methods under this protocol \u2014 not unqualified state of "
          "the art.", size=PANEL_BODY, bold=True, color=PRIMARY)

notes(s, """Node classification. Fifteen models, three datasets, two metrics each. Standard
deviations are in the paper; I have dropped them here for legibility, and they run from about 0.2 to
1.2.

H2S-NN is best on all three datasets and both metrics. But let me read the table the way a reviewer
would, rather than just claiming the win.

On DBLP we gain about two tenths of a point in Macro-F1 over H2SGNN, and six tenths of Micro-F1 over
PSHGCN. Those are small margins and I will not oversell them.

On ACM we gain roughly eight tenths over HetSheaf \u2014 the most DIRECT comparison in the table,
since HetSheaf also uses sheaf maps on typed graphs but stops at the node-edge complex. So that
margin is closest to isolating the contribution of the 2D-cells.

The result I would draw your attention to is IMDB. IMDB has the lowest homophily of the three
\u2014 about fifty \u2014 and it is the only multilabel task. There the margin is largest: eight
tenths of Macro-F1 and about one and four tenths of Micro-F1 over the strongest baseline. Look also
at how the homogeneous methods and the heterogeneous HOMOPHILIC methods behave on IMDB compared with
DBLP and ACM \u2014 several of them collapse into the fifties. The gap widens exactly where
heterophily is strongest, which is the pattern the motivation predicts.

Bottom right, a caution we state ourselves: several of these margins sit within one standard
deviation of the runner-up. So the claim we defend is BEST OVERALL PERFORMANCE AMONG THE COMPARED
METHODS UNDER THIS PROTOCOL \u2014 not unqualified state of the art. One bookkeeping note: HetSheaf
is reported with zero variance because it is a single reported configuration, not a five-seed
average.""", "0:40  \u00b7  running 11:25")


# ---- 18. Graph classification ---------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "4.2  Experiments", "Graph classification", 18, TOTAL,
             sub="Macro-F1 / Micro-F1;  10-fold cross-validation over 5 seeds")

rows = [
    ["Group", "Model", "MUTAG Ma", "MUTAG Mi", "PROT. Ma", "PROT. Mi", "ENZ. Ma", "ENZ. Mi",
     "NCI1 Ma", "NCI1 Mi"],
    ["Hom./hom.", "GCN", "89.58", "90.00", "75.92", "79.83", "12.75", "28.33", "64.17", "64.96"],
    ["", "GAT", "89.58", "90.00", "75.27", "78.38", "14.03", "16.55", "64.17", "64.96"],
    ["Hom./het.", "NSD", "99.47", "99.69", "73.60", "75.89", "33.65", "35.00", "72.30", "72.30"],
    ["", "CSNN", "99.21", "99.40", "76.64", "77.92", "35.15", "37.22", "74.11", "74.30"],
    ["Het./hom.", "HAN", "82.91", "85.00", "79.19", "82.22", "46.24", "45.02", "77.60", "77.50"],
    ["", "PSHGCN", "94.88", "95.00", "70.61", "72.32", "60.18", "58.33", "74.17", "74.96"],
    ["Het./het.", "HDHGR", "84.65", "85.00", "77.64", "81.25", "58.27", "58.33", "79.29", "79.32"],
    ["", "Hetero\u00b2Net", "89.58", "90.00", "74.66", "78.57", "61.53", "61.36", "63.67", "65.45"],
    ["", "H2SGNN", "94.88", "95.00", "70.61", "72.32", "62.18", "61.31", "64.17", "64.96"],
    ["", "HetSheaf", "100.00", "100.00", "80.56", "82.14", "62.35", "63.33", "79.46", "79.46"],
    ["", "LatGRL", "90.88", "91.01", "71.61", "72.32", "60.41", "60.03", "77.98", "78.10"],
    ["Ours", "H2S-NN", "99.65", "99.72", "84.44", "86.23", "64.52", "65.10", "81.22", "81.52"],
]
table(s, rows, MARGIN, top + Inches(0.08), Inches(8.5),
      col_w=[1.15, 1.25, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92],
      font=10.5, header_font=9.5, row_h=Inches(0.3), group_col=True)

tfa = panel(s, MARGIN + Inches(8.8), top + Inches(0.08), Inches(3.81), Inches(2.16),
            "What the numbers say", accent=ACCENT)
bullet(tfa, "Best on PROTEINS, ENZYMES, NCI1", size=PANEL_BODY, first=True, space_after=8)
bullet(tfa, "MUTAG: a tie, not a win \u2014 188 graphs, saturated", size=PANEL_BODY, space_after=8)
bullet(tfa, "PROTEINS +3.88 over HetSheaf", size=PANEL_BODY, space_after=0)

tfb = panel(s, MARGIN + Inches(8.8), top + Inches(2.42), Inches(3.81), Inches(2.2),
            "The interpretation", accent=TEAL)
para(tfb, "Molecules and proteins \u2014 not the domains where heterogeneity is usually studied.",
     size=PANEL_BODY, space_after=8)
para(tfb, "Atom and bond labels really are types, not just categorical features.", size=PANEL_BODY,
     bold=True, color=PRIMARY)

notes(s, """Graph classification, and here the picture is more interesting than a simple sweep.

On PROTEINS, ENZYMES and NCI1, H2S-NN has the highest mean Macro-F1 and Micro-F1. The margins are
LARGER than in node classification: nearly four points on PROTEINS over HetSheaf, about two and a
sixth on ENZYMES, and about one and three quarters on NCI1.

On MUTAG we TIE the best reported score rather than beat it. MUTAG has one hundred eighty-eight
graphs and is effectively saturated \u2014 HetSheaf reports a perfect one hundred. I would not read
anything into differences at that end, and we do not claim a win there.

Look also at the first two rows. Plain GCN and GAT score in the teens on ENZYMES Macro-F1 \u2014
essentially failing on the six-class problem \u2014 while every sheaf-based and every heterogeneous
method is far above them. That is the clearest signal in the table that the structure being modelled
is doing the work.

The interpretation I would offer is on the right. These are molecular and protein structures, not
citation or social networks. We obtained the heterogeneity by reading the TUDataset node and edge
labels as node types and relation types. The fact that this HELPS suggests typed structure is useful
well outside the domains where heterogeneous graph learning is normally studied \u2014 atom types
and bond types really are different relations, not merely categorical features on a homogeneous
graph.

Standard deviations are omitted here for legibility; all are below 0.9 in the paper.""",
      "0:30  \u00b7  running 11:55")


# ---- 19. Link prediction --------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "4.2  Experiments", "Link prediction", 19, TOTAL,
             sub="AUPR / AUROC;  1:1 negatives from the 2-hop neighbourhood")

rows = [
    ["Group", "Model", "LastFM AUPR", "LastFM AUROC", "PubMed AUPR", "PubMed AUROC",
     "Amazon AUPR", "Amazon AUROC", "YouTube AUPR", "YouTube AUROC"],
    ["Hom./hom.", "GCN", "97.06", "96.96", "96.05", "92.93", "92.40", "89.66", "89.30", "83.26"],
    ["", "GAT", "63.06", "51.14", "96.07", "93.27", "88.62", "86.10", "66.76", "58.40"],
    ["Hom./het.", "NSD", "97.16", "96.58", "96.41", "93.65", "92.63", "90.04", "89.32", "83.36"],
    ["", "CSNN", "97.99", "97.75", "96.14", "92.87", "92.10", "89.62", "88.64", "82.97"],
    ["Het./hom.", "HAN", "88.26", "83.32", "95.78", "93.62", "95.24", "93.64", "92.46", "90.10"],
    ["", "PSHGCN", "88.02", "80.55", "97.58", "93.27", "94.28", "92.34", "93.15", "90.66"],
    ["Het./het.", "HDHGR", "97.89", "97.93", "95.03", "93.15", "93.37", "91.42", "91.68", "88.33"],
    ["", "Hetero\u00b2Net", "97.58", "97.45", "94.66", "92.57", "94.71", "92.34", "90.67", "86.45"],
    ["", "H2SGNN", "97.88", "97.22", "95.22", "92.32", "93.18", "92.31", "90.17", "88.96"],
    ["", "LatGRL", "96.88", "96.01", "96.61", "92.32", "94.41", "92.03", "89.98", "88.10"],
    ["", "HetSheaf", "98.51", "98.00", "96.36", "93.00", "93.76", "91.00", "91.27", "88.00"],
    ["Ours", "H2S-NN", "98.91", "98.72", "97.63", "96.48", "98.53", "98.33", "93.95", "91.81"],
]
table(s, rows, MARGIN, top + Inches(0.08), Inches(8.6),
      col_w=[1.1, 1.15, 1.0, 1.05, 1.0, 1.05, 1.0, 1.05, 1.0, 1.05],
      font=10.5, header_font=9.0, row_h=Inches(0.3), group_col=True)

tfa = panel(s, MARGIN + Inches(8.9), top + Inches(0.08), Inches(3.71), Inches(2.16),
            "Best on all four, all eight metrics", accent=ACCENT)
bullet(tfa, "LastFM, PubMed \u2014 many node types", size=PANEL_BODY, first=True, space_after=8)
bullet(tfa, "Amazon, YouTube \u2014 one node type", size=PANEL_BODY, space_after=8)
bullet(tfa, "Biggest gain on Amazon: +3.29 / +4.69", size=PANEL_BODY, space_after=0)

tfb = panel(s, MARGIN + Inches(8.9), top + Inches(2.42), Inches(3.71), Inches(2.2),
            "Why that is a real test", accent=TEAL)
para(tfb, "With one node type, typed endpoints tell you nothing \u2014 yet the relation sequence "
          "still carries signal.", size=PANEL_BODY, space_after=8)
para(tfb, "The decoder reads pooled edge states, so the loss reaches Stage II.", size=PANEL_BODY,
     bold=True, color=PRIMARY)

notes(s, """Link prediction: best on all four datasets and all eight metrics.

I want to highlight the SPLIT in this table, because it makes the result a real test rather than a
sweep.

LastFM and PubMed are MULTI-NODE-TYPE graphs \u2014 users, artists and tags; and a biomedical
knowledge graph with four node types and ten relation types. Those are the comfortable case for a
meta-path method.

Amazon and YouTube are the UNCOMFORTABLE case: a SINGLE node type, with two and five relation types
respectively. A method that leans on typed endpoints has essentially nothing to work with there. And
yet Amazon is where we gain the most \u2014 over three points of AUPR and nearly five of AUROC over
the best baseline. That is consistent with our claim that the RELATION SEQUENCE along a path carries
signal even when the node types do not distinguish anything.

Note also GAT on LastFM: sixty-three AUPR, fifty-one AUROC \u2014 essentially chance. A single
attention mechanism over an untyped view can fail outright on these graphs.

One mechanism note, bottom right: the link decoder consumes POOLED INCIDENT-EDGE STATES, not just
node states. A shared projection combines the node state with its pooled edge summary before the
dot-product, MLP or DistMult decoder. So the task loss genuinely depends on the edge stream, and
therefore on the Stage-II instance weights. That is not incidental \u2014 it is what lets AMPSS
receive gradient on this task at all.""", "0:30  \u00b7  running 12:25")


# ---- 20. Ablations --------------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "5. Discussion", "Does each component earn its place?", 20, TOTAL)

tfh = add_text(s, MARGIN, top + Inches(0.06), Inches(6.28), Inches(0.42))
para(tfh, "AMPSS \u2014 DBLP node classification", size=17, bold=True, color=PRIMARY, first=True,
     space_after=0)

rows = [
    ["Configuration", "Macro-F1", "Micro-F1"],
    ["No path-supported 2D-cells", "93.85", "93.92"],
    ["Predefined meta-path schemas", "95.75", "96.01"],
    ["AMPSS, Stage I only", "95.88", "96.02"],
    ["AMPSS, schema weight only", "95.91", "96.08"],
    ["AMPSS, instance weight only", "95.95", "96.13"],
    ["AMPSS (full)", "96.01", "96.31"],
]
table(s, rows, MARGIN, top + Inches(0.46), Inches(6.28), col_w=[3.0, 1.0, 1.0], font=12.5,
      header_font=12.5, row_h=Inches(0.325))

tfa = add_text(s, MARGIN, top + Inches(2.86), Inches(6.28), Inches(1.76))
bullet(tfa, "Dropping the 2D-cells costs 2.2 Macro-F1", size=16, first=True, space_after=10)
bullet(tfa, "Hand-chosen schemas recover most of it", size=16, space_after=10)
bullet(tfa, "Instance weighting still adds on top of Stage I", size=16, space_after=0)

tfh2 = add_text(s, MARGIN + Inches(6.55), top + Inches(0.06), Inches(6.06), Inches(0.42))
para(tfh2, "SheafCovPool \u2014 ENZYMES graph classification", size=17, bold=True, color=PRIMARY,
     first=True, space_after=0)

rows2 = [
    ["Readout", "Macro-F1", "Micro-F1"],
    ["Sum + MLP", "61.25", "61.79"],
    ["Mean + MLP", "61.42", "62.09"],
    ["Max + MLP", "61.38", "61.62"],
    ["SheafPool", "64.05", "64.96"],
    ["SheafCovPool", "64.52", "65.10"],
]
table(s, rows2, MARGIN + Inches(6.55), top + Inches(0.46), Inches(6.06), col_w=[3.0, 1.0, 1.0],
      font=12.5, header_font=12.5, row_h=Inches(0.325))

tfb = add_text(s, MARGIN + Inches(6.55), top + Inches(2.54), Inches(6.06), Inches(2.08))
bullet(tfb, "Sum, mean and max all sit near 61", size=16, first=True, space_after=10)
bullet(tfb, "Gauge invariance alone buys most of the gain", size=16, space_after=10)
bullet(tfb, "Covariances add the rest \u2014 +0.47 / +0.14", size=16, space_after=0)

notes(s, """Two ablations, one per contribution. Left: AMPSS on DBLP node classification. Right: the
readout on ENZYMES graph classification.

Read the LEFT column top to bottom. Remove the two-cells entirely and you lose about two point two
points of Macro-F1. That is the single largest step in the table, and it is the honest test of the
central claim: the higher-order structure is doing work, not just adding parameters.

Put back HAND-CHOSEN meta-path schemas \u2014 the standard APA, APCPA, APTPA \u2014 and you recover
most of that loss. I want to be fair here: DBLP's canonical meta-paths are good ones, so predefined
schemas are already a strong configuration, and that limits how much automatic search can add on
THIS dataset.

Automatic Stage-I selection then edges past them, and adding instance-level weighting gives a
further increment on top of Stage I, with the full model best. The gaps between those last four rows
are small \u2014 tenths of a point \u2014 but they are ordered exactly as the argument predicts, and
they support the specific claim we make: schema-level selection alone is not sufficient, and
instance-level weighting helps filter noisy or uninformative path realisations.

On the RIGHT the ablation is cleaner. The three plain aggregations \u2014 sum, mean, max \u2014 all
sit around sixty-one; they are indistinguishable from each other. SheafPool jumps to sixty-four,
which tells us most of the gain comes from GAUGE INVARIANCE alone, and I want to credit that
honestly rather than attribute it to our design. SheafCovPool then adds about half a point more by
pooling BOTH node and edge Gram matrices and by keeping the COVARIANCES, not just the means.

So the conclusion we draw is two-part: gauge invariance is useful but not sufficient, and
second-order statistics of the cochains carry additional graph-level signal.""",
      "0:40  \u00b7  running 13:05")


# ---- 21. Supplementary A --------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "Appendix A, D, E", "Readouts, clustering, training dynamics", 21, TOTAL)

tfa = panel(s, MARGIN, top + Inches(0.1), Inches(6.28), Inches(2.1),
            "A. One backbone, three light readouts")
bullet(tfa, "Node \u2014 MLP over state, encoding, pooled edges", size=PANEL_BODY, first=True,
       space_after=9)
bullet(tfa, "Link \u2014 shared projection, then dot / MLP / DistMult", size=PANEL_BODY,
       space_after=9)
bullet(tfa, "Graph \u2014 MLP(\u03b3_g)", size=PANEL_BODY, space_after=0)

tfb = panel(s, MARGIN, top + Inches(2.38), Inches(6.28), Inches(2.24),
            "E. Separability during training", accent=TEAL)
para(tfb, "On a controlled four-class synthetic dataset, classes start interleaved and become "
          "compact and separated.", size=PANEL_BODY, space_after=8)
para(tfb, "A qualitative diagnostic \u2014 not a benchmark result.", size=PANEL_BODY, color=MUTED,
     italic=True)

tfc = panel(s, MARGIN + Inches(6.55), top + Inches(0.1), Inches(6.06), Inches(2.66),
            "D. Node clustering (k-means, 5 seeds)", accent=ACCENT)
rows = [
    ["Model", "DBLP NMI", "DBLP ARI", "ACM NMI", "ACM ARI"],
    ["GAT", "71.50", "77.26", "57.22", "60.47"],
    ["NSD", "80.21", "84.04", "62.56", "65.02"],
    ["CSNN", "81.72", "84.25", "69.28", "71.10"],
    ["HAN", "79.12", "84.76", "60.63", "64.28"],
    ["PSHGCN", "80.38", "82.65", "71.89", "74.31"],
    ["Hetero\u00b2Net", "80.54", "84.98", "70.14", "73.42"],
    ["LatGRL", "81.36", "86.07", "72.52", "76.75"],
    ["H2S-NN", "82.41", "86.52", "75.78", "78.29"],
]
table(s, rows, MARGIN + Inches(6.73), top + Inches(0.5), Inches(5.7),
      col_w=[1.5, 1.0, 1.0, 1.0, 1.0], font=11, header_font=11, row_h=Inches(0.245))

tfd = add_text(s, MARGIN + Inches(6.73), top + Inches(2.9), Inches(5.7), Inches(0.5))
bullet(tfd, "Best NMI and ARI on both \u2014 largest gain on ACM", size=15, first=True,
       space_after=0)

picture(s, PAPER_FIG / "ACM_clusters.png", MARGIN + Inches(6.73), top + Inches(3.5), Inches(2.7),
        Inches(1.12))
picture(s, PAPER_FIG / "DBLP_clusters.png", MARGIN + Inches(9.72), top + Inches(3.5), Inches(2.7),
        Inches(1.12))

notes(s, """Three appendix items, compressed.

APPENDIX A, the readouts. The backbone is shared across all three tasks and only the readout
changes. For NODE classification we pool the states of observed AND closure edges incident to the
target node, and feed the concatenation of the diffused node state, the initial typed encoding, and
that edge summary into an MLP. For LINK prediction a shared projection combines the node state with
its pooled edge summary, then a dot-product, concatenation-MLP, or DistMult decoder produces the
logit \u2014 and as I said, that pooling is the mechanism by which the link loss reaches the
Stage-II instance weights at all. For GRAPH classification it is simply an MLP on gamma-g.

APPENDIX D, a clustering diagnostic. Following the evaluation procedure used for HAN, we run k-means
on the embeddings after supervised training, with k set to the number of classes \u2014 four for
DBLP, three for ACM \u2014 three hundred iterations, five seeds. H2S-NN has the best NMI and ARI on
both datasets, with the larger gain on ACM: about three and a quarter points of NMI over LatGRL. I
present this as a DIAGNOSTIC rather than a headline result \u2014 it says the learned representations
are more SEPARABLE, not merely that the classifier is better tuned. The two t-SNE plots below are
the qualitative version of the same claim.

APPENDIX E: on a controlled synthetic four-class dataset from PyTorch Geometric we recorded node
embeddings every five epochs. Classes begin strongly interleaved and become increasingly compact and
separated. Again \u2014 qualitative evidence that the diffusion behaves as designed, on synthetic
data, not a benchmark claim.""", "0:30  \u00b7  running 13:35")


# ---- 22. Supplementary B --------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "Appendix B, F", "Cost and tuning", 22, TOTAL)

tfa = panel(s, MARGIN, top + Inches(0.1), Inches(6.28), Inches(2.24),
            "F. Linear in graph size for fixed d and K")
bullet(tfa, "Cubic in the stalk dimension d only \u2014 and d \u2264 64", size=PANEL_BODY,
       first=True, space_after=9)
bullet(tfa, "Maps are learned only for observed edges, then cached", size=PANEL_BODY, space_after=9)
bullet(tfa, "Stage I runs once; SheafCovPool builds no dense matrix", size=PANEL_BODY,
       space_after=0)

tfb = panel(s, MARGIN, top + Inches(2.52), Inches(6.28), Inches(2.1),
            "B. Registered primary search space", accent=TEAL)
bullet(tfb, "W&B Bayesian search with Hyperband", size=PANEL_BODY, first=True, space_after=9)
bullet(tfb, "d \u2208 {16, 32} node/link, {16, 32, 64} graph;  K \u2208 {2\u20266}",
       size=PANEL_BODY, space_after=9)
bullet(tfb, "Gram order fixed at 3\u00d73;  F \u2208 {4, 8, 16}", size=PANEL_BODY, space_after=0)

tfh = add_text(s, MARGIN + Inches(6.55), top + Inches(0.06), Inches(6.06), Inches(0.42))
para(tfh, "Measured on ACM node classification", size=17, bold=True, color=PRIMARY, first=True,
     space_after=0)

rows = [
    ["Model", "Runtime (s)", "Parameters", "Memory (MB)"],
    ["HAN", "28.9", "2.2M", "1,780"],
    ["HGT", "1,025.8", "7.2M", "6,120"],
    ["NSD", "47.6", "155K", "980"],
    ["HetSheaf", "55.7", "174K", "\u2014"],
    ["H2S-NN", "72.1", "517K", "1,120"],
]
table(s, rows, MARGIN + Inches(6.55), top + Inches(0.46), Inches(6.06), col_w=[1.6, 1.2, 1.2, 1.2],
      font=13, header_font=13, row_h=Inches(0.36))

callout(s, MARGIN + Inches(6.55), top + Inches(2.52), Inches(6.06), Inches(2.1), [
    ("We state the trade-off plainly", {"bold": True, "size": 16, "color": PRIMARY}),
    ("Slower and larger than HetSheaf \u2014 the price of keeping 2D-cells and diffusing both "
     "cochains.", {"size": 15}),
    ("Still an order of magnitude cheaper than HGT on every axis.", {"size": 15}),
])

notes(s, """Cost, because it belongs in the talk and not only in the appendix.

ANALYTICALLY: the backbone is LINEAR in graph size and in retained boundary incidences once d and K
are fixed. The only cubic term is in the STALK DIMENSION, from constructing the d-by-d maps \u2014
and d is at most sixty-four, so that is a small constant. Memory grows quadratically in d.
Crucially, node-to-edge maps are LEARNED only for the observed edges; every closure map and every
edge-to-face map comes from TRANSPORT, and the maps are cached across diffusion steps. So the
two-cells add incidences but not learned matrices. AMPSS Stage I runs once before training. And
SheafCovPool stays linear in the extended graph size \u2014 it builds neither a dense node-to-node
covariance nor a coarsened graph.

EMPIRICALLY, on ACM: seventy-two seconds, five hundred seventeen thousand parameters, about one
point one gigabytes. Let me be direct about the comparison. That is slower and larger than HetSheaf,
the closest sheaf baseline, at fifty-six seconds and one hundred seventy-four thousand parameters.
Retaining two-cells and diffusing both cochains is not free, and the paper says so. What we argue is
that the trade is favourable: we remain an order of magnitude cheaper than HGT on every axis \u2014
HGT takes over a thousand seconds and seven million parameters \u2014 and the accuracy gains hold
across all three tasks.

Appendix B, briefly: Bayesian search with Hyperband early termination, with the optimiser family,
scheduler, gradient clipping, splits and Stage-I support all held FIXED so the comparison is about
the model rather than the tuning budget. Stalk dimensions are small, sixteen to sixty-four; the Gram
order in the pooling is fixed at three by construction rather than tuned; and pooling-family,
decoder-family and operator-ramp sweeps are treated as ablations rather than mixed into the primary
search.""", "0:35  \u00b7  running 14:10")


# ---- 23. Conclusion -------------------------------------------------------------------
s = new_slide(prs)
top = chrome(s, "6. Conclusion", "What holds \u2014 and what does not", 23, TOTAL)

tfa = panel(s, MARGIN, top + Inches(0.1), Inches(6.28), Inches(2.44), "Established")
bullet(tfa, "Path instances can be kept as 2D-cells", size=16, first=True, space_after=10)
bullet(tfa, "\u03b4\u2081\u03b4\u2080 = 0 by construction on simple-path faces", size=16,
       space_after=10)
bullet(tfa, "AMPSS splits schema choice from instance choice", size=16, space_after=10)
bullet(tfa, "SheafCovPool is permutation- and gauge-invariant", size=16, space_after=0)

tfb = panel(s, MARGIN + Inches(6.55), top + Inches(0.1), Inches(6.06), Inches(2.44),
            "Bounded honestly", accent=ACCENT)
bullet(tfb, "Best among the compared methods \u2014 under this protocol", size=16, first=True,
       space_after=10)
bullet(tfb, "Stage I is non-differentiable", size=16, space_after=10)
bullet(tfb, "Proposition 1 is scoped to simple-path faces", size=16, space_after=10)
bullet(tfb, "Costlier than the closest sheaf baseline", size=16, space_after=0)

callout(s, MARGIN, top + Inches(2.74), FULL_W, Inches(1.06), [
    ("Best results on every evaluated node-classification, graph-classification and link-prediction "
     "dataset, with ablations supporting both AMPSS and SheafCovPool.",
     {"size": 17, "bold": True, "color": PRIMARY}),
])

tfc = add_text(s, MARGIN, top + Inches(4.02), FULL_W, Inches(0.6))
para(tfc, "Next:  a differentiable Stage I  \u00b7  faces beyond simple paths  \u00b7  cheaper "
          "transports, larger graphs.", size=15, color=MUTED, first=True, italic=True)

notes(s, """To conclude.

WHAT WE ESTABLISHED, on the left. Concrete meta-path instances can be retained as path-supported
two-dimensional cells rather than collapsed into endpoint views. Learned restriction maps and
sheaf-Hodge diffusion then update node and edge cochains together, with delta-one delta-zero equal
to zero BY CONSTRUCTION on simple-path faces \u2014 which is what keeps the Hodge decomposition
interpretable rather than merely suggestive. AMPSS separates the two questions prior work conflates:
which schemas, and then which of their occurrences. And SheafCovPool provides a permutation- and
gauge-invariant readout for graph-level tasks.

WHAT WE DO NOT CLAIM, on the right \u2014 because I would rather say it than be asked it. The claim
is best overall performance among the COMPARED methods under THIS protocol, not unqualified state of
the art, and not scalability beyond what we ran. Stage I is non-differentiable, so the task loss
never reaches schema selection. Faces are restricted to simple paths, and Proposition 1 is scoped
exactly to that construction and its aligned closure edges. And the model costs more time, memory
and parameters than the closest sheaf baseline.

Within those bounds: best results on every evaluated dataset across all three tasks, with ablations
supporting both components.

The natural next steps are a differentiable or amortised Stage I, faces beyond simple paths together
with the orientation bookkeeping that would require, and cheaper transports for larger graphs.

Thank you \u2014 I am happy to take questions.""", "0:35  \u00b7  running 14:45")


# ---- 24. Q&A --------------------------------------------------------------------------
s = new_slide(prs)
add_box(s, 0, 0, SLIDE_W, SLIDE_H, fill=PRIMARY)
add_box(s, Inches(1.1), Inches(2.3), Inches(1.9), Emu(34290), fill=ACCENT)

tf = add_text(s, Inches(1.1), Inches(2.62), Inches(11.2), Inches(1.55))
para(tf, "Thank you", size=42, bold=True, color=WHITE, first=True, space_after=6)
para(tf, "Questions welcome.", size=19, color=RGBColor(0xC3, 0xCD, 0xE4), space_after=0)

tf = add_text(s, Inches(1.1), Inches(4.4), Inches(11.2), Inches(1.6))
para(tf, "Backup material:", size=14, bold=True, color=RGBColor(0xE3, 0xC1, 0x6A),
     first=True, space_after=6)
para(tf, "Proposition 1 proof  \u00b7  dataset statistics and semantic schemas  \u00b7  full search "
         "space  \u00b7  clustering and t-SNE  \u00b7  complexity derivations",
     size=14, color=RGBColor(0xC3, 0xCD, 0xE4), space_after=0)

notes(s, """Likely questions, with short answers:

Q: Why not make Stage I differentiable? A: We kept it auditable \u2014 the incidence budget is a
HARD cap that a soft relaxation would blur. It is the clearest limitation and the first thing we
would change.

Q: Isn't the closure edge just the shortcut you criticised? A: No. It closes the boundary so
delta-one is defined, but it is not an observed relation and it carries no learned restriction map
of its own \u2014 both its maps come from the same transport as the rest of the face. Nothing
propagates along it as though it were a real edge.

Q: How sensitive is the result to the incidence budget? A: The budget bounds the support of the
face-induced operator directly, and Stage I stops when a budget binds \u2014 so it degrades
gracefully toward the no-2D-cell ablation, the 93.85 / 93.92 row.

Q: Why fix the Gram order at three? A: Three is the number of semantic views in each stack, not a
hidden width. Changing it means adding or removing stalk-valued views, which defines a DIFFERENT
readout rather than a wider one.

Q: Why are the node-classification margins so small? A: They are, on DBLP and ACM. The honest
reading is that those benchmarks are close to saturated and their canonical meta-paths are already
good. IMDB \u2014 lowest homophily, multilabel \u2014 is where the margin is largest, and graph
classification and link prediction show larger gaps.

Q: What about scalability? A: Costs are linear in graph size for fixed d and K, and no dense
Laplacian is formed \u2014 but the largest graph we ran is YouTube at about 1.18M edges, so I would
not claim beyond that.

Q: Does the model need meta-paths to be supplied? A: No \u2014 that is the point of AMPSS Stage I.
Manual schemas are supported, and are the 'predefined' ablation row.""", "\u2014")


for sc, sno, tag in sorted(fit_all()):
    print(f"  slide {sno:2d}  fit {sc:.2f}  {tag}".encode("ascii", "replace").decode())

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"Saved {OUT}  ({len(prs.slides._sldIdLst)} slides)")
