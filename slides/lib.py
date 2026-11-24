"""Drawing helpers: every element goes into the .pptx (python-pptx, on the style template) and into a PIL preview.
Units: px on a 1920x1080 grid (slide is 10 x 5.625 in, so 1 px = 1/192 in = 0.375 pt)."""
import copy, math, os, re
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from tplrender import draw_layout

# Slidesgo "Elegant Black and White Thesis Defense" template (not redistributed here; download it yourself)
TEMPLATE = os.environ.get("SLIDES_TEMPLATE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "style_template.pptx"))
NAVY, BLUE, TINT, TINT2, WHITE, INK, MUTED = "0E2A47", "869FB2", "E3E9EE", "F2F5F7", "FFFFFF", "0E2A47", "5B6B7C"
BAND, FIB = "E9EEF2", "869FB2"
HEAD, BODY = "Figtree Black", "Hanken Grotesk"
# measuring/preview fonts (the real fonts are not installed here); widths scaled to stay conservative
PF = {HEAD: ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 1.12),
      BODY: ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 1.04),
      BODY + "B": ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 1.06)}
_fc = {}
def pfont(face, bold, size):
    key = HEAD if face == HEAD else (BODY + "B" if bold else BODY)
    k = (key, size)
    if k not in _fc: _fc[k] = ImageFont.truetype(PF[key][0], max(6, int(round(size * PF[key][1]))))
    return _fc[k]

EMU = 914400 / 192
def E(v): return Emu(int(round(v * EMU)))
def rgb(h): return RGBColor.from_string(h)
def hx(h): return "#" + h
def PT(px): return Pt(px * 0.375)

prs = Presentation(TEMPLATE)
# drop the template's example slides (layouts, masters and theme stay)
lst = prs.slides._sldIdLst
for sid in list(lst):
    prs.part.drop_rel(sid.rId); lst.remove(sid)
LAYOUT = {l.name: l for l in prs.slide_masters[0].slide_layouts}
WARN = []
SLIDES = []

def parse(par, color, bold):
    runs = []
    for tok in re.split(r"(\*\*.*?\*\*)", par):
        if not tok: continue
        runs.append((tok[2:-2], True, color) if tok.startswith("**") else (tok, bold, color))
    return runs

def wrap(runs, face, size, width):
    words = []
    for t, b, c in runs:
        for w in re.split(r"( )", t):
            if w: words.append((w, b, c))
    lines, cur, cw = [], [], 0
    for w, b, c in words:
        wl = pfont(face, b, size).getlength(w)
        if w == " ":
            if cur: cur.append((w, b, c)); cw += wl
            continue
        if cw + wl > width and cur:
            while cur and cur[-1][0] == " ": cur.pop()
            lines.append(cur); cur, cw = [], 0
        if wl > width: WARN.append(f"word too wide: {w!r}")
        cur.append((w, b, c)); cw += wl
    if cur: lines.append(cur)
    return lines or [[]]

class S:
    def __init__(self, layout, key, notes=""):
        self.key = key
        self.s = prs.slides.add_slide(LAYOUT[layout])
        self.im = Image.new("RGB", (1920, 1080), "white"); self.d = ImageDraw.Draw(self.im)
        draw_layout(LAYOUT[layout]._element, self.im)
        self.n = 0
        if notes: self.s.notes_slide.notes_text_frame.text = notes
        SLIDES.append(self)

    def _nm(self, sh, kind):
        self.n += 1; sh.name = f"{self.key}-{kind}-{self.n}"

    # ---------- text into a placeholder or a new text box
    def _fill(self, tf, paras, size, color, face, bold, align, ls, gap):
        tf.word_wrap = True
        for i, par in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
            p.line_spacing = PT(size * ls)
            p.space_before = Pt(0)
            p.space_after = PT(gap) if i < len(paras) - 1 else Pt(0)
            for t, b, c in parse(par, color, bold):
                r = p.add_run(); r.text = t; f = r.font
                f.name = face; f.size = PT(size); f.bold = b if face == BODY else None; f.color.rgb = rgb(c)

    def _draw(self, x, y, w, h, paras, size, color, face, bold, align, anchor, ls, gap, check):
        allines = [wrap(parse(p, color, bold), face, size, w) for p in paras]
        total = sum(len(l) for l in allines) * size * ls + gap * (len(paras) - 1)
        if check and total > h + 1: WARN.append(f"{self.key}: overflow {total:.0f}>{h:.0f} in {paras[0][:50]!r}")
        cy = y + {"t": 0, "m": max(0, (h - total) / 2), "b": max(0, h - total)}[anchor]
        for lines in allines:
            for ln in lines:
                lw = sum(pfont(face, b, size).getlength(t) for t, b, c in ln)
                cx = x + {"l": 0, "c": (w - lw) / 2, "r": w - lw}[align]
                for t, b, c in ln:
                    f = pfont(face, b, size)
                    self.d.text((cx, cy + size * (ls - 1) / 2 + size * 0.02), t, font=f, fill=hx(c)); cx += f.getlength(t)
                cy += size * ls
            cy += gap
        return total

    def text(self, x, y, w, h, paras, size=37, color=INK, face=BODY, bold=False, align="l", anchor="t", ls=1.2, gap=None, check=True):
        if isinstance(paras, str): paras = [paras]
        gap = size * 0.5 if gap is None else gap
        tb = self.s.shapes.add_textbox(E(x), E(y), E(w), E(h)); self._nm(tb, "text")
        tf = tb.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
        self._fill(tf, paras, size, color, face, bold, align, ls, gap)
        return self._draw(x, y, w, h, paras, size, color, face, bold, align, anchor, ls, gap, check)

    def ph(self, idx, paras, x, y, w, h, size, color=INK, face=HEAD, bold=False, align="l", anchor="t", ls=1.1, gap=0):
        if isinstance(paras, str): paras = [paras]
        p = self.s.placeholders[idx]
        p.left, p.top, p.width, p.height = E(x), E(y), E(w), E(h)
        tf = p.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
        self._fill(tf, paras, size, color, face, bold, align, ls, gap)
        return self._draw(x, y, w, h, paras, size, color, face, bold, align, anchor, ls, gap, True)

    def title(self, t):
        """Content-slide title: affirmative sentence, Figtree Black, up to 3 lines above the content area."""
        for size in (58, 52, 46):
            n = len(wrap(parse(t, NAVY, False), HEAD, size, 1610))
            if n <= 2 or (n == 3 and size <= 52): break
        self.ph(0, t, 152, 78, 1610, 196, size, anchor="m")

    # ---------- shapes
    def rect(self, x, y, w, h, fill=None, line=None, lw=3, r=0, dash=False):
        shp = self.s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if r else MSO_SHAPE.RECTANGLE, E(x), E(y), E(w), E(h))
        if r: shp.adjustments[0] = min(0.5, r / min(w, h))
        if fill: shp.fill.solid(); shp.fill.fore_color.rgb = rgb(fill)
        else: shp.fill.background()
        if line:
            shp.line.color.rgb = rgb(line); shp.line.width = E(lw)
            if dash: shp.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        else: shp.line.fill.background()
        shp.shadow.inherit = False; self._nm(shp, "box")
        if fill: self.d.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=hx(fill))
        if line:
            if dash: self._dash([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)], line, lw)
            else: self.d.rounded_rectangle([x, y, x + w, y + h], radius=r, outline=hx(line), width=max(1, int(round(lw))))
        return shp

    def oval(self, x, y, w, h=None, fill=NAVY, line=None, lw=3):
        h = w if h is None else h
        shp = self.s.shapes.add_shape(MSO_SHAPE.OVAL, E(x), E(y), E(w), E(h))
        if fill: shp.fill.solid(); shp.fill.fore_color.rgb = rgb(fill)
        else: shp.fill.background()
        if line: shp.line.color.rgb = rgb(line); shp.line.width = E(lw)
        else: shp.line.fill.background()
        shp.shadow.inherit = False; self._nm(shp, "dot")
        self.d.ellipse([x, y, x + w, y + h], fill=hx(fill) if fill else None, outline=hx(line) if line else None, width=lw)

    def _dash(self, pts, color, w, on=14, off=10):
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            L = math.hypot(x2 - x1, y2 - y1); t = 0
            while t < L:
                t2 = min(t + on, L)
                self.d.line([(x1 + (x2 - x1) * t / L, y1 + (y2 - y1) * t / L), (x1 + (x2 - x1) * t2 / L, y1 + (y2 - y1) * t2 / L)], fill=hx(color), width=max(1, int(round(w))))
                t += on + off

    def line(self, x1, y1, x2, y2, color=NAVY, w=3, dash=False):
        c = self.s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, E(x1), E(y1), E(x2), E(y2))
        c.line.color.rgb = rgb(color); c.line.width = E(w)
        if dash: c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        self._nm(c, "line")
        if dash: self._dash([(x1, y1), (x2, y2)], color, w)
        else: self.d.line([(x1, y1), (x2, y2)], fill=hx(color), width=max(1, int(round(w))))

    def poly(self, pts, color=None, w=3, fill=None, closed=False):
        fb = self.s.shapes.build_freeform(E(pts[0][0]), E(pts[0][1]), scale=1.0)
        fb.add_line_segments([(E(x), E(y)) for x, y in pts[1:]], close=closed)
        shp = fb.convert_to_shape()
        if fill: shp.fill.solid(); shp.fill.fore_color.rgb = rgb(fill)
        else: shp.fill.background()
        if color: shp.line.color.rgb = rgb(color); shp.line.width = E(w)
        else: shp.line.fill.background()
        shp.shadow.inherit = False; self._nm(shp, "shape")
        if fill: self.d.polygon(pts, fill=hx(fill))
        if color: self.d.line(pts + ([pts[0]] if closed else []), fill=hx(color), width=max(1, int(round(w))), joint="curve")

    def arrow(self, x1, y1, x2, y2, color=NAVY, w=5, head=22, dash=False):
        """Straight arrow: line plus a filled triangle head at (x2, y2)."""
        L = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / L, (y2 - y1) / L
        bx, by = x2 - ux * head, y2 - uy * head
        self.line(x1, y1, bx, by, color, w, dash)
        px, py = -uy * head * 0.55, ux * head * 0.55
        self.poly([(x2, y2), (bx + px, by + py), (bx - px, by - py)], fill=color, closed=True)

    def table(self, x, y, w, cols, rows, size=34, highlight=(), right=(), pad=(22, 14)):
        widths = [w * c for c in cols]; heights = []
        for r in rows:
            mx = 1
            for txt, cw in zip(r, widths):
                mx = max(mx, len(wrap(parse(txt, INK, False), BODY, size, cw - 2 * pad[0])))
            heights.append(mx * size * 1.2 + 2 * pad[1])
        gt = self.s.shapes.add_table(len(rows), len(cols), E(x), E(y), E(w), E(sum(heights))); self._nm(gt, "table")
        tbl = gt.table; tblPr = tbl._tbl.tblPr; tblPr.set("bandRow", "0"); tblPr.set("firstRow", "0")
        for j, cw in enumerate(widths): tbl.columns[j].width = E(cw)
        cy = y
        for i, (r, rh) in enumerate(zip(rows, heights)):
            tbl.rows[i].height = E(rh)
            fill = NAVY if i == 0 else (TINT if i in highlight else (WHITE if i % 2 else TINT2))
            col = WHITE if i == 0 else INK
            self.d.rectangle([x, cy, x + w, cy + rh], fill=hx(fill))
            cx = x
            for j, (txt, cw) in enumerate(zip(r, widths)):
                cell = tbl.cell(i, j); cell.fill.solid(); cell.fill.fore_color.rgb = rgb(fill)
                cell.margin_left = cell.margin_right = E(pad[0]); cell.margin_top = cell.margin_bottom = E(pad[1])
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                al = "r" if j in right else "l"
                b = i == 0 or i in highlight
                self._fill(cell.text_frame, [txt], size, col, BODY, b, al, 1.2, 0)
                lines = wrap(parse(txt, col, b), BODY, size, cw - 2 * pad[0])
                ty = cy + (rh - len(lines) * size * 1.2) / 2
                for ln in lines:
                    lw = sum(pfont(BODY, bb, size).getlength(t) for t, bb, c in ln)
                    tx = cx + pad[0] + (cw - 2 * pad[0] - lw if al == "r" else 0)
                    for t, bb, c in ln:
                        f = pfont(BODY, bb, size); self.d.text((tx, ty + size * 0.12), t, font=f, fill=hx(c)); tx += f.getlength(t)
                    ty += size * 1.2
                cx += cw
            self.d.line([(x, cy + rh), (x + w, cy + rh)], fill=hx(TINT), width=2)
            cy += rh
        return cy

    def bar_chart(self, x, y, w, h, cats, vals, fmt='0', colors=None, horizontal=True, vmax=None, size=32, label_fmt=None, gap=60):
        cd = CategoryChartData(); cd.categories = cats; cd.add_series("s", vals)
        ct = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
        gf = self.s.shapes.add_chart(ct, E(x), E(y), E(w), E(h), cd); self._nm(gf, "chart")
        ch = gf.chart; ch.has_legend = False; ch.has_title = False
        ch.font.name = BODY; ch.font.size = PT(size); ch.font.color.rgb = rgb(INK)
        pl = ch.plots[0]; pl.gap_width = gap; pl.vary_by_categories = False
        pl.has_data_labels = True; dl = pl.data_labels; dl.number_format = fmt; dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = PT(size); dl.font.name = BODY; dl.font.bold = True
        ser = pl.series[0]; ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(NAVY)
        cols = colors or [NAVY] * len(vals)
        for i, c in enumerate(cols):
            pt = ser.points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(c)
        va = ch.value_axis; va.visible = False; va.has_major_gridlines = False
        if vmax: va.maximum_scale = vmax
        va.minimum_scale = 0
        ca = ch.category_axis; ca.format.line.color.rgb = rgb(TINT); ca.tick_labels.font.size = PT(size)
        ca.has_major_gridlines = False
        if horizontal: ca.reverse_order = True
        # preview
        vmax = vmax or max(vals) * 1.15
        n = len(vals); lab = label_fmt or (lambda v: f"{v}")
        if horizontal:
            lw = max(pfont(BODY, False, size).getlength(c) for c in cats) + 24
            bh = h / n
            for i, (c, v) in enumerate(zip(cats, vals)):
                by = y + i * bh; bwid = (w - lw - 140) * v / vmax; hh = bh * 100 / (100 + gap)
                self.d.text((x + lw - 16 - pfont(BODY, False, size).getlength(c), by + bh / 2 - size * 0.6), c, font=pfont(BODY, False, size), fill=hx(INK))
                self.d.rectangle([x + lw, by + (bh - hh) / 2, x + lw + bwid, by + (bh + hh) / 2], fill=hx(cols[i]))
                self.d.text((x + lw + bwid + 12, by + bh / 2 - size * 0.6), lab(v), font=pfont(BODY, True, size), fill=hx(INK))
        else:
            bw = w / n; base = y + h - size * 2.6
            for i, (c, v) in enumerate(zip(cats, vals)):
                ww = bw * 100 / (100 + gap); bx = x + i * bw + (bw - ww) / 2
                top = base - (base - y - size * 1.4) * v / vmax
                self.d.rectangle([bx, top, bx + ww, base], fill=hx(cols[i]))
                f = pfont(BODY, True, size); t = lab(v)
                self.d.text((bx + ww / 2 - f.getlength(t) / 2, top - size * 1.3), t, font=f, fill=hx(INK))
                f2 = pfont(BODY, False, size)
                for k, part in enumerate(c.split("\n")):
                    self.d.text((bx + ww / 2 - f2.getlength(part) / 2, base + 10 + k * size * 1.15), part, font=f2, fill=hx(INK))
        return ch

    def line_chart(self, x, y, w, h, cats, vals, size=30, xtitle="", ytitle=""):
        cd = CategoryChartData(); cd.categories = cats; cd.add_series("s", vals)
        gf = self.s.shapes.add_chart(XL_CHART_TYPE.LINE, E(x), E(y), E(w), E(h), cd); self._nm(gf, "chart")
        ch = gf.chart; ch.has_legend = False; ch.has_title = False
        ch.font.name = BODY; ch.font.size = PT(size); ch.font.color.rgb = rgb(INK)
        ser = ch.plots[0].series[0]; ser.format.line.color.rgb = rgb(NAVY); ser.format.line.width = E(6); ser.smooth = False
        ser.marker.style = None
        from pptx.enum.chart import XL_MARKER_STYLE, XL_TICK_LABEL_POSITION
        ser.marker.style = XL_MARKER_STYLE.NONE
        va = ch.value_axis; va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = rgb(TINT)
        va.minimum_scale = 0; va.tick_labels.font.size = PT(size); va.format.line.fill.background()
        va.tick_labels.number_format = '0.00000'; va.tick_labels.number_format_is_linked = False
        ca = ch.category_axis; ca.tick_labels.font.size = PT(size); ca.format.line.color.rgb = rgb(BLUE)
        ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
        from pptx.oxml.ns import qn
        cax = ca._element
        for tag, val in (("c:tickLblSkip", "10"), ("c:tickMarkSkip", "10")):
            el = cax.find(qn(tag))
            if el is None:
                el = etree.SubElement(cax, qn(tag)); el.set("val", val)
        # reorder children of catAx to schema order is required: noMultiLvlLbl must be last
        nm = cax.find(qn("c:noMultiLvlLbl"))
        if nm is not None: cax.remove(nm); cax.append(nm)
        # preview
        lx = x + 170; by = y + h - 60; ty = y + 20
        vmax = max(vals)
        for k in range(5):
            gy = by - (by - ty) * k / 4
            self.d.line([(lx, gy), (x + w - 20, gy)], fill=hx(TINT), width=2)
            t = f"{vmax * k / 4:.5f}"; f = pfont(BODY, False, size)
            self.d.text((lx - 12 - f.getlength(t), gy - size * 0.6), t, font=f, fill=hx(INK))
        pts = [(lx + (x + w - 20 - lx) * i / (len(vals) - 1), by - (by - ty) * v / vmax) for i, v in enumerate(vals)]
        self.d.line(pts, fill=hx(NAVY), width=6)
        for i in range(0, len(cats), 10):
            f = pfont(BODY, False, size); t = str(cats[i])
            self.d.text((pts[i][0] - f.getlength(t) / 2, by + 12), t, font=f, fill=hx(INK))
        return ch

def save(path, prevdir):
    os.makedirs(prevdir, exist_ok=True)
    for f in os.listdir(prevdir):
        if f.startswith("p") and f.endswith(".png"): os.remove(os.path.join(prevdir, f))
    for i, sl in enumerate(SLIDES, 1):
        sl.im.save(os.path.join(prevdir, f"p{i:02d}.png"))
    prs.core_properties.title = "How submission v3 measures muscle architecture"
    prs.save(path)
