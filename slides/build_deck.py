"""Submission v3 explainer deck on the style template: one message per slide, affirmative titles.

Usage: python slides/build_deck.py OUT.pptx [short]
"short" builds the ~10-minute version (background slides hidden). PNG previews go to slides/prev/."""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

OUT = sys.argv[1]
SHORT = len(sys.argv) > 2 and sys.argv[2] == "short"
PREV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prev")
L, R, W, TOP, BOT = 152, 1768, 1616, 310, 990

SECTIONS = [("01", "Background", "The challenge, the measurements and the data"),
            ("02", "Preprocessing", "What happens to the images before training"),
            ("03", "The model", "The U-Net, how it was trained and the v3 models"),
            ("04", "From masks to measurements", "Geometry and the millimetre scale"),
            ("05", "Results", "Validation and the leaderboard")]

# short version: background is hidden, so the visible sections are renumbered
SEC_KEYS = ["sec-bg", "sec-pre", "sec-mod", "sec-geo", "sec-res"]
if SHORT:
    SECTIONS[2] = ("03", "The model", "The U-Net architecture and the v3 models")
    SECTIONS[1] = ("02", "Preprocessing", "Cropping, image size and the input test")
    SHOWN = [("01",) + SECTIONS[1][1:], ("02",) + SECTIONS[2][1:], ("03",) + SECTIONS[3][1:], ("04",) + SECTIONS[4][1:]]
    NUMS = ["00", "01", "02", "03", "04"]
else:
    SHOWN = SECTIONS; NUMS = [s_[0] for s_ in SECTIONS]
def section(i, notes):
    _, title, sub = SECTIONS[i]; num = NUMS[i]
    s = S("SECTION_HEADER", SEC_KEYS[i], notes)
    s.d.rectangle([152, 111, 325, 259], fill="#0E2A47")
    s.ph(2, num, 152, 111, 173, 148, 96, color=WHITE, align="c", anchor="m")
    s.ph(0, title, 255, 400, 1064, 190, 88 if len(title) < 16 else 72, anchor="b")
    s.ph(1, sub, 255, 600, 1064, 100, 40, face=BODY, color=INK)
    return s

def content(key, title, notes):
    s = S("TITLE_ONLY", key, notes); s.title(title); return s

def caption(s, x, y, w, t, size=32, align="c", color=INK, h=None, bold=False):
    return s.text(x, y, w, h or size * 2.6, t, size=size, align=align, color=color, bold=bold)

# ---------------------------------------------------------------- schematic ultrasound image
def muscle(s, x, y, w, mt=False, pa=False, fl=False, fibres=True, hi=False, bands=True, bg=NAVY, mirror=False, rot=0, zoom=1.0, light=0):
    """Drawn B-mode schematic in a 760x560 frame scaled to width w. Returns (scale, mapper)."""
    k = w / 760; h = 560 * k
    s.rect(x, y, w, h, fill=bg, r=14 * k)
    cx, cy = 380, 280
    def P(px, py):
        if mirror: px = 760 - px
        px, py = cx + (px - cx) * zoom, cy + (py - cy) * zoom
        if rot:
            a = math.radians(rot); dx, dy = px - cx, py - cy
            px, py = cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a)
        return (x + px * k, y + py * k)
    def clipseg(a, b):
        # clip segment to frame (Liang-Barsky)
        (x1, y1), (x2, y2) = P(*a), P(*b)
        t0, t1 = 0.0, 1.0; dx, dy = x2 - x1, y2 - y1
        for p, q in ((-dx, x1 - x), (dx, x + w - x1), (-dy, y1 - y), (dy, y + h - y1)):
            if p == 0:
                if q < 0: return None
                continue
            t = q / p
            if p < 0: t0 = max(t0, t)
            else: t1 = min(t1, t)
        if t0 > t1: return None
        return (x1 + dx * t0, y1 + dy * t0, x1 + dx * t1, y1 + dy * t1)
    def seg(a, b, col, wd):
        c = clipseg(a, b)
        if c: s.line(*c, col, max(2, wd * k))
    band, fib = (BAND, FIB) if not light else ("FFFFFF", "C9D6E0")
    if bands:
        seg((-200, 122), (960, 101), band, 16); seg((-200, 438), (960, 390), band, 18)
    if fibres:
        for a, b in [((-40, 428), (240, 115)), ((80, 425), (360, 112)), ((320, 415), (600, 108)), ((440, 410), (720, 106)), ((560, 405), (840, 104))]:
            seg(a, b, fib, 5)
    if hi or pa or fl:
        seg((200, 421), (480, 111), WHITE, 9)
    if fl:
        seg((200, 421), (480, 111), WHITE, 9)
        for (px, py) in [(200, 421), (480, 111)]:
            X, Y = P(px, py); s.oval(X - 11 * k * 1.6, Y - 11 * k * 1.6, 22 * k * 1.6, fill=WHITE)
    if pa:
        arc = [P(200 + 95 * math.cos(math.radians(-t)), 421 + 95 * math.sin(math.radians(-t))) for t in range(4, 50, 3)]
        s.poly(arc, WHITE, max(2, 5 * k))
    if mt:
        X1, Y1 = P(680, 132); X2, Y2 = P(680, 386)
        s.arrow(X1, Y2, X1, Y1, WHITE, max(2, 5 * k), 22 * k * 1.4); s.arrow(X1, Y1, X1, Y2, WHITE, max(2, 5 * k), 22 * k * 1.4)
    return k, P

# ================================================================= cover
s = S("TITLE", "cover", "This talk presents our entry in the UMUD challenge: a pipeline that measures three properties of leg muscles in ultrasound images. Submission v3 is the version that scored best on the leaderboard. No prior knowledge of ultrasound or deep learning is assumed; a glossary is at the end.")
s.ph(0, "How submission v3 measures muscle architecture in ultrasound images", 228, 200, 1238, 420, 84, anchor="b", ls=1.08)
s.ph(1, "UMUD challenge · U-Net segmentation and geometry · leaderboard score 0.50942", 228, 660, 1100, 110, 38, face=BODY)

# ================================================================= index
s = content("index", "The presentation follows our solution from the images to the leaderboard score" if SHORT else "The presentation follows the project from the data to the leaderboard score", "The talk has five parts. First the background: what the challenge asks and what the data looks like. Then the preprocessing, the model, the geometry that turns model output into measurements, and the results.")
for i, (num, t, sub) in enumerate(SHOWN):
    y = TOP + i * 132
    s.rect(L, y, 110, 104, fill=NAVY)
    s.text(L, y, 110, 104, num, size=56, face=HEAD, color=WHITE, align="c", anchor="m")
    s.text(L + 150, y + 4, 1100, 54, t, size=42, face=HEAD, color=NAVY)
    s.text(L + 150, y + 60, 1100, 44, sub, size=32, color=MUTED)

# ================================================================= 01 background
section(0, "This part explains the challenge and the data: what an ultrasound image of a muscle shows, which three numbers we must predict, how the experts measured them, how the score works, and what the training and test images look like.")

s = content("bg-image", "An ultrasound image shows the muscle from the side: two bright bands with slanted fibres between them", "An ultrasound probe is placed on the skin along a leg muscle. The image is a side view into the leg. The top is the skin side and the bottom is deeper; images are never upside down. The two bright bands are the aponeuroses, the connective tissue that wraps the muscle. The faint slanted lines between them are fascicles, bundles of muscle fibres. All pictures in this deck are drawn schematics, because the competition rules do not allow the data to be shared.")
k, P = muscle(s, L, TOP + 20, 900)
fx = L + 900
for (py, t) in [(30, "Skin side (top of the image)"), (112, "Superficial aponeurosis"), (265, "Fascicles (fibre bundles)"), (420, "Deep aponeurosis"), (520, "Deeper inside the leg")]:
    yy = TOP + 20 + py * k
    s.line(fx - 30 if py in (112, 420) else fx, yy, fx + 70, yy, NAVY, 3)
    s.text(fx + 90, yy - 24, 640, 50, t, size=36, color=NAVY, bold=py in (112, 265, 420))

s = content("bg-measures", "The challenge asks for three numbers per image: muscle thickness, pennation angle and fascicle length", "Muscle thickness is the gap between the two bands, measured straight across at three places and averaged. Pennation angle is how slanted the fibres are, measured against the deep band. Fascicle length is the length of a fibre from band to band; when a fibre runs out of the picture, it is extended in a straight line. The test set has 309 images.")
pw = 500
for i, (kw, t, sub) in enumerate([(dict(mt=True), "Muscle thickness (MT, mm)", "gap between the bands, at 3 places"),
                                  (dict(pa=True), "Pennation angle (PA, °)", "fibre angle against the deep band"),
                                  (dict(fl=True), "Fascicle length (FL, mm)", "fibre length from band to band")]):
    x = L + i * (pw + 58)
    muscle(s, x, TOP + 20, pw, **kw)
    caption(s, x, TOP + 400, pw, t, size=36, bold=True, color=NAVY, h=50)
    caption(s, x, TOP + 456, pw, sub, size=32, color=MUTED, h=90)

s = content("bg-experts", "Two experts measured each test image by hand, and their average is the reference value", "For every test image, two researchers marked the two aponeuroses, three fascicles, three angles and three thickness measurements. Their results were averaged. When they disagreed by more than 4 degrees, 10 mm or 1 mm, they discussed and re-measured.")
def person(s, x, y, col=NAVY):
    s.oval(x + 45, y, 70, fill=col); s.rect(x + 15, y + 82, 130, 110, fill=col, r=50)
for i in range(2):
    person(s, L + 80, TOP + 40 + i * 300)
    caption(s, L + 250, TOP + 75 + i * 300, 420, f"Expert {i + 1}: 3 fascicles, 3 angles, 3 thicknesses", size=32, align="l", h=90)
    s.arrow(L + 700, TOP + 135 + i * 300, L + 860, TOP + 290 + (-60 if i == 0 else 60), NAVY, 5)
s.oval(L + 870, TOP + 150, 280, fill=TINT)
s.text(L + 870, TOP + 150, 280, 280, ["Average"], size=44, face=HEAD, color=NAVY, align="c", anchor="m")
s.arrow(L + 1170, TOP + 290, L + 1300, TOP + 290, NAVY, 5)
s.rect(L + 1320, TOP + 200, 290, 180, fill=NAVY, r=18)
s.text(L + 1320, TOP + 200, 290, 180, ["Reference value"], size=38, face=HEAD, color=WHITE, align="c", anchor="m")

s = content("bg-score", "The score divides each error by a tolerance and averages the three, so lower is better", "The UMUD score is a normalised mean absolute error. Each error is divided by a fixed tolerance so the three measurements count about equally despite their different units. The organisers have not published the tolerances. A reasonable guess is the raters' discussion thresholds: 4 degrees, 10 mm and 1 mm. If that is right, a 1 mm thickness error costs as much as a 4 degree angle error.")
for i, (a, b) in enumerate([("|PA error|", "tolerance PA"), ("|FL error|", "tolerance FL"), ("|MT error|", "tolerance MT")]):
    x = L + i * 400
    s.rect(x, TOP + 80, 330, 260, fill=TINT, r=18)
    s.text(x, TOP + 110, 330, 60, a, size=42, face=HEAD, color=NAVY, align="c")
    s.line(x + 50, TOP + 205, x + 280, TOP + 205, NAVY, 4)
    s.text(x, TOP + 230, 330, 60, b, size=36, color=NAVY, align="c")
    if i < 2: s.text(x + 330, TOP + 160, 70, 80, "+", size=60, face=HEAD, color=NAVY, align="c")
s.text(L + 1200, TOP + 160, 80, 80, "÷ 3", size=50, face=HEAD, color=NAVY, check=False)
s.text(L + 1310, TOP + 130, 300, 140, ["= score"], size=60, face=HEAD, color=NAVY, anchor="m")
caption(s, L, TOP + 420, 1450, "Tolerances are not published. The experts’ re-check thresholds (4°, 10 mm, 1 mm) are a likely stand-in.", size=32, align="l", color=MUTED)

s = content("bg-counts", "The training data has 2,761 fascicle images and 1,048 aponeurosis images, and no image has both labels", "The training images come from a published dataset (Ritsche et al., 2024). One folder has images with fascicle masks, the other has images with aponeurosis masks. No image appears with both, so we train two separate models. Circle areas are proportional to the image counts.")
r1, r2 = 300, 300 * math.sqrt(1048 / 2761)
s.oval(L + 160, TOP + 30, 2 * r1, fill=NAVY)
s.text(L + 160, TOP + 30, 2 * r1, 2 * r1, ["2,761", "fascicle images"], size=44, face=HEAD, color=WHITE, align="c", anchor="m", gap=8)
s.oval(L + 980, TOP + 30 + r1 - r2, 2 * r2, fill=BLUE)
s.text(L + 980, TOP + 30 + r1 - r2, 2 * r2, 2 * r2, ["1,048", "aponeurosis images"], size=36, face=HEAD, color=WHITE, align="c", anchor="m", gap=8)
caption(s, L + 760, TOP + 30 + r1 - 30, 220, "0 overlap", size=34, bold=True, color=NAVY)

s = content("bg-masks", "The labels are hand-drawn masks: white pixels mark the structure on a black image", "There are no measured numbers in the training data, only masks. A mask is a black image the same size as the ultrasound image, with the structure painted white. The aponeurosis masks show the two bands. The fascicle masks show some of the fibres as thin lines.")
pw = 500
muscle(s, L, TOP + 30, pw)
caption(s, L, TOP + 420, pw, "Ultrasound image", size=34, bold=True, color=NAVY, h=50)
muscle(s, L + 558, TOP + 30, pw, fibres=False, bg="000000")
caption(s, L + 558, TOP + 420, pw, "Aponeurosis mask", size=34, bold=True, color=NAVY, h=50)
k2, P2 = muscle(s, L + 1116, TOP + 30, pw, bands=False, fibres=False, bg="000000")
for a, b in [((80, 380), (300, 140)), ((330, 390), (520, 150)), ((470, 360), (640, 170))]:
    s.line(*P2(*a), *P2(*b), WHITE, 2)
caption(s, L + 1116, TOP + 420, pw, "Fascicle mask", size=34, bold=True, color=NAVY, h=50)

s = content("bg-thin", "Fascicle masks cover only 0.3 % of the pixels, ten times less than aponeurosis masks", "Fascicle masks are lines one or two pixels wide, drawn on only some of the fibres. On average they cover 0.3 percent of an image, against about 3 percent for aponeurosis masks. A model that predicts nothing would already be 99.7 percent right on fascicle pixels, which is why the loss function and the mask preparation need care.")
s.bar_chart(L, TOP + 60, 1400, 360, ["Aponeurosis mask", "Fascicle mask"], [3.0, 0.3], fmt='0.0" %"', colors=[BLUE, NAVY], vmax=3.6, size=38, label_fmt=lambda v: f"{v:.1f} %")
caption(s, L, TOP + 480, 1300, "Share of image pixels that are white in the mask (average)", size=32, align="l", color=MUTED)

s = content("bg-shift", "The test images add a new scanner, a new muscle and patients with cerebral palsy", "The test set is unpublished data from new participants. Two of its scanners also appear in training, but the Philips Lumify is new. Rectus femoris is a muscle not seen in training. Some participants have cerebral palsy, while the training data has only healthy people. This makes the test harder than our own validation.")
s.table(L, TOP + 20, 1560, [0.22, 0.39, 0.39], [
    ["", "Training images", "Test images"],
    ["Scanners", "5 models (Siemens, Telemed ×2, Hitachi, Philips HD11)", "Siemens, Telemed + **Philips Lumify (new)**"],
    ["Muscles", "vastus lateralis, gastrocnemius, tibialis anterior", "vastus lateralis, gastrocnemius + **rectus femoris (new)**"],
    ["Participants", "healthy adults, mostly 18–35", "healthy + **cerebral palsy (new)**, 18–28"]], size=34)

s = content("bg-layouts", "Most of the 309 test images are scanner screenshots in five different screen layouts", "We grouped the test images by their screen layout, found from the image content. Siemens and Telemed screenshots make up most of the set. The 58 PNG images come from the new Philips Lumify scanner. The layouts matter because each one places the scan and the ruler differently.")
s.bar_chart(L, TOP + 20, 1450, 520, ["Siemens screenshot", "Telemed screenshot", "Philips Lumify (new)", "Telemed, smaller screen", "Telemed, cropped scan"], [91, 90, 58, 50, 20], colors=[NAVY, NAVY, BLUE, NAVY, NAVY], vmax=110, size=34)
caption(s, L, TOP + 580, 1300, "Number of test images per layout", size=32, align="l", color=MUTED)

s = content("bg-dupes", "Half of the fascicle training images are exact copies of another image", "1,385 of the 2,761 fascicle images are exact duplicates. Neighbouring files are also frames of the same video, so they look almost the same. If copies landed in both the training and the validation set, the validation score would look better than it really is.")
tot = 1520; d = tot * 1385 / 2761
s.rect(L, TOP + 120, d, 150, fill=NAVY); s.rect(L + d, TOP + 120, tot - d, 150, fill=BLUE)
s.text(L, TOP + 120, d, 150, ["1,385 exact duplicates"], size=40, face=HEAD, color=WHITE, align="c", anchor="m")
s.text(L + d, TOP + 120, tot - d, 150, ["1,376 unique"], size=40, face=HEAD, color=WHITE, align="c", anchor="m")
caption(s, L, TOP + 300, 1400, "Fascicle training images (2,761 in total)", size=32, align="l", color=MUTED)
for j in range(8):
    s.rect(L + j * 120, TOP + 420, 100, 74, fill=TINT, line=BLUE, lw=3)
caption(s, L + 980, TOP + 420, 600, "Neighbouring files are frames of one video", size=32, align="l", color=NAVY, h=90)

s = content("bg-split", "We split the data by video clip, so near-copies never end up in both training and validation", "Images were grouped into clips of consecutive frames and duplicates were kept in the same group. Whole groups then went to training (85 percent) or validation (15 percent). That gives 890 and 158 aponeurosis images, and 2,341 and 420 fascicle images.")
for c in range(6):
    x = L + c * 190
    for j in range(3):
        s.rect(x + j * 14, TOP + 40 + j * 14, 130, 96, fill=WHITE, line=NAVY if c < 5 else BLUE, lw=3)
    caption(s, x, TOP + 180, 170, f"clip {c + 1}", size=30, color=MUTED, h=40)
s.arrow(L + 450, TOP + 240, L + 450, TOP + 380, NAVY, 5)
s.arrow(L + 1040, TOP + 240, L + 1260, TOP + 380, BLUE, 5)
s.rect(L, TOP + 400, 900, 200, fill=NAVY, r=18)
s.text(L + 30, TOP + 400, 840, 200, ["Training: 85 %", "890 aponeurosis · 2,341 fascicle"], size=36, color=WHITE, align="c", anchor="m", gap=10)
s.rect(L + 960, TOP + 400, 600, 200, fill=BLUE, r=18)
s.text(L + 990, TOP + 400, 540, 200, ["Validation: 15 %", "158 aponeurosis · 420 fascicle"], size=36, color=WHITE, align="c", anchor="m", gap=10)

s = content("bg-pipeline", "Our pipeline turns each image into three numbers in six steps", "Steps 1 and 2 prepare the images and are applied the same way to training and test images. Step 3 is the deep-learning part. Steps 4 to 6 are ordinary geometry and image reading, with no learning. No step looks at the file name or the order of the files, because the organisers may re-run the code on renamed, shuffled images.")
steps = [("1", "Crop", "the scan"), ("2", "Prepare", "inputs and masks"), ("3", "U-Nets", "draw the structures"), ("4", "Geometry", "measures in pixels"), ("5", "Scale", "pixels to mm"), ("6", "Check", "fill gaps, clip")]
for i, (n, t, sub) in enumerate(steps):
    x = L + i * 272; col = BLUE if i < 2 else (NAVY if i == 2 else "4A6378")
    s.oval(x + 50, TOP + 60, 150, fill=col)
    s.text(x + 50, TOP + 60, 150, 150, n, size=64, face=HEAD, color=WHITE, align="c", anchor="m")
    caption(s, x, TOP + 240, 250, t, size=40, bold=True, color=NAVY, h=56)
    caption(s, x, TOP + 300, 250, sub, size=32, color=MUTED, h=90)
    if i < 5: s.arrow(x + 210, TOP + 135, x + 262, TOP + 135, NAVY, 4, 18)
for (x0, x1, t) in ([(0, 2, "01 Preprocessing"), (2, 3, "02 Model"), (3, 6, "03 Measuring")] if SHORT else [(0, 2, "02 Preprocessing"), (2, 3, "03 Model"), (3, 6, "04 Measuring")]):
    s.line(L + x0 * 272 + 20, TOP + 450, L + x1 * 272 - 40, TOP + 450, NAVY, 3)
    caption(s, L + x0 * 272, TOP + 470, (x1 - x0) * 272 - 20, t, size=32, bold=True, color=NAVY, h=50)

# ================================================================= 02 preprocessing
section(1, "Before training, every image is cropped and resized, the masks are fixed and thickened, and we tested several ways of enhancing the image. During training, the images are changed randomly so the model learns what matters.")

s = content("pre-crop", "Cropping keeps the grey, textured scan and removes the text, ruler and colour bar around it", "Many images are screenshots of the scanner screen. The text, rulers and colour bars would confuse the model and waste resolution. The program finds the longest run of rows and columns that are grey (no colour) and textured (brightness changes locally, like ultrasound speckle) and draws a box around them. The box is stored, so the results can be mapped back to the full screenshot and the ruler is still available for the millimetre scale. We checked the crop by eye on every image layout.")
fx, fy, k = L, TOP + 10, 1.1
s.rect(fx, fy, 820 * k, 540 * k, fill="20242C", r=16)
Q = lambda x, y: (fx + x * k, fy + y * k)
for yy, ww in [(34, 150), (58, 110), (82, 130)]:
    s.rect(*Q(30, yy), ww * k, 12 * k, fill="5B6577", r=4)
for yy, x1 in [(130, 140), (190, 148), (250, 140), (310, 148), (370, 140), (430, 148)]:
    s.line(*Q(x1, yy), *Q(160, yy), "C9D2DE", 3)
for j in range(8):
    g = int(242 - j * (242 - 26) / 7); s.rect(*Q(740, 130 + j * 25), 26 * k, 25 * k, fill=f"{g:02X}{g:02X}{g:02X}")
s.rect(*Q(190, 120), 500 * k, 340 * k, fill="4A5363")
s.line(*Q(190, 160), *Q(690, 150), BAND, 10); s.line(*Q(190, 400), *Q(690, 380), BAND, 12)
for a, b in [((230, 398), (400, 158)), ((350, 394), (520, 155)), ((470, 389), (640, 152))]:
    s.line(*Q(*a), *Q(*b), FIB, 3)
s.rect(*Q(182, 112), 516 * k, 356 * k, line=WHITE, lw=5, dash=True)
rx = fx + 820 * k + 70
for i, (t, sub) in enumerate([("Kept", "the scan: grey, textured pixels (dashed box)"), ("Removed", "settings text, ruler, colour bar")]):
    y = TOP + 60 + i * 250
    s.text(rx, y, 600, 60, t, size=44, face=HEAD, color=NAVY)
    s.text(rx, y + 66, 600, 120, sub, size=36)

s = content("pre-apo-size", "The aponeurosis model sees the whole crop, squeezed to 512 × 512 pixels", "The bands are wide and run across the whole image, so this model needs the full view more than fine detail. A neural network needs a fixed input size, so every crop is resized to a 512 by 512 square, whatever its shape.")
muscle(s, L + 40, TOP + 60, 640)
caption(s, L + 40, TOP + 560, 640, "Cropped scan (any shape)", size=34, bold=True, color=NAVY, h=50)
s.arrow(L + 740, TOP + 300, L + 900, TOP + 300, NAVY, 6)
s.rect(L + 960, TOP + 30, 470, 470, fill=NAVY, r=14)
k3 = 470 / 760
for (a, b, wd) in [((0, 115), (760, 100), 16), ((0, 440), (760, 400), 18)]:
    s.line(L + 960, TOP + 30 + a[1] * 470 / 560, L + 1430, TOP + 30 + b[1] * 470 / 560, BAND, 10)
for a, b in [((60, 420), (240, 120)), ((300, 410), (480, 110)), ((520, 400), (700, 105))]:
    s.line(L + 960 + a[0] * k3, TOP + 30 + a[1] * 470 / 560, L + 960 + b[0] * k3, TOP + 30 + b[1] * 470 / 560, FIB, 4)
caption(s, L + 960, TOP + 560, 470, "512 × 512 model input", size=34, bold=True, color=NAVY, h=50)

s = content("pre-fasc-tiles", "The fascicle model keeps more detail: 512-pixel tiles from an image scaled to 768 pixels high", "Fibres are thin and faint, and shrinking the image would erase them. So the crop is scaled to 768 pixels high, keeping its shape, and the model works on 512 by 512 tiles. During training, 70 percent of the tiles are centred on a labelled fibre, so the model sees enough of the rare thing it must learn. On test images, a 512-pixel window slides across with 50 percent overlap and the overlapping answers are averaged.")
k4, P4 = muscle(s, L, TOP + 20, 800)
th = 560 * k4; ts = th * 512 / 768
s.rect(L + 40, TOP + 20 + (th - ts) / 2, ts, ts, line=WHITE, lw=6)
s.rect(L + 40 + ts / 2, TOP + 20 + (th - ts) / 2 + 8, ts, ts - 16, line="C9D6E0", lw=6, dash=True)
caption(s, L, TOP + 20 + th + 16, 800, "Crop scaled to 768 px high; tiles are 512 × 512", size=32, bold=True, color=NAVY, h=50)
s.text(L + 960, TOP + 60, 640, 200, ["Tile 1 (solid) and tile 2 (dashed) overlap by 50 %"], size=38, color=NAVY)
s.text(L + 960, TOP + 280, 640, 200, ["Where tiles overlap, their answers are averaged"], size=38, color=NAVY)

s = content("pre-mask-size", "Each mask is stretched to the size of its image, so mask and image line up pixel for pixel", "Some masks were saved at a different size and shape from their image. Stretching the mask to the image size lines it up again; we checked this visually. The resize uses nearest-neighbour interpolation, so the mask stays pure black and white. Masks then get the same crop and resize as their image.")
for i, (dx, t) in enumerate([(70, "Before: mask saved at another size"), (0, "After: stretched to the image")]):
    x = L + i * 820
    k5, P5 = muscle(s, x, TOP + 20, 700, fibres=False)
    sc = 0.8 if i == 0 else 1.0
    for (a, b) in [((0, 122), (760, 101)), ((0, 438), (760, 390))]:
        s.line(x + (a[0] * sc + dx) * k5, TOP + 20 + a[1] * sc * k5, x + (b[0] * sc + dx) * k5, TOP + 20 + b[1] * sc * k5, "F2F2F2" if i else "FFFFFF", 4)
    caption(s, x, TOP + 20 + 560 * k5 + 30, 700, t, size=34, bold=True, color=NAVY, h=50)
s.arrow(L + 715, TOP + 270, L + 805, TOP + 270, NAVY, 5, 20)

s = content("pre-thicken", "Fascicle lines are thickened from 1–2 to about 5 pixels, which keeps their direction and makes them easier to learn", "The expert fascicle lines are so thin that a prediction off by one pixel would count as completely wrong. Widening the lines to about 5 pixels (a morphological dilation) makes the target easier to hit without changing the direction of the fibres, and direction is all the measurements need. The aponeurosis masks are left as they are.")
for i, (wd, col, t) in enumerate([(2, WHITE, "Expert mask: 1–2 pixel lines"), (9, BLUE, "Training target: about 5 pixels")]):
    x = L + 120 + i * 760
    s.rect(x, TOP + 20, 560, 480, fill="000000", r=14)
    for a, b in [((60, 420), (280, 90)), ((190, 450), (410, 120)), ((320, 430), (520, 170))]:
        s.line(x + a[0], TOP + 20 + a[1], x + b[0], TOP + 20 + b[1], col, wd)
    caption(s, x, TOP + 530, 560, t, size=34, bold=True, color=NAVY, h=50)
s.arrow(L + 700, TOP + 260, L + 860, TOP + 260, NAVY, 6)

s = content("pre-channels", "We tested filling the three input channels with plain grey, local-contrast and line-highlighted versions of the image", "A model input has three channels, like the red, green and blue of a photo. Ultrasound is grey, so the channels can hold different versions of the same picture. Local contrast (CLAHE) cuts the image into an 8 by 8 grid and stretches each tile's contrast; faint areas get brighter, but so does the grain. Line highlighting uses a ridge filter (Sato) that lights up thin bright lines of 1 to 3 pixels and darkens everything else. We also tried grain removal before channels 2 and 3. These were experiments: we kept only what helped.")
for i, (t, sub) in enumerate([("Plain grey", "the cropped image, unchanged"), ("Local contrast", "CLAHE: contrast stretched per tile"), ("Line highlighting", "Sato ridge filter: thin lines light up")]):
    x = L + i * 548
    if i == 0: muscle(s, x, TOP + 20, 500)
    if i == 1:
        k6, P6 = muscle(s, x, TOP + 20, 500, light=1)
        for g in range(1, 8):
            s.line(x + g * 500 / 8, TOP + 20, x + g * 500 / 8, TOP + 20 + 368, "4A6378", 2)
            s.line(x, TOP + 20 + g * 368 / 8, x + 500, TOP + 20 + g * 368 / 8, "4A6378", 2)
    if i == 2: muscle(s, x, TOP + 20, 500, bg="050B14", bands=True, light=1)
    caption(s, x, TOP + 420, 500, f"{i + 1} · {t}", size=36, bold=True, color=NAVY, h=50)
    caption(s, x, TOP + 476, 500, sub, size=32, color=MUTED, h=90)

s = content("pre-experiment", "Plain grey input gave the best fascicle overlap (Dice 0.579) and the lowest angle error (0.75°)", "Each variant trained a fascicle model for 20 epochs on the same split and was scored on the same validation images. Dice measures overlap with the expert drawing, from 0 to 1, so higher is better. Angle error is the difference in fibre direction from the expert drawing, so lower is better. The differences are small, but the simplest input won on both, and grain removal made the angle worse. A modern network learns its own contrast adjustments, which is likely why hand-made enhancements add little.")
s.table(L, TOP + 40, 1560, [0.14, 0.50, 0.18, 0.18], [
    ["Variant", "Input to the fascicle model", "Dice ↑", "Angle error ↓"],
    ["A", "plain grey in all 3 channels", "0.579", "0.75°"],
    ["B", "grey + local contrast + line highlighting", "0.569", "0.76°"],
    ["C", "B + mild grain removal", "0.573", "0.80°"],
    ["D", "B + strong grain removal", "0.571", "0.83°"]], size=36, highlight=(1,), right=(2, 3))

s = content("pre-augment", "During training, each image is randomly mirrored, rotated, zoomed and re-lit, so the model never sees the same picture twice", "Augmentation makes a model more reliable when training data is limited. Each change is applied to the image and its mask together, so they stay aligned. Mirroring happens half the time, rotation is up to 10 degrees, zoom and shift up to 10 percent, brightness and contrast up to 20 percent, plus a gamma (mid-tone) change. Upside-down flips are never used, because the skin is always at the top of a real scan. At test time each image is also predicted mirrored and the two answers are averaged.")
vars_ = [(dict(), "Original"), (dict(mirror=True), "Mirrored (50 %)"), (dict(rot=10), "Rotated (±10°)"), (dict(zoom=1.18), "Zoomed (±10 %)"), (dict(light=1, bg="3B4F63"), "Brighter (±20 %)")]
for i, (kw, t) in enumerate(vars_):
    x = L + i * 330
    muscle(s, x, TOP + 60, 290, **kw)
    caption(s, x - 10, TOP + 290, 310, t, size=32, bold=i == 0, color=NAVY, h=50)

s = content("pre-narrow", "A narrow-window augmentation imitates the narrow scans of the new Lumify scanner", "The new Philips Lumify scanner in the test set shows a narrower scan than the training scanners. For the aponeurosis model only, 30 percent of training images are replaced by a random slice of 40 to 100 percent of the width, stretched back to full width.")
k7, P7 = muscle(s, L, TOP + 40, 760)
s.rect(L + 260, TOP + 40, 300, 560 * k7, line=WHITE, lw=6, dash=True)
caption(s, L, TOP + 40 + 560 * k7 + 20, 760, "Random slice: 40–100 % of the width", size=34, bold=True, color=NAVY, h=50)
s.arrow(L + 800, TOP + 250, L + 920, TOP + 250, NAVY, 6)
muscle(s, L + 960, TOP + 40, 600, zoom=1.35)
caption(s, L + 960, TOP + 40 + 560 * 600 / 760 + 20, 600, "Stretched back to full width (30 % of images)", size=34, bold=True, color=NAVY, h=90)

# ================================================================= 03 model
section(2, "This part explains what a U-Net is, the exact architecture we used, how it was trained, and which three models make up submission v3.")

s = content("mod-seg", "A segmentation model gives every pixel a probability of belonging to the structure", "Segmentation means answering, for every pixel, the question: is this part of the structure? The model outputs a probability map the same size as the image. Values near 1 mean the model is sure the pixel belongs to the structure. Pixels above 0.5 count as structure.")
muscle(s, L, TOP + 40, 640)
s.arrow(L + 690, TOP + 250, L + 830, TOP + 250, NAVY, 6)
import random
random.seed(3)
gx, gy, cs = L + 880, TOP + 20, 76
for r in range(7):
    for c in range(9):
        on = abs((6 - r) * 1.1 - c * 0.9 + 1.5) < 0.9
        p = 0.9 + random.random() * 0.09 if on else random.random() * 0.12
        g = int(255 * (1 - p)); col = f"{int(14 + (255 - 14) * (1 - p)):02X}{int(42 + (255 - 42) * (1 - p)):02X}{int(71 + (255 - 71) * (1 - p)):02X}"
        s.rect(gx + c * cs, gy + r * cs, cs - 4, cs - 4, fill=col)
        if (r + c) % 3 == 0:
            s.text(gx + c * cs, gy + r * cs, cs - 4, cs - 4, f"{p:.1f}".lstrip("0") if p < 1 else "1", size=27, color=WHITE if p > 0.5 else NAVY, align="c", anchor="m", check=False)
caption(s, gx, gy + 7 * cs + 20, 9 * cs, "Probability per pixel (0 to 1)", size=34, bold=True, color=NAVY, h=50)

def ushape(s, hi_skip=False, labels=True):
    ox, oy = L + 60, TOP + 10
    lv = [(0, 40, 36, 170), (100, 170, 48, 120), (210, 280, 60, 84), (330, 370, 72, 58)]
    sc = 1.45
    def B(x, y, w, h, col): s.rect(ox + x * sc, oy + y * sc * 0.82, w * sc, h * sc * 0.82, fill=col, r=4)
    skipc = NAVY if hi_skip else TINT
    for (x, y, w, h) in lv:
        yy = oy + (y + h / 2) * sc * 0.82
        s.line(ox + (x + w) * sc, yy, ox + (1000 - x - w) * sc * 1.0 - 0, yy, skipc, 6 if hi_skip else 4, dash=True)
    for (x, y, w, h) in lv: B(x, y, w, h, BLUE if not hi_skip else TINT)
    B(470 - 40, 450, 80, 44, NAVY if not hi_skip else TINT)
    for (x, y, w, h) in lv: B(1000 - x - w, y, w, h, "4A6378" if not hi_skip else TINT)
    return ox, oy, sc

s = content("mod-unet", "A U-Net first shrinks the image to recognise what is there, then grows it back to say exactly where", "The left half (the encoder) shrinks the image step by step. Each step sees a wider area and learns more abstract features: edges, then textures, then whole bands. The right half (the decoder) grows this summary back to full size and draws the outline. The U-Net was designed in 2015 for medical images and is still a standard choice for this kind of task.")
ox, oy, sc = ushape(s)
caption(s, ox, oy + 440 * sc * 0.82 + 10, 640, "Encoder: shrink, ask “what?”", size=34, bold=True, color=BLUE, align="l", h=50)
caption(s, ox + 1000 * sc - 640, oy + 440 * sc * 0.82 + 10, 640, "Decoder: grow, ask “where?”", size=34, bold=True, color="4A6378", align="r", h=50)
caption(s, ox, oy - 0, 200, "image in", size=30, color=NAVY, align="l", h=40)
caption(s, ox + 1000 * sc - 200, oy, 200, "mask out", size=30, color=NAVY, align="r", h=40)

s = content("mod-skip", "Skip connections copy fine detail across the U, so thin fibres are not lost while shrinking", "Shrinking the image gives the model a wide view, but it throws away fine detail. The skip connections (dashed lines) pass the detail from each encoder level straight to the decoder level of the same size. Without them, a fibre only two pixels wide would disappear in the shrinking half.")
ox, oy, sc = ushape(s, hi_skip=True)
caption(s, ox + 300 * sc, oy + 20, 400 * sc, "skip connections", size=36, bold=True, color=NAVY, h=50)

# ---- detailed architecture diagram
s = content("mod-arch", "Our U-Net combines a pretrained ResNet34 encoder with a five-block decoder and outputs a one-channel probability map", "This is the exact network, built with the segmentation-models-pytorch library: smp.Unet with a ResNet34 encoder, ImageNet weights, 3 input channels and 1 output class. Sizes are for a 512 by 512 input and read as channels × height × width. The encoder halves the size five times. Each decoder block doubles the size, joins the skip connection from the encoder level of the same size, and applies two 3×3 convolutions with batch normalisation and ReLU. A final 3×3 convolution gives one value per pixel, and a sigmoid turns it into a probability. The network has 24.4 million weights.")
enc = [("Input image", "3 × 512²"), ("Stem: 7×7 conv", "64 × 256²"), ("Layer 1: pool + 3 blocks", "64 × 128²"),
       ("Layer 2: 4 blocks", "128 × 64²"), ("Layer 3: 6 blocks", "256 × 32²"), ("Layer 4: 3 blocks", "512 × 16²")]
dec = [("Block 5 + 3×3 head", "1 × 512²"), ("Decoder block 4", "32 × 256²"), ("Decoder block 3", "64 × 128²"),
       ("Decoder block 2", "128 × 64²"), ("Decoder block 1", "256 × 32²")]
rh, bw, bh = 104, 600, 80
ex, dx0 = L, R - bw
for i in range(6):
    y = TOP - 14 + i * rh; ind = i * 22
    a, asz = enc[i]; c = TINT if i == 0 else BLUE; tc = NAVY if i == 0 else WHITE
    s.rect(ex + ind, y, bw - ind, bh, fill=c, r=10)
    s.text(ex + ind + 18, y, bw - ind - 200, bh, a, size=30, color=tc, bold=True, anchor="m")
    s.text(ex + bw - 190, y, 172, bh, asz, size=30, color=tc, align="r", anchor="m")
    if i < 5:
        s.arrow(ex + ind + 50, y + bh, ex + ind + 50, y + rh, NAVY, 4, 16)
        b, bsz = dec[i]; c2 = TINT if i == 0 else "4A6378"; tc2 = NAVY if i == 0 else WHITE
        s.rect(dx0, y, bw - ind, bh, fill=c2, r=10)
        s.text(dx0 + 18, y, bw - ind - 200, bh, b, size=30, color=tc2, bold=True, anchor="m")
        s.text(dx0 + bw - ind - 190, y, 172, bh, bsz, size=30, color=tc2, align="r", anchor="m")
        if 1 <= i <= 4:
            s.arrow(ex + bw + 14, y + bh / 2, dx0 - 14, y + bh / 2, NAVY, 3, 16, dash=True)
        if i <= 3:
            s.arrow(dx0 + 50, y + rh, dx0 + 50, y + bh, NAVY, 4, 16)
y5 = TOP - 14 + 5 * rh
s.arrow(ex + bw + 14, y5 + bh / 2, dx0 + 40, TOP - 14 + 4 * rh + bh + 4, NAVY, 4, 18)
s.text(ex + bw + 150, y5 + 34, 330, 50, "upsample ×2", size=29, color=NAVY)
s.text(ex + bw + 20, TOP - 14, dx0 - ex - bw - 40, bh, "sigmoid → probability", size=29, color=NAVY, align="c", anchor="m")
s.text(ex + bw + 20, TOP - 14 + 2 * rh - 58, dx0 - ex - bw - 40, 50, "skip connections", size=29, color=NAVY, bold=True, align="c")
lgy = TOP - 14 + 6 * rh + 14
for j, (c, tt) in enumerate([(BLUE, "Encoder: ResNet34, ImageNet weights"), ("4A6378", "Decoder: upsample ×2, join skip, two 3×3 convs")]):
    x = L + j * 620
    s.rect(x, lgy + 8, 34, 34, fill=c, r=4)
    s.text(x + 50, lgy, 760, 50, tt, size=29, color=NAVY)

s = content("mod-pretrained", "The encoder starts from weights learned on 1.2 million everyday photos, so it already detects edges and textures", "The encoder is a ResNet34 initialised with weights trained on ImageNet, a collection of 1.2 million everyday photos. Ultrasound looks nothing like those photos, but the low-level features learned there (edges, lines, textures) transfer well. Starting from them means the model does not have to learn everything from scratch.")
for j in range(6):
    r_, c_ = divmod(j, 3)
    s.rect(L + c_ * 150, TOP + 60 + r_ * 150, 130, 130, fill=[BLUE, TINT, "4A6378", TINT, BLUE, "C9D6E0"][j], r=10)
caption(s, L, TOP + 380, 430, "ImageNet photos", size=34, bold=True, color=NAVY, h=50)
s.arrow(L + 480, TOP + 190, L + 600, TOP + 190, NAVY, 6)
s.rect(L + 630, TOP + 100, 360, 180, fill=NAVY, r=18)
s.text(L + 630, TOP + 100, 360, 180, ["ResNet34", "encoder"], size=40, face=HEAD, color=WHITE, align="c", anchor="m", gap=4)
caption(s, L + 600, TOP + 300, 420, "learns edges, lines, textures", size=32, color=MUTED, h=50)
s.arrow(L + 1030, TOP + 190, L + 1150, TOP + 190, NAVY, 6)
muscle(s, L + 1180, TOP + 60, 420)
caption(s, L + 1180, TOP + 380, 420, "reused on ultrasound", size=34, bold=True, color=NAVY, h=50)

s = content("mod-loss", "The loss mixes Dice and cross-entropy equally, so predicting nothing cannot score well", "Training adjusts the weights to make the loss smaller. Cross-entropy scores each pixel separately. Dice scores the overlap of the whole shape. Fascicles cover only 0.3 percent of pixels, so with cross-entropy alone an all-black prediction would already look 99.7 percent right. Dice gives that prediction the worst possible score, so the mix forces the model to draw the fibres.")
s.text(L, TOP + 30, 1600, 90, "Loss = 0.5 × Dice loss + 0.5 × cross-entropy", size=56, face=HEAD, color=NAVY, align="c")
for i, (t, v, sub) in enumerate([("Cross-entropy alone", "99.7 % right", "for an all-black fascicle prediction"), ("Dice", "0 overlap", "for the same all-black prediction")]):
    x = L + 60 + i * 790
    s.rect(x, TOP + 200, 700, 340, fill=TINT if i == 0 else NAVY, r=20)
    col = NAVY if i == 0 else WHITE
    s.text(x + 40, TOP + 230, 620, 60, t, size=36, color=col, bold=True, align="c")
    s.text(x + 40, TOP + 300, 620, 110, v, size=80, face=HEAD, color=col, align="c")
    s.text(x + 40, TOP + 430, 620, 90, sub, size=32, color=col, align="c")

s = content("mod-train", "Each U-Net trains with AdamW and a learning rate that starts at 0.0001 and falls to zero along a cosine curve", "AdamW is a standard optimiser. The learning rate sets how big each weight update is. It starts at 0.0001 and follows a cosine curve down to zero by the last epoch, so early updates are large and late ones are fine adjustments. Each step uses 8 images, mixed-precision arithmetic speeds things up, and the random seed is fixed at 42. Training ran on one NVIDIA A30 GPU on the university cluster. The curve shown is for the 46-epoch aponeurosis models.")
ep = list(range(0, 47)); lr = [1e-4 * 0.5 * (1 + math.cos(math.pi * e / 46)) for e in ep]
s.line_chart(L, TOP + 10, 1180, 600, ep, lr, size=30)
caption(s, L + 170, TOP + 620, 1000, "Epoch", size=32, color=MUTED, h=46)
for i, (t, v) in enumerate([("Optimiser", "AdamW"), ("Batch", "8 images"), ("Hardware", "1 × A30 GPU")]):
    y = TOP + 40 + i * 190
    s.text(L + 1250, y, 360, 46, t, size=32, color=MUTED)
    s.text(L + 1250, y + 50, 360, 70, v, size=48, face=HEAD, color=NAVY)

s = content("mod-v3", "Submission v3 uses three U-Nets: two for the aponeuroses and one for the fascicles", "Aponeurosis model A uses the plain grey image and model B the enhanced input (grey, local contrast and line highlighting). The fascicle model uses plain grey, the winner of the enhancement test. All three were trained on every labelled image.")
cards = [("Aponeurosis U-Net A", "plain grey input", "1,048 images · 46 epochs", BLUE), ("Aponeurosis U-Net B", "grey + contrast + lines", "1,048 images · 46 epochs", BLUE), ("Fascicle U-Net", "plain grey input", "2,761 images · 19 epochs", NAVY)]
for i, (t, inp, sub, col) in enumerate(cards):
    x = L + i * 548
    s.rect(x, TOP + 40, 500, 460, fill=col, r=20)
    s.text(x + 40, TOP + 80, 420, 120, t, size=44, face=HEAD, color=WHITE, ls=1.1)
    s.text(x + 40, TOP + 240, 420, 60, inp, size=36, color=WHITE)
    s.text(x + 40, TOP + 330, 420, 120, sub, size=36, color=WHITE, bold=True)

s = content("mod-alldata", "v3 repeats the v2 recipe on 100 % of the images, stopping at the epoch counts that v2's validation picked", "An epoch is one full pass through the training images. Too few epochs leave the model undertrained; too many and it starts memorising. The v2 run trained on 85 percent of the images and used the other 15 percent to find the best number of epochs: 46 for the aponeurosis models and 19 for the fascicle model. v3 retrained on everything with those numbers fixed. The cost is that v3 has no held-out images left, so its validation numbers come from v2.")
s.text(L, TOP + 40, 200, 120, "v2", size=72, face=HEAD, color=NAVY, anchor="m")
s.rect(L + 220, TOP + 40, 1100, 120, fill=NAVY); s.rect(L + 1320, TOP + 40, 194, 120, fill=BLUE)
s.text(L + 220, TOP + 40, 1100, 120, "85 % training", size=40, color=WHITE, bold=True, align="c", anchor="m")
s.text(L + 1320, TOP + 40, 194, 120, "15 % val.", size=34, color=WHITE, bold=True, align="c", anchor="m")
s.arrow(L + 1417, TOP + 175, L + 1417, TOP + 270, NAVY, 5)
s.text(L + 900, TOP + 190, 500, 80, "picks 46 and 19 epochs", size=34, color=NAVY, align="r", anchor="m")
s.text(L, TOP + 300, 200, 120, "v3", size=72, face=HEAD, color=NAVY, anchor="m")
s.rect(L + 220, TOP + 300, 1294, 120, fill=NAVY)
s.text(L + 220, TOP + 300, 1294, 120, "100 % training, 46 and 19 epochs", size=40, color=WHITE, bold=True, align="c", anchor="m")

s = content("mod-ensemble", "The two aponeurosis U-Nets are averaged, which gave fewer thickness errors than either model alone", "Each aponeurosis model outputs a probability map. v3 uses the average of the two maps. On the v2 validation images, the averaged map measured muscle thickness better than either model on its own.")
for i, t in enumerate(["U-Net A (grey)", "U-Net B (enhanced)"]):
    y = TOP + 40 + i * 300
    s.rect(L, y, 460, 200, fill=BLUE, r=18)
    s.text(L, y, 460, 200, t, size=40, face=HEAD, color=WHITE, align="c", anchor="m")
    s.arrow(L + 480, y + 100, L + 700, TOP + 290, NAVY, 5)
s.oval(L + 720, TOP + 170, 240, fill=TINT)
s.text(L + 720, TOP + 170, 240, 240, "Average", size=40, face=HEAD, color=NAVY, align="c", anchor="m")
s.arrow(L + 980, TOP + 290, L + 1100, TOP + 290, NAVY, 5)
k8, P8 = muscle(s, L + 1130, TOP + 140, 420, fibres=False, bg="000000")
caption(s, L + 1130, TOP + 140 + 560 * k8 + 20, 420, "Aponeurosis map", size=34, bold=True, color=NAVY, h=50)

s = content("mod-tta", "Every test image is also predicted mirrored, and the two answers are averaged", "This is called test-time augmentation. The model predicts the image and its left-right mirror image. The second answer is flipped back and the two are averaged, which smooths out random errors.")
muscle(s, L, TOP + 20, 400); muscle(s, L, TOP + 360, 400, mirror=True)
caption(s, L + 420, TOP + 130, 240, "original", size=32, color=MUTED, align="l", h=46)
caption(s, L + 420, TOP + 470, 240, "mirrored", size=32, color=MUTED, align="l", h=46)
for y in (TOP + 160, TOP + 500):
    s.arrow(L + 660, y, L + 860, TOP + 410, NAVY, 5)
s.rect(L + 880, TOP + 320, 300, 180, fill=NAVY, r=18)
s.text(L + 880, TOP + 320, 300, 180, ["U-Net, then", "average"], size=38, face=HEAD, color=WHITE, align="c", anchor="m", gap=4)
s.arrow(L + 1200, TOP + 410, L + 1300, TOP + 410, NAVY, 5)
s.text(L + 1320, TOP + 360, 300, 100, "one prediction", size=38, face=HEAD, color=NAVY, anchor="m")

# ================================================================= 04 measurements
section(3, "The models output drawings. This part shows how plain geometry, with no learning, turns those drawings into the three numbers, and how pixels become millimetres.")

s = content("geo-mt", "Muscle thickness is the perpendicular gap between the inner edges of the bands, at 25, 50 and 75 % of the width", "We fit a line to each band. If there are more than two bands (two stacked muscles), we take the pair with the most fibres between them. The gap is measured perpendicular to the bands, inner edge to inner edge, at three places, and averaged, following the experts' protocol.")
k9, P9 = muscle(s, L + 200, TOP + 10, 860)
for f in (0.25, 0.5, 0.75):
    X1, Y1 = P9(760 * f, 122 - 21 * f + 8); X2, Y2 = P9(760 * f, 438 - 48 * f - 9)
    s.arrow(X1, Y2, X1, Y1, WHITE, 5, 20); s.arrow(X1, Y1, X1, Y2, WHITE, 5, 20)
    caption(s, X1 - 70, TOP + 10 + 560 * k9 + 14, 140, f"{int(f * 100)} %", size=34, bold=True, color=NAVY, h=50)

s = content("geo-pa", "Pennation angle is the median angle, against the deep band, of the 7 longest fibre lines", "The fibre drawing is thinned to lines one pixel wide, and a straight line is fitted to each piece. Pieces more than 8 degrees away from the main direction are dropped. The angle of each remaining line against the deep aponeurosis is measured, and the median of the 7 longest is the answer. Using 7 lines instead of 3 makes the result less sensitive to one bad line.")
muscle(s, L + 200, TOP + 10, 860, pa=True)
k10 = 860 / 760
s.text(L + 200 + 300 * k10, TOP + 10 + 330 * k10, 200, 60, "PA", size=40, face=HEAD, color=WHITE, check=False)

s = content("geo-fl", "Fascicle length follows each fibre line until it meets both bands, even beyond the edge of the image", "Each of the 7 fibre lines is extended in a straight line until it meets the line fitted to each band, beyond the picture edge if needed. The distance between the two meeting points is the fascicle length, and the median of the 7 is the answer. This follows the experts' protocol of straight-line extrapolation. Fascicle length has no ground truth in the training data, so it is the least certain of the three values.")
k11, P11 = muscle(s, L + 200, TOP + 60, 760, fibres=True)
s.line(*P11(200, 421), *P11(480, 111), WHITE, 8)
X1, Y1 = P11(480, 111); X2, Y2 = P11(560, 22)
s.line(X1, Y1, X2, Y2 - 60, NAVY, 5, dash=True)
s.oval(*[v - 14 for v in P11(200, 421)], 28, fill=WHITE)
s.oval(X2 - 14, Y2 - 74, 28, fill=NAVY)
caption(s, X2 + 30, Y2 - 110, 560, "meeting point outside the image", size=32, bold=True, color=NAVY, align="l", h=50)

s = content("geo-scale", "Millimetres come from the on-screen ruler: the tick spacing in pixels divided by the tick distance in mm", "No pixel size is given with the images, but most test images are screenshots that show a ruler with tick marks. The program finds evenly spaced ticks beside or below the scan, measures their spacing in pixels, and divides by the tick distance (5 or 10 mm). One screen layout has non-square pixels, so widths and heights get separate scales. On Siemens screenshots the tick distance changes with depth; we inferred it from the probe width and checked it against the printed depth label on 6 images, all within about 2 percent. Angles need no scale.")
rx0, ry0 = L + 160, TOP + 20
s.line(rx0, ry0, rx0, ry0 + 560, NAVY, 5)
for j in range(8):
    yy = ry0 + j * 80
    s.line(rx0, yy, rx0 + (60 if j % 2 == 0 else 34), yy, NAVY, 5)
s.arrow(rx0 + 100, ry0 + 160, rx0 + 100, ry0 + 240, NAVY, 4, 16); s.arrow(rx0 + 100, ry0 + 240, rx0 + 100, ry0 + 160, NAVY, 4, 16)
s.text(rx0 + 130, ry0 + 170, 400, 60, "80 px = 5 mm", size=40, face=HEAD, color=NAVY)
s.text(L + 700, TOP + 120, 900, 300, ["scale = 80 px ÷ 5 mm", "= 16 px per mm"], size=56, face=HEAD, color=NAVY, gap=20)
caption(s, L + 700, TOP + 420, 880, "Example numbers; each image gets its own scale.", size=32, align="l", color=MUTED)

s = content("geo-confidence", "Every test image got a scale from its ruler: 217 with high confidence and 92 with medium confidence", "None of the 309 test images needed a guessed scale. Medium confidence is mainly the Siemens screenshots, where the tick size changes with depth, and some cropped Telemed scans. Thickness and length are reported in millimetres, so a wrong scale would spoil two of the three numbers. The idea of reading the ruler comes from the organisers' own tool, DL_Track_US; here it runs fully automatically.")
s.bar_chart(L, TOP + 60, 1400, 300, ["High confidence", "Medium confidence"], [217, 92], colors=[NAVY, BLUE], vmax=250, size=38)
caption(s, L, TOP + 420, 1300, "Number of test images (309 in total)", size=32, align="l", color=MUTED)

s = content("geo-safety", "Values outside the official ranges are moved to the nearest limit, and failed measurements take the median of the other images", "The official test ranges are 5 to 45 degrees for the angle, 30 to 200 mm for fascicle length and 10 to 50 mm for thickness. If nothing usable is found on an image, the median of the other test images is used. In v3 only 5 or 6 images per value needed this, mostly one short video clip where the deep band was barely visible.")
for i, (t, lo, hi_, unit) in enumerate([("Pennation angle", 5, 45, "°"), ("Fascicle length", 30, 200, " mm"), ("Muscle thickness", 10, 50, " mm")]):
    y = TOP + 40 + i * 190
    s.text(L, y + 10, 420, 60, t, size=38, face=HEAD, color=NAVY)
    s.rect(L + 460, y + 20, 1000, 40, fill=TINT, r=20)
    s.rect(L + 660, y + 20, 600, 40, fill=NAVY, r=20)
    caption(s, L + 560, y + 70, 200, f"{lo}{unit}", size=32, bold=True, color=NAVY, h=46)
    caption(s, L + 1160, y + 70, 200, f"{hi_}{unit}", size=32, bold=True, color=NAVY, h=46)
caption(s, L + 460, TOP + 600, 1000, "Dark bar: allowed range. Values outside are clipped to its ends.", size=32, color=MUTED, h=50)

# ================================================================= 05 results
section(4, "This part shows how well the models did on our own held-out images and on the competition leaderboard, and what the numbers do and do not tell us.")

s = content("res-dice", "On held-out images, predicted masks overlap the expert masks with Dice 0.83 for aponeuroses and 0.58 for fascicles", "Dice measures overlap from 0 (none) to 1 (identical). The fascicle score is lower partly because the experts drew only some of the fibres, so a correct line the experts left out counts as an error. These numbers come from the v2 models on the 15 percent of images they never saw.")
s.bar_chart(L + 100, TOP + 20, 1100, 600, ["Aponeurosis", "Fascicle"], [0.83, 0.58], fmt='0.00', colors=[BLUE, NAVY], horizontal=False, vmax=1.0, size=40, label_fmt=lambda v: f"{v:.2f}", gap=80)

s = content("res-mt", "Thickness error on held-out images averaged 0.55 mm, about half the 1 mm the experts tolerate between themselves", "This compares thickness measured on our predicted aponeurosis masks with thickness measured on the expert masks, in millimetres, on the v2 validation images. The median error is only 0.18 mm; a few images with large errors pull the mean up. The 1 mm reference is the raters' re-check threshold.")
s.bar_chart(L, TOP + 40, 1400, 420, ["Median error", "Mean error", "Experts’ threshold"], [0.18, 0.55, 1.0], fmt='0.00" mm"', colors=[BLUE, NAVY, "C9D6E0"], vmax=1.2, size=38, label_fmt=lambda v: f"{v:.2f} mm")

s = content("res-pa", "Fibre direction differed from the expert drawings by 0.75° on average, far below the experts’ 4° threshold", "This compares the dominant fibre angle of our predicted fascicle masks with that of the expert masks on the v2 validation images. The 4 degree reference is the raters' re-check threshold.")
s.bar_chart(L, TOP + 60, 1400, 300, ["Our angle error", "Experts’ threshold"], [0.75, 4.0], fmt='0.00"°"', colors=[NAVY, "C9D6E0"], vmax=4.8, size=38, label_fmt=lambda v: f"{v:.2f}°")

s = content("res-lb", "Submission v3 scored 0.50942 on the leaderboard, our best result and slightly better than v2", "Lower is better. v3 used the same recipe as v2 but trained on all labelled images, and it improved the score a little. A score of about 0.5 means our average error is about half the tolerance for each value. A later attempt to improve fascicle length by changing only the geometry rules found no reliable gain, so v3 remains our best submission.")
s.bar_chart(L + 100, TOP + 20, 1000, 600, ["v2", "v3"], [0.51394, 0.50942], fmt='0.00000', colors=[BLUE, NAVY], horizontal=False, vmax=0.6, size=40, label_fmt=lambda v: f"{v:.5f}", gap=90)
s.text(L + 1200, TOP + 120, 420, 200, ["Leaderboard score", "lower is better"], size=36, color=NAVY, gap=10)

s = content("res-limits", "The validation numbers are optimistic, so the leaderboard is the real test", "Our validation compares predicted masks with expert masks on training-style images. The test set has a new scanner, a new muscle and new patients, and its reference values are the experts' measurements, not masks. Fascicle length has no ground truth in the training data at all. So the real errors are larger than the validation numbers suggest.")
for i, (t, sub) in enumerate([("Masks, not measurements", "validation compares drawings"), ("Same kind of images", "test adds a new scanner, muscle, patients"), ("No length labels", "fascicle length is never checked directly")]):
    x = L + i * 548
    s.oval(x + 175, TOP + 30, 150, fill=NAVY)
    s.text(x + 175, TOP + 30, 150, 150, "!", size=80, face=HEAD, color=WHITE, align="c", anchor="m")
    caption(s, x, TOP + 220, 500, t, size=38, bold=True, color=NAVY, h=110)
    caption(s, x, TOP + 330, 500, sub, size=32, color=MUTED, h=100)

# ================================================================= appendix
s = content("glossary", "The glossary explains the terms used in this presentation", "Reference for anyone new to ultrasound or deep learning.")
s.table(L, TOP - 20, 1560, [0.24, 0.76], [
    ["Term", "Meaning"],
    ["Aponeurosis", "sheet of connective tissue wrapping the muscle; a bright band"],
    ["Fascicle", "bundle of muscle fibres; a faint slanted line"],
    ["Mask", "black image with the structure painted white"],
    ["Segmentation", "deciding for every pixel whether it belongs to a structure"],
    ["Pretrained", "starting from a network already trained on other images"],
    ["Epoch", "one full pass through all training images"],
    ["Augmentation", "random realistic changes to training images"],
    ["Ensemble", "averaging the answers of several models"],
    ["Dice", "overlap of two drawings, from 0 (none) to 1 (identical)"],
    ["Validation", "labelled images kept out of training to test the model"]], size=30, pad=(20, 8))

s = S("TITLE", "end", "Thank you. Presentation template by Slidesgo.")
s.ph(0, "Thank you", 228, 330, 1238, 220, 120, anchor="b")
s.ph(1, "Questions are welcome · Template: Slidesgo", 228, 580, 1238, 90, 40, face=BODY)

if SHORT:
    VISIBLE = ["cover", "index", "bg-pipeline", "sec-pre", "pre-crop", "pre-fasc-tiles", "pre-experiment",
               "sec-mod", "mod-arch", "mod-v3", "mod-alldata", "sec-geo", "geo-mt", "geo-pa", "geo-fl", "geo-scale",
               "sec-res", "res-mt", "res-lb", "end"]
    HIDDEN = ["sec-bg", "bg-image", "bg-measures", "bg-experts", "bg-score", "bg-counts", "bg-masks", "bg-thin",
              "bg-shift", "bg-layouts", "bg-dupes", "bg-split"]
    order = VISIBLE[:2] + HIDDEN + VISIBLE[2:]
    by_key = {sl.key: sl for sl in SLIDES}
    assert all(k in by_key for k in order), [k for k in order if k not in by_key]
    lst = prs.slides._sldIdLst
    ids = {id(sl.s._element): sid for sl in SLIDES for sid in lst if prs.part.related_part(sid.rId) is sl.s.part}
    for sid in list(lst): lst.remove(sid)
    keep = set(order)
    for sl in SLIDES:
        if sl.key not in keep: prs.part.drop_rel(ids[id(sl.s._element)].rId)
    for k in order:
        lst.append(ids[id(by_key[k].s._element)])
        if k in HIDDEN: by_key[k].s._element.set("show", "0")
    SLIDES[:] = [by_key[k] for k in VISIBLE]
    by_key["bg-pipeline"].s.notes_slide.notes_text_frame.text = (
        "Everyone here works on the same challenge, so we skip the background and go straight to our solution. "
        "Because the training labels are drawn masks rather than numbers, we train models to draw the structures and then measure the drawings. "
        "Steps 1 and 2 prepare the images, step 3 is the deep-learning part, and steps 4 to 6 are geometry and image reading with no learning. "
        "No step looks at file names or file order.")
save(OUT, PREV)
print(len(SLIDES), "slides")
print("\n".join(WARN) or "no warnings")
