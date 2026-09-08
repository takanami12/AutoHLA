"""AutoHLA (public) architecture figure, same visual language as aenet_arch.drawio.png."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch

W, H = 2080, 1580
D = 12  # depth of the 3-D blocks

BLUE = ("#dae8fc", "#6c8ebf")
RED = ("#f8cecc", "#b85450")
GREEN = ("#d5e8d4", "#82b366")
ORANGE = ("#ffe6cc", "#d79b00")
GREY = ("#f5f5f5", "#666666")
PURPLE = ("#e1d5e7", "#9673a6")
TEAL = "#008080"

fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=170)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)


def cube(x, y, w, h, color, dashed=False, d=D, lw=1.1):
    fc, ec = color
    ls = (0, (4, 3)) if dashed else "solid"
    kw = dict(ec=ec, lw=lw, ls=ls, zorder=2)
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fc=fc, **kw))
    ax.add_patch(Polygon([(x, y), (x + d, y - d), (x + w + d, y - d), (x + w, y)],
                         fc=fc, alpha=0.75, **kw))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y - d), (x + w + d, y + h - d),
                          (x + w, y + h)], fc=fc, alpha=0.55, **kw))


def txt(x, y, s, size=11, ha="center", va="top", color="#000000", weight=None, style=None):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, fontweight=weight,
            fontstyle=style, linespacing=1.35, zorder=4)


def arrow(x1, y1, x2, y2, dashed=False, color="#000000", lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=lw, color=color, zorder=3,
                                 linestyle=(0, (4, 3)) if dashed else "solid",
                                 shrinkA=0, shrinkB=0))


def elbow(x1, y1, x2, y2, dashed=False, color="#000000"):
    """Elbow connector: horizontal -> vertical -> horizontal."""
    xm = (x1 + x2) / 2
    ax.plot([x1, xm, xm], [y1, y1, y2], color=color, lw=1.2, zorder=3,
            solid_capstyle="round", linestyle=(0, (4, 3)) if dashed else "solid")
    arrow(xm, y2, x2, y2, dashed, color)


def dashbox(x, y, w, h, label, color="#999999", fs=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
                                fc="none", ec=color, ls=(0, (5, 4)), lw=1.1, zorder=1))
    txt(x + w / 2 + 60, y - 26, label, fs, color=color, weight="bold")


# ---------------- title ----------------
txt(880, 22, "AutoHLA Architecture", 26, ha="left", weight="bold")
txt(880, 62, "S1: backbone + decoder on masked dosage (self-supervised) — skippable with --no-s1\n"
             "S2: curriculum 2-digit -> 4-digit (rare-weighted BCE) + SharedNet + per-gene head\n"
             "Post-hoc: pair + ridge -> beta blend -> MAP-diploid (only when n_train < 4000)",
    11.5, ha="left", color="#666666")

# ---------------- input ----------------
txt(211, 90, "Input", 14, weight="bold", color=TEAL)
cube(176, 120, 70, 480, GREY)
txt(211, 618, "L x 2\n(dosage, missing)", 11)
txt(211, 664, "cM channel & stem removed", 10, color="#999999", style="italic")

# ---------------- encoder 1 ----------------
txt(400, 90, "Encoder-1", 14, weight="bold", color=TEAL)
cube(345, 200, 20, 320, RED, d=8)
cube(380, 180, 55, 360, BLUE)
txt(407, 558, "L/2 x 64\nconv k64 s2 + GN + GELU", 11)

# ---------------- encoder 2 ----------------
txt(560, 90, "Encoder-2", 14, weight="bold", color=TEAL)
cube(505, 265, 18, 200, RED, d=8)
cube(540, 250, 42, 230, BLUE)
txt(561, 498, "z: L/4 x 64\nconv k64 s2 + GN + GELU\n(bottleneck)", 11)

# ---------------- sharednet ----------------
txt(683, 292, "SharedNet", 14, weight="bold", color=TEAL)
cube(660, 320, 45, 80, GREEN)
txt(683, 415, "flatten(z) ->\n1 x 256\nLinear+LN+GELU+Drop", 11)

# ---------------- per-gene heads ----------------
for y, name in ((190, "HLA_Block A\n1x1xn_A"), (270, "HLA_Block B\n1x1xn_B"),
                (420, "HLA_Block G (7 genes)\n1x1xn_G")):
    cube(800, y, 30, 45, GREEN, d=8)
    txt(846, y + 2, name, 11, ha="left")
txt(815, 348, "...", 18, weight="bold")
txt(815, 500, "fc1->bn->fc2->bn->fc3", 10, color="#666666")

# ---------------- sigmoid output ----------------
txt(1035, 242, "sigmoid dosage", 14, weight="bold", color=TEAL)
cube(1020, 280, 30, 200, ORANGE, d=8)
txt(1035, 490, "concat allele dosage,\nevery gene", 11)

# ---------------- post-hoc layer ----------------
dashbox(1300, 176, 410, 414, "Post-hoc layer  (n_train < 4000; disable with --no-posthoc)",
        color="#9673a6", fs=11.5)
cube(1350, 215, 34, 90, PURPLE, d=8)
txt(1367, 318, "PairEnergyHead\nfine-tunes a COPY of the net\nrank 16, UNORDERED pairs\nBCE + 0.2 pair_loss", 10)
cube(1350, 430, 34, 70, PURPLE, d=8)
txt(1367, 512, "Ridge on z (256)\nlambda picked on val\n(z of the BASE trunk)", 10)
cube(1560, 300, 30, 110, PURPLE, d=8)
txt(1576, 424, "blend\nbeta_rare / beta_common\naf_split 0.20", 10)

cube(1800, 300, 30, 110, ORANGE, d=8)
txt(1848, 305, "MAP-diploid\ntop-5, tau\n(HW tau=2 fallback)", 10, ha="left")
txt(1816, 470, "calls.csv\nallele_1, allele_2, posterior", 11, weight="bold")
arrow(1816, 410, 1816, 460)

# ---------------- S1 pretext ----------------
dashbox(520, 660, 390, 200, "S1 only (pretext, never runs in S2)")
ax.plot([561, 561], [655, 700], color="#999999", lw=1.2, ls=(0, (4, 3)), zorder=3)
cube(545, 705, 30, 60, BLUE, dashed=True, d=8)
txt(560, 775, "Decoder-1\nk63 + upsample", 10)
cube(650, 695, 40, 80, BLUE, dashed=True, d=8)
txt(660, 788, "Decoder-2\nk63 + upsample", 10)
cube(770, 685, 60, 100, GREY, dashed=True, d=8)
txt(815, 806, "ReconHead\nL x 3 (BCE at masked sites)", 10)

# ---------------- edges ----------------
arrow(246, 360, 345, 360)
arrow(365, 360, 380, 360)
arrow(435, 360, 505, 360)
arrow(523, 360, 540, 360)
arrow(582, 360, 660, 360)
for ey, ty in ((344, 212), (360, 292), (376, 442)):
    elbow(705, ey, 800, ty)
for sy, ty in ((212, 320), (292, 380), (442, 440)):
    elbow(830, sy, 1020, ty)
arrow(1050, 340, 1350, 275)                        # base dosage -> pair
ax.plot([600, 600, 1300, 1300], [500, 600, 600, 465], color="#9673a6", lw=1.2,
        ls=(0, (4, 3)), zorder=3)                  # z -> ridge, routed underneath
arrow(1300, 465, 1350, 465, dashed=True, color="#9673a6")
arrow(1384, 275, 1560, 335)                        # pair -> blend
arrow(1384, 465, 1560, 385)                        # ridge -> blend
arrow(1590, 355, 1800, 355)                        # blend -> decode
arrow(561, 480, 561, 700, dashed=True, color="#999999")
arrow(575, 735, 650, 735, dashed=True, color="#999999")
arrow(690, 735, 770, 735, dashed=True, color="#999999")

# ---------------- legend ----------------
txt(1150, 690, "Legend", 13, ha="left", weight="bold")
for i, (col, lab, dsh) in enumerate((
        (BLUE, "conv1d + GroupNorm + GELU", False),
        (RED, "stride-2 downsample (x1/4 total)", False),
        (GREEN, "Linear/FC + GELU (head)", False),
        (ORANGE, "sigmoid output / hard-call", False),
        (PURPLE, "post-hoc layer (pair, ridge, blend)", False),
        (GREY, "S1 only (pretext)", True))):
    y = 720 + i * 32
    cube(1150, y, 34, 22, col, dashed=dsh, d=7)
    txt(1200, y + 3, lab, 11.5, ha="left")

txt(30, 890, "Measured and DROPPED from the public build:  cM channel · nonlinear stem · cross-attention decoder ·\n"
             "TF blocks · pair MLP scorer · pair prior · SWA · learned phaser · HLA_DPA1",
    10.5, ha="left", color="#999999", style="italic")


# ================= LEARNED BLOCKS IN DETAIL =================
# Same visual language as the diagram above: 3-D blocks, NO text inside them;
# colours are explained by the dedicated legend below.
ax.plot([0, W], [1010, 1010], color="#cccccc", lw=1)
txt(30, 1024,
    "Learned blocks in detail   (ridge = closed-form solution, blend = 1-2 scalars per gene -> no DL block, skipped)",
    11.5, ha="left", color="#666666", style="italic")


def chain(items, x0, ymid, gap=14):
    """Draw a chain of blocks, captions BELOW. items = (w, h, color, dashed, caption)."""
    x = x0
    prev = None
    for w, h, col, dashed, cap in items:
        y = ymid - h / 2
        cube(x, y, w, h, col, dashed=dashed, d=8)
        txt(x + w / 2, ymid + h / 2 + 12, cap, 9.5)
        if prev is not None:
            arrow(prev, ymid, x - 2, ymid)
        prev = x + w + 8
        x += w + gap
    return x


# ---- Panel A: HLA_Block ----
dashbox(35, 1090, 1090, 300, "HLA_Block  (one per gene, 7 genes)", color="#82b366", fs=12)
chain([
    (26, 150, GREY, False, "z\n1 x 256"),
    (30, 170, GREEN, False, "fc1\n-> 256|384"),
    (16, 170, BLUE, False, "BN"),
    (16, 170, BLUE, False, "ReLU"),
    (16, 170, GREY, True, "Drop\n0.3"),
    (26, 120, GREEN, False, "fc2\n-> 128|256"),
    (16, 120, BLUE, False, "BN"),
    (16, 120, BLUE, False, "ReLU"),
    (16, 120, GREY, True, "Drop\n0.1"),
    (30, 190, GREEN, False, "fc3\n-> n_allele"),
    (26, 190, ORANGE, False, "sigmoid"),
], 70, 1190, gap=60)
txt(580, 1330,
    "256 | 128 for A, C, DQB1     ·     384 | 256 for B, DRB1, DQA1, DPB1     ·     block height ~ layer width",
    10, color="#666666")
txt(580, 1358,
    "independent BCE per allele (rare-weighted, w <= 10 in the 4-digit stage) — NO competing softmax within a gene",
    10, color="#666666")

# ---- Panel B: PairEnergyHead ----
dashbox(1160, 1090, 880, 330, "PairEnergyHead  (unordered pairs, added on top of the AutoNet logits)",
        color="#9673a6", fs=12)
cube(1200, 1120, 26, 110, GREY, d=8);   txt(1213, 1242, "z\n1 x 256", 9.5)
cube(1310, 1140, 22, 70, PURPLE, d=8);  txt(1321, 1222, "context c\n256 -> 16", 9.5)
cube(1415, 1235, 28, 60, PURPLE, d=8);  txt(1429, 1302, "E_g\nn_allele x 16", 9.5)
cube(1540, 1120, 30, 110, PURPLE, d=8); txt(1555, 1242, "interaction\nrank 16", 9.5)
cube(1700, 1130, 26, 90, ORANGE, d=8);  txt(1713, 1232, "softmax\nover all pairs", 9.5)
cube(1840, 1130, 26, 90, ORANGE, d=8);  txt(1853, 1232, "scatter_add\n-> dosage", 9.5)
cube(1200, 1290, 26, 55, ORANGE, d=8);  txt(1213, 1357, "score\nsigmoid", 9.5)
cube(1310, 1290, 22, 55, GREY, d=8);    txt(1321, 1357, "unary u\nlogit(score)", 9.5)
cube(1540, 1280, 30, 75, GREY, d=8);    txt(1555, 1367, "pair logit (i,j)", 9.5)
arrow(1234, 1175, 1310, 1175)
arrow(1340, 1175, 1540, 1175)
arrow(1451, 1250, 1540, 1205)
arrow(1234, 1317, 1310, 1317)
arrow(1340, 1317, 1540, 1317)
arrow(1555, 1238, 1555, 1278)
arrow(1578, 1300, 1700, 1200)
arrow(1734, 1175, 1840, 1175)
txt(1600, 1392,
    "(i,i) = homozygous  ·  +log2 for heterozygous pairs  ·  loss = BCE + 0.2 · pair_loss, the trunk is fine-tuned too",
    9.5, color="#666666")

# ---- callout lines from the main diagram ----
for x0, y0, x1, y1, col in ((830, 465, 580, 1085, "#82b366"),
                            (1384, 305, 1600, 1085, "#9673a6")):
    ax.plot([x0, x1], [y0, y1], color=col, lw=1, ls=(0, (2, 4)), zorder=1, alpha=0.55)

# ---- legend for the detail panels ----
txt(35, 1462, "Detail-block legend", 12, ha="left", weight="bold")
det = [(GREEN, False, "Linear / FC with parameters"),
       (BLUE, False, "BatchNorm1d, ReLU"),
       (GREY, True, "Dropout"),
       (GREY, False, "parameter-free op (logit, pair sum, scatter_add)"),
       (PURPLE, False, "learned pair parameters (context c, embedding E_g, interaction)"),
       (ORANGE, False, "sigmoid / softmax")]
for k, (col, dsh, lab) in enumerate(det):
    xl = 35 + (k % 3) * 660
    yl = 1496 + (k // 3) * 34
    cube(xl, yl, 30, 22, col, dashed=dsh, d=7)
    txt(xl + 42, yl + 3, lab, 10.5, ha="left")

out = str(Path(__file__).resolve().with_suffix(""))
fig.savefig(out + ".png", dpi=170)
fig.savefig(out + ".svg")
print("written", out + ".png")
