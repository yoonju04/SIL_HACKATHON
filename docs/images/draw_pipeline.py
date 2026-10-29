"""Draw docs/images/system_pipeline.png (laya-judge layout) with Pillow, in the original dark style.

  python docs/images/draw_pipeline.py [out.png]
Layout in 772-px design units, rendered at S x for a sharp README image (README shows it at width 640)."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 1326 / 772
W, H = 772, 1150
BG = (21, 21, 21)
ARROW = (137, 135, 129)
LEGEND = (192, 191, 180)
STYLE = {   # fill, border, title, subtitle
    "nvidia": ((60, 52, 137), (117, 110, 186), (232, 229, 255), (196, 191, 245)),
    "judge": ((104, 64, 8), (201, 142, 52), (255, 231, 184), (243, 196, 120)),
    "green": ((8, 80, 65), (50, 141, 115), (196, 240, 224), (92, 201, 164)),
    "dark": ((21, 21, 21), (58, 58, 58), (238, 238, 232), (160, 158, 150)),
}
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def f(path, size):
    return ImageFont.truetype(path, int(size * S), index=1)   # index 1 = KR face of the CJK collection


TITLE, SUB, LEG = f(FONT_B, 14.5), f(FONT_R, 12), f(FONT_R, 12.5)


def P(*xy):
    return tuple(int(round(v * S)) for v in xy)


def box(d, x0, y0, x1, y1, kind, title, sub):
    fill, border, tc, sc = STYLE[kind]
    d.rounded_rectangle(P(x0, y0, x1, y1), radius=int(8 * S), fill=fill, outline=border, width=max(1, int(1.2 * S)))
    cx = (x0 + x1) / 2
    d.text(P(cx, (y0 + y1) / 2 - 10), title, font=TITLE, fill=tc, anchor="mm")
    d.text(P(cx, (y0 + y1) / 2 + 12), sub, font=SUB, fill=sc, anchor="mm")


def line(d, pts, style="solid", color=ARROW, width=1.4):
    pts = [P(*p) for p in pts]
    w = max(1, int(width * S))
    if style == "solid":
        d.line(pts, fill=color, width=w)
        return
    dash, gap = ((7, 5) if style == "dashed" else (2, 4))
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        L = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5
        n, t = max(1, int(L)), 0.0
        while t < L:
            t2 = min(L, t + dash * S)
            d.line([(xa + (xb - xa) * t / L, ya + (yb - ya) * t / L), (xa + (xb - xa) * t2 / L, ya + (yb - ya) * t2 / L)],
                   fill=color, width=w)
            t = t2 + gap * S


def head(d, x, y, direction, color=ARROW):
    s = 5
    pts = {"down": [(x - s, y - s * 1.4), (x, y), (x + s, y - s * 1.4)],
           "left": [(x + s * 1.4, y - s), (x, y), (x + s * 1.4, y + s)],
           "right": [(x - s * 1.4, y - s), (x, y), (x - s * 1.4, y + s)]}[direction]
    d.line([P(*p) for p in pts], fill=color, width=max(1, int(1.4 * S)), joint="curve")


def arrow_down(d, x, y0, y1):
    line(d, [(x, y0), (x, y1)])
    head(d, x, y1, "down")


def main(out: Path) -> None:
    im = Image.new("RGB", P(W, H), BG)
    d = ImageDraw.Draw(im)
    L, R, CX = 171, 566, 368.5
    rows = {}
    y = 19
    for key in ("profile", "hyp", "nemA", "feas", "split", "gp", "orch", "carla", "mon", "cosmos"):
        rows[key] = y
        y += 105
    bh = 64
    box(d, L, rows["profile"], R, rows["profile"] + bh, "dark", "Safety profile", "코드 관측 사실, 근거 포함")
    box(d, L, rows["hyp"], R, rows["hyp"] + bh, "dark", "Hypotheses", "취약 가설 H-01~03, 판정 규칙")
    box(d, L, rows["nemA"], R, rows["nemA"] + bh, "nvidia", "Nemotron A: 시나리오 생성", "가설 근거 인용, 실패 few-shot")
    box(d, L, rows["feas"], R, rows["feas"] + bh, "green", "타당성 검사 + 후보 확장", "G-01~04, 시드당 섭동 10개")
    box(d, 590, rows["feas"], 752, rows["feas"] + bh, "dark", "Scenario space", "ODD × 고장")
    box(d, L, rows["split"], 362, rows["split"] + bh, "green", "결정론적 feature", "DF-01~09 (고장 역학 포함)")
    box(d, 374, rows["split"], R, rows["split"] + bh, "judge", "판정자 ×25 (J1–J9) + 회귀", "Laya 기본 · 결과는 라벨로만")
    box(d, L, rows["gp"], R, rows["gp"] + bh, "green", "GP surrogate", "타깃 safety_margin, 평균·불확실성")
    box(d, L, rows["orch"], R, rows["orch"] + bh, "green", "Orchestrator 선택", "level-set 경계 + 다양성 + 무작위")
    box(d, L, rows["carla"], R, rows["carla"] + bh, "dark", "CARLA 짝 실행", "baseline(캐시) vs fault, Nebius 병렬")
    box(d, L, rows["mon"], R, rows["mon"] + bh, "dark", "Monitor → Error table", "safety_margin, 결과 5분류")
    box(d, 590, rows["mon"], 752, rows["mon"] + bh, "dark", "Safety spec", "P-01~03")
    box(d, L, rows["cosmos"], R, rows["cosmos"] + bh, "nvidia", "Cosmos 영상 해설", "검증 통과 해설만 피드백, 판정 불변")
    for a, b in (("profile", "hyp"), ("hyp", "nemA"), ("nemA", "feas"), ("gp", "orch"), ("orch", "carla"),
                 ("carla", "mon"), ("mon", "cosmos")):
        arrow_down(d, CX, rows[a] + bh, rows[b] - 2)
    # split into DF / judges and back into GP
    yb = rows["feas"] + bh + 18
    line(d, [(CX, rows["feas"] + bh), (CX, yb)])
    line(d, [(266, yb), (470, yb)])
    arrow_down(d, 266, yb, rows["split"] - 2)
    arrow_down(d, 470, yb, rows["split"] - 2)
    arrow_down(d, 266, rows["split"] + bh, rows["gp"] - 2)
    arrow_down(d, 470, rows["split"] + bh, rows["gp"] - 2)
    # side inputs
    line(d, [(590, rows["feas"] + bh / 2), (568, rows["feas"] + bh / 2)]); head(d, 568, rows["feas"] + bh / 2, "left")
    line(d, [(590, rows["mon"] + bh / 2), (568, rows["mon"] + bh / 2)]); head(d, 568, rows["mon"] + bh / 2, "left")
    # feedback: GP retrain (solid), failure few-shot + accepted explanations (dashed), hypothesis update (dotted)
    ym = rows["mon"] + bh / 2
    line(d, [(L, ym - 12), (123, ym - 12), (123, rows["gp"] + bh / 2), (169, rows["gp"] + bh / 2)])
    head(d, 169, rows["gp"] + bh / 2, "right")
    yc = rows["cosmos"] + bh / 2
    line(d, [(L, yc), (89, yc), (89, rows["nemA"] + bh / 2), (166, rows["nemA"] + bh / 2)], "dashed")
    head(d, 168, rows["nemA"] + bh / 2, "right")
    line(d, [(L, ym + 18), (53, ym + 18), (53, rows["hyp"] + bh / 2), (166, rows["hyp"] + bh / 2)], "dotted")
    head(d, 168, rows["hyp"] + bh / 2, "right")
    # legend
    ly = rows["cosmos"] + bh + 34
    x = 19
    for kind, label in (("nvidia", "NVIDIA (Nemotron·Cosmos)"), ("judge", "판정자 (Laya·Jev·Nemotron)"), ("green", "결정론·학습"),
                        ("dark", "파일·시뮬")):
        fill, border, _, _ = STYLE[kind]
        d.rounded_rectangle(P(x, ly - 8, x + 16, ly + 8), radius=int(3 * S), fill=fill, outline=border, width=max(1, int(S)))
        d.text(P(x + 24, ly), label, font=LEG, fill=LEGEND, anchor="lm")
        x += 24 + d.textlength(label, font=LEG) / S + 28
    ly += 32
    x = 19
    for style, label in (("solid", "GP 재학습"), ("dashed", "실패 few-shot · 해설"), ("dotted", "가설 상태 갱신")):
        line(d, [(x, ly), (x + 30, ly)], style)
        d.text(P(x + 40, ly), label, font=LEG, fill=LEGEND, anchor="lm")
        x += 40 + d.textlength(label, font=LEG) / S + 34
    im.save(out, optimize=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("system_pipeline.png"))
