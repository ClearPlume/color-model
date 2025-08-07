from colorsys import rgb_to_hsv, rgb_to_hls

from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor


def lab_to_all(l: float, a: float, b: float) -> tuple[list[int], str, list[int], list[int], list[int]]:
    """
    将 CIELAB 色彩值转换为常见颜色空间表示：
    :param l: L 分量（0-100）
    :param a: a 分量（-128 ~ 127）
    :param b: b 分量（-128 ~ 127）
    :return: (RGB [int], HEX str, HSV [int], HSL [int], LAB [int])
    """
    # CIELAB → sRGB（线性空间）
    lab = LabColor(l, a, b)
    rgb: sRGBColor = convert_color(lab, sRGBColor)

    # 限定 RGB 范围（部分 Lab 会映射出界）
    r = min(max(rgb.clamped_rgb_r, 0.0), 1.0)
    g = min(max(rgb.clamped_rgb_g, 0.0), 1.0)
    b_ = min(max(rgb.clamped_rgb_b, 0.0), 1.0)

    # RGB（0~255）
    rgb_255 = [round(r * 255), round(g * 255), round(b_ * 255)]

    # HEX
    hex_code = '#{:02X}{:02X}{:02X}'.format(*rgb_255)

    # HSV（H: 0-360, S/V: 0-100）
    h, s, v = rgb_to_hsv(r, g, b_)
    hsv = [round(h * 360), round(s * 100), round(v * 100)]

    # HSL（H: 0-360, S/L: 0-100）
    h, l_, s_ = rgb_to_hls(r, g, b_)
    hsl = [round(h * 360), round(s_ * 100), round(l_ * 100)]

    # LAB（整数化）
    lab_int = [round(l), round(a), round(b)]

    return rgb_255, hex_code, hsv, hsl, lab_int


def lab_to_hex(l, a, b) -> str:
    lab_color = LabColor(l, a, b)
    rgb_color = convert_color(lab_color, sRGBColor, target_illuminant='d65')

    rgb_clipped = [
        max(0, min(255, round(rgb_color.clamped_rgb_r * 255))),
        max(0, min(255, round(rgb_color.clamped_rgb_g * 255))),
        max(0, min(255, round(rgb_color.clamped_rgb_b * 255)))
    ]
    return "#{:02X}{:02X}{:02X}".format(*rgb_clipped)
