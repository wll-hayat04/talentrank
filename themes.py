"""Palettes de l'interface. Chaque thème fournit les variables CSS et les couleurs des graphiques."""

BASE = dict(on_accent="#05201B", on_hi="#1B1405", heat_text="#06100E")

THEMES = {
    "Turquoise (actuel)": dict(
        canvas="#0C1016", sheet="#141A22", inset="#1A222C", raised="#212B37", line="#242E3B", line2="#364355", side="#10151C",
        ink="#E7ECF3", ink2="#9AA6B6", accent="#2FC4AE", accent2="#4DD6C2", on_accent="#05201B", hi="#E9A83A", on_hi="#1B1405",
        track="#263140", bar="#2B7F76", seq=("#2E8A80", "#8CF2E0"), div=("#E5614D", "#1E2733", "#2FC4AE"),
        series=["#26A894", "#C6851F", "#6585E8", "#E5614D", "#9C70DC", "#6FA83A", "#D86892", "#3A9FD2"]),
    "A · Bleu nuit et or": dict(
        canvas="#0B1220", sheet="#121B2E", inset="#182339", raised="#1F2C46", line="#1F2B45", line2="#33425F", side="#0E1628",
        ink="#E6ECF7", ink2="#9FADC4", accent="#5B9DFF", accent2="#86B8FF", on_accent="#04142B", hi="#F5B942", on_hi="#241803",
        track="#1E2A44", bar="#3A6BB8", seq=("#3A78C8", "#A9CEFF"), div=("#E5614D", "#18233A", "#5B9DFF"),
        series=["#4F8EF0", "#D9A22E", "#26A894", "#E5614D", "#9C70DC", "#6FA83A", "#D86892", "#3A9FD2"]),
    "B · Violet nuit et pêche": dict(
        canvas="#14101F", sheet="#1B1630", inset="#231D3B", raised="#2C2548", line="#2D2548", line2="#433866", side="#110D1B",
        ink="#ECE8F7", ink2="#AAA2C4", accent="#A18CFF", accent2="#BBAAFF", on_accent="#150C33", hi="#FF9F68", on_hi="#2A1202",
        track="#2A2240", bar="#7664C4", seq=("#7E6AD0", "#CFC3FF"), div=("#E5614D", "#231D3B", "#A18CFF"),
        series=["#8E78F0", "#F08A55", "#26A894", "#E5614D", "#6585E8", "#6FA83A", "#D86892", "#3A9FD2"]),
    "C · Vert forêt et sable": dict(
        canvas="#0D1512", sheet="#131E19", inset="#19281F", raised="#213329", line="#20312A", line2="#2F4A3E", side="#0A110E",
        ink="#E5F0EA", ink2="#9DB5A9", accent="#5FD0A0", accent2="#82E0B9", on_accent="#052015", hi="#E8C27A", on_hi="#2A1D05",
        track="#1B2B24", bar="#3B8F70", seq=("#3B9A78", "#A6EBCB"), div=("#E5614D", "#19281F", "#5FD0A0"),
        series=["#3FB88A", "#D1A850", "#6585E8", "#E5614D", "#9C70DC", "#3A9FD2", "#D86892", "#8FA83A"]),
    "D · Graphite et rouge brique": dict(
        canvas="#141414", sheet="#1B1B1B", inset="#232323", raised="#2C2C2C", line="#2A2A2A", line2="#444444", side="#101010",
        ink="#EDEDED", ink2="#A3A3A3", accent="#E4572E", accent2="#F27A55", on_accent="#1A0A04", hi="#EDEDED", on_hi="#141414",
        track="#2A2A2A", bar="#8A8A8A", seq=("#7A7A7A", "#E8E8E8"), div=("#E4572E", "#232323", "#EDEDED"),
        series=["#E4572E", "#9A9A9A", "#6585E8", "#26A894", "#C6851F", "#9C70DC", "#D86892", "#3A9FD2"]),
}
THEMES["Projection (clair, fort contraste)"] = dict(
    canvas="#ECEAF4", sheet="#FFFFFF", inset="#F5F3FB", raised="#E9E5F7", line="#C9C3DE", line2="#8F86AE", side="#FFFFFF",
    ink="#0E0B1A", ink2="#2F2A45", accent="#4B2BC9", accent2="#3A1FA8", on_accent="#FFFFFF", hi="#1E7B4F", on_hi="#FFFFFF",
    good="#1E7B4F", good_tint="#E3F3EA", bad="#B5462E", bad_tint="#F8E6E1",
    track="#DDD7F0", bar="#7A63E0", seq=("#F1EEFF", "#9B86F2"), div=("#E08A74", "#FFFFFF", "#6FBF94"),
    series=["#4B2BC9", "#B5462E", "#1E7B4F", "#B8860B", "#1565C0", "#C2185B", "#5D7A12", "#6D4C41"],
    light=True)
DEFAULT = "Projection (clair, fort contraste)"


def _rgb(h):
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))


def css_vars(t):
    return (":root{" f"--canvas:{t['canvas']};--sheet:{t['sheet']};--inset:{t['inset']};--raised:{t['raised']};"
            f"--line:{t['line']};--line-strong:{t['line2']};--side:{t['side']};--ink:{t['ink']};--ink-2:{t['ink2']};"
            f"--accent:{t['accent']};--accent-dark:{t['accent2']};--on-accent:{t['on_accent']};--saffron:{t['hi']};"
            f"--on-hi:{t['on_hi']};--track:{t['track']};--bar:{t['bar']};--glow:{_rgb(t['accent'])};"
            f"--accent-tint:rgba({_rgb(t['accent'])},.12);"
            f"--good:{t.get('good', '#4CC38A')};--good-tint:{t.get('good_tint', 'rgba(76,195,138,.14)')};"
            f"--bad:{t.get('bad', '#F0695A')};--bad-tint:{t.get('bad_tint', 'rgba(240,105,90,.14)')};" "}")
