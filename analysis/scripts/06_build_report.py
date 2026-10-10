"""Step 6 - Inline the data bundle and Leaflet CSS into the report page.

Writes two copies:
  * index.html (repo root)                      - full HTML document for GitHub Pages, with download links
  * analysis/outputs/MaMya_DamBreak_Report.html - fragment for the claude.ai Artifact (host adds the skeleton)
"""
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
ROOT = ANALYSIS.parent
WEB = ANALYSIS / "web"
OUT = ANALYSIS / "outputs"

html = (WEB / "template.html").read_text(encoding="utf-8")
css = (WEB / "leaflet.css").read_text(encoding="utf-8")
bundle = (WEB / "bundle.json").read_text(encoding="utf-8").replace("</", "<\\/")
html = html.replace("/*__LEAFLET_CSS__*/", css).replace("__BUNDLE__", bundle)

# Artifact fragment: no downloads (the artifact host serves only the page itself)
(OUT / "MaMya_DamBreak_Report.html").write_text(html.replace("<!--DOWNLOADS-->", ""), encoding="utf-8")

# GitHub Pages: full document, head/body split at the masthead, relative download links
DOWNLOADS = (
    '<div class="wrap" style="padding-block:0 8px"><p class="mapnote">Downloads: '
    '<a href="analysis/outputs/kml/MaMya_DamBreak_Assessment.kmz">Google Earth KMZ (all layers)</a> · '
    '<a href="analysis/outputs/kml/MaMya_revised_flood_path.kml">Revised flood path KML</a> · '
    '<a href="analysis/outputs/villages_at_risk.csv">Villages at risk (CSV)</a> · '
    '<a href="analysis/outputs/village_tract_summary.csv">Village tract summary (CSV)</a> · '
    '<a href="analysis/outputs/figures/map_flood_S1.png">S1 map (PNG)</a> · '
    '<a href="https://github.com/geonet-myanmar/mamya-dam-breach-risk-analysis">Source code and data on GitHub</a>'
    '</p></div>'
)
page = html.replace("<!--DOWNLOADS-->", DOWNLOADS)
split = page.index('<header class="mast">')
head, body = page[:split], page[split:]
doc = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    '<meta name="description" content="Rapid dam-break assessment for Ma Mya Reservoir, Myanaung, Myanmar: '
    'revised flood paths, flood depth and arrival time, and village evacuation priorities.">\n'
    '<style>[hidden]{display:none!important}img{max-width:100%}</style>\n'
    f'{head}\n</head>\n<body>\n{body}\n</body>\n</html>\n'
)
(ROOT / "index.html").write_text(doc, encoding="utf-8")
print(f"index.html {len(doc)/1e6:.2f} MB · artifact fragment {len(html)/1e6:.2f} MB")
