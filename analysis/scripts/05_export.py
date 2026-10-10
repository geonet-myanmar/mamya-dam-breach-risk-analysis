"""Step 5 - Deliverables: Google Earth KML, web-map overlays + data bundle, static figures."""
import base64
import io
import json
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis" / "data"
OUT = ROOT / "analysis" / "outputs"
WEB = ROOT / "analysis" / "web"
GPKG = DATA / "work.gpkg"
UTM = 32646

PRIO_HEX = {1: "#d03b3b", 2: "#ec835a", 3: "#fab219", 4: "#7d8a99", 0: "#9aa3ad"}
PRIO_TXT = {1: "P1 Evacuate immediately on breach warning", 2: "P2 Evacuate, wave in 3-12 h",
            3: "P3 Prepare, wave >12 h or worst case only", 4: "Alert, within 1 km of worst-case extent"}
DEPTH_BINS = [0.05, 0.5, 1.5, 3.0]
DEPTH_HEX = ["#b7d3f6", "#6da7ec", "#2a78d6", "#104281"]
ARR_BINS = [1, 3, 6, 12, 24]
ARR_HEX = ["#7f2704", "#b93d0b", "#e2600f", "#f6913f", "#fbbd84", "#fde1c4"]
CROP = (94.98, 17.75, 95.40, 18.16)  # lon/lat window of the flood area for overlays


def kml_color(hexstr, alpha="ff"):
    h = hexstr.lstrip("#")
    return f"{alpha}{h[4:6]}{h[2:4]}{h[0:2]}"


def coords(geom):
    return " ".join(f"{x:.6f},{y:.6f},0" for x, y in geom.coords)


def poly_kml(g):
    parts = [g] if g.geom_type == "Polygon" else list(g.geoms)
    out = []
    for p in parts:
        inner = "".join(f"<innerBoundaryIs><LinearRing><coordinates>{coords(r)}</coordinates></LinearRing></innerBoundaryIs>"
                        for r in p.interiors)
        out.append(f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords(p.exterior)}</coordinates>"
                   f"</LinearRing></outerBoundaryIs>{inner}</Polygon>")
    return "<MultiGeometry>" + "".join(out) + "</MultiGeometry>"


def write_kml():
    paths = gpd.read_file(GPKG, layer="path_revised").to_crs(UTM)
    paths["geometry"] = paths.geometry.simplify(20)
    paths = paths.to_crs(4326)
    user = gpd.read_file(GPKG, layer="path_user")
    st = gpd.read_file(GPKG, layer="routing_stations")
    vil = gpd.read_file(GPKG, layer="villages_exposure")
    fp = gpd.read_file(GPKG, layer="flood_depth_classes")
    ext = gpd.read_file(GPKG, layer="flood_extent")
    fac = gpd.read_file(GPKG, layer="facilities_at_risk")
    dam = gpd.read_file(GPKG, layer="dam")

    styles = []
    for k, c in PRIO_HEX.items():
        styles.append(f'<Style id="p{k}"><IconStyle><color>{kml_color(c)}</color><scale>{1.1 if k == 1 else 0.9}</scale>'
                      f'<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle>'
                      f'<LabelStyle><scale>{0.8 if k in (1, 2) else 0}</scale></LabelStyle></Style>')
    for i, c in enumerate(DEPTH_HEX, start=1):
        styles.append(f'<Style id="d{i}"><LineStyle><width>0</width></LineStyle><PolyStyle><color>{kml_color(c, "b3")}</color>'
                      f'<outline>0</outline></PolyStyle></Style>')
    styles += [
        '<Style id="pathA"><LineStyle><color>ff2b2bd0</color><width>4</width></LineStyle></Style>',
        '<Style id="pathB"><LineStyle><color>ff0080ff</color><width>3</width></LineStyle></Style>',
        '<Style id="user"><LineStyle><color>ff2dc0fb</color><width>2</width></LineStyle></Style>',
        '<Style id="ext"><LineStyle><color>ff6b3610</color><width>2</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>',
        '<Style id="km"><IconStyle><scale>0.6</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/donut.png</href>'
        '</Icon></IconStyle><LabelStyle><scale>0.7</scale></LabelStyle></Style>',
        '<Style id="fac"><IconStyle><scale>0.8</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/schools.png</href>'
        '</Icon></IconStyle></Style>',
        '<Style id="dam"><IconStyle><scale>1.4</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/caution.png</href>'
        '</Icon></IconStyle></Style>',
    ]

    def path_folder():
        pm = [f'<Placemark><name>Ma Mya Dam (မမြဆည်)</name><styleUrl>#dam</styleUrl><description>Storage at FSL 86.38 million m3 '
              f'(ODM/DoI). Embankment subsidence reported 10 Oct 2026 07:30; spillway release from 09:00.</description>'
              f'<Point><coordinates>{dam.geometry.iloc[0].x},{dam.geometry.iloc[0].y},0</coordinates></Point></Placemark>']
        for _, r in paths.iterrows():
            pm.append(f'<Placemark><name>{escape(r["name"])}</name><styleUrl>#path{r.branch}</styleUrl>'
                      f'<description>{escape(r.method)}; length {r.length_km:.1f} km</description>'
                      f'<LineString><tessellate>1</tessellate><coordinates>{coords(r.geometry)}</coordinates></LineString></Placemark>')
        pm.append(f'<Placemark><name>Original projected path (ယူဆ ရေလမ်းကြောင်း)</name><styleUrl>#user</styleUrl>'
                  f'<LineString><tessellate>1</tessellate><coordinates>{coords(user.geometry.iloc[0])}</coordinates></LineString></Placemark>')
        km = [f'<Placemark><name>{r.branch} {r.chain_m/1000:.0f} km · {r.S1_arrival_h:.1f} h</name><styleUrl>#km</styleUrl>'
              f'<description>Branch {r.branch}, {r.chain_m/1000:.0f} km below dam. Best-estimate breach (S1): wave arrives '
              f'{r.S1_arrival_h:.1f} h, peak {r.S1_peak_h:.1f} h after breach starts; peak flow {r.S1_Qpeak:,.0f} m3/s; '
              f'water rises {r.S1_rise_m:.1f} m.</description><Point><coordinates>{r.geometry.x},{r.geometry.y},0</coordinates>'
              f'</Point></Placemark>' for _, r in st[(st.chain_m % 5000) == 0].iterrows()]
        return ('<Folder><name>Flood path (revised)</name>' + "".join(pm) +
                '<Folder><name>Arrival time markers (every 5 km)</name>' + "".join(km) + '</Folder></Folder>')

    def flood_folder():
        out = []
        for k, lab in [("S1", "S1 Full breach, best estimate"), ("S2", "S2 Partial breach"), ("S3", "S3 Upper envelope")]:
            pm = []
            sub = fp[fp.scenario == k]
            for _, r in sub.iterrows():
                pm.append(f'<Placemark><name>Depth {escape(r.depth_label)}</name><styleUrl>#d{r.depth_class}</styleUrl>'
                          f'{poly_kml(r.geometry)}</Placemark>')
            vis = "1" if k == "S1" else "0"
            out.append(f'<Folder><name>{lab}: depth</name><visibility>{vis}</visibility>' + "".join(pm) + '</Folder>')
            e = ext[ext.scenario == k].geometry.iloc[0]
            out.append(f'<Placemark><name>{lab}: outline</name><visibility>{vis}</visibility><styleUrl>#ext</styleUrl>'
                       f'{poly_kml(e)}</Placemark>')
        return '<Folder><name>Modelled flood extent</name>' + "".join(out) + '</Folder>'

    def village_folder():
        out = []
        for p in [1, 2, 3, 4]:
            sub = vil[vil.priority == p].sort_values("S1_arrival_h")
            pm = []
            for _, r in sub.iterrows():
                arr = "not reached" if pd.isna(r.S1_arrival_h) else f"{r.S1_arrival_h:.1f} h"
                desc = (f"<b>{escape(str(r.village_mm))}</b> ({escape(str(r.village))})<br/>Village tract: {escape(str(r.vt))}"
                        f"<br/>Township: {escape(str(r.ts))}<br/>Pcode: {r.vlg_pcode}<br/><br/>{PRIO_TXT[p]}"
                        f"<br/>S1 depth: {r.S1_depth_m:.1f} m, wave arrival: {arr}<br/>S2 depth: {r.S2_depth_m:.1f} m"
                        f"<br/>S3 depth: {r.S3_depth_m:.1f} m<br/>Est. population (WorldPop 2020): {r.pop_est_worldpop:,.0f}"
                        f"<br/>{r.dist_from_dam_km:.1f} km from dam")
                pm.append(f'<Placemark><name>{escape(str(r.village_mm))}</name><styleUrl>#p{p}</styleUrl>'
                          f'<description><![CDATA[{desc}]]></description>'
                          f'<Point><coordinates>{r.geometry.x:.6f},{r.geometry.y:.6f},0</coordinates></Point></Placemark>')
            out.append(f'<Folder><name>{PRIO_TXT[p]} ({len(sub)})</name>' + "".join(pm) + '</Folder>')
        fpm = [f'<Placemark><name>{escape(str(r["name"]))}</name><styleUrl>#fac</styleUrl><description>{r.facility_type}; '
               f'S1 depth {r.S1_depth_m:.1f} m</description><Point><coordinates>{r.geometry.x},{r.geometry.y},0</coordinates>'
               f'</Point></Placemark>' for _, r in fac.iterrows()]
        out.append('<Folder><name>Schools, health facility, bridges in flood zone</name><visibility>0</visibility>'
                   + "".join(fpm) + '</Folder>')
        return '<Folder><name>Villages at risk</name>' + "".join(out) + '</Folder>'

    head = ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            '<name>{}</name><description>Rapid desk assessment, 10 Oct 2026. Model results, not an official warning.'
            '</description>' + "".join(styles))
    full = head.format("Ma Mya Dam break assessment") + path_folder() + flood_folder() + village_folder() + "</Document></kml>"
    (OUT / "kml" / "MaMya_DamBreak_Assessment.kml").write_text(full, encoding="utf-8")
    import zipfile
    with zipfile.ZipFile(OUT / "kml" / "MaMya_DamBreak_Assessment.kmz", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", full.encode("utf-8"))
    pth = head.format("Ma Mya revised flood path") + path_folder() + "</Document></kml>"
    (OUT / "kml" / "MaMya_revised_flood_path.kml").write_text(pth, encoding="utf-8")
    print("KML written")


# ------------------------------------------------------------------------------------------------
def to_mercator(src_path, bounds_ll, res=30, resampling=Resampling.nearest):
    with rasterio.open(src_path) as src:
        from rasterio.warp import transform_bounds
        b = transform_bounds("EPSG:4326", "EPSG:3857", *bounds_ll)
        w = int((b[2] - b[0]) / res); h = int((b[3] - b[1]) / res)
        tr = rasterio.transform.from_bounds(*b, w, h)
        out = np.full((h, w), np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), out, dst_transform=tr, dst_crs="EPSG:3857", resampling=resampling,
                  src_nodata=src.nodata, dst_nodata=np.nan)
    return out


def png_b64(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def classify_rgba(arr, bins, hexes, alpha=210):
    rgba = np.zeros(arr.shape + (4,), dtype="uint8")
    ok = ~np.isnan(arr)
    idx = np.digitize(np.where(ok, arr, -1), bins)
    for i, hx in enumerate(hexes):
        m = ok & (idx == i + (1 if bins is DEPTH_BINS else 0))
        rgba[m, :3] = [int(hx[j:j + 2], 16) for j in (1, 3, 5)]
        rgba[m, 3] = alpha
    return rgba


def web_bundle():
    bounds_full = (94.85, 17.75, 95.60, 18.45)
    # hillshade (whole window, 60 m) - grey with alpha so it works on light and dark grounds
    dem = to_mercator(DATA / "dem_utm.tif", bounds_full, res=60, resampling=Resampling.bilinear)
    dem = np.where(dem < -1000, np.nan, dem)
    z = np.nan_to_num(dem, nan=np.nanmin(dem))
    gy, gx = np.gradient(z * 2.0, 60)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs = np.clip(hs, 0, 1)
    rgba = np.zeros(dem.shape + (4,), dtype="uint8")
    shade = (1 - hs)
    rgba[..., :3] = 20
    rgba[..., 3] = (np.clip(shade * 1.6 - 0.35, 0, 1) * 200).astype("uint8")
    # elevation tint: highlands slightly darker so the Bago-Rakhine foothills read
    tint = np.clip((z - 20) / 200, 0, 1)
    rgba[..., 3] = np.maximum(rgba[..., 3], (tint * 90).astype("uint8"))
    rgba[np.isnan(dem)] = 0
    layers = {"hillshade": {"img": png_b64(rgba), "bounds": [[bounds_full[1], bounds_full[0]], [bounds_full[3], bounds_full[2]]]}}

    for k in ["S1", "S2", "S3"]:
        d = to_mercator(DATA / f"depth_{k}_env.tif", CROP)
        d = np.where(d > 0.05, d, np.nan)
        layers[f"depth_{k}"] = {"img": png_b64(classify_rgba(d, DEPTH_BINS, DEPTH_HEX)),
                                "bounds": [[CROP[1], CROP[0]], [CROP[3], CROP[2]]]}
        a = np.fmin(to_mercator(DATA / f"arrival_{k}_A.tif", CROP), to_mercator(DATA / f"arrival_{k}_B.tif", CROP))
        a = np.where(np.isnan(d), np.nan, a)
        layers[f"arrival_{k}"] = {"img": png_b64(classify_rgba(a, ARR_BINS, ARR_HEX)),
                                  "bounds": [[CROP[1], CROP[0]], [CROP[3], CROP[2]]]}

    def gj(gdf, cols, simplify=None, prec=5):
        g = gdf.copy()
        if simplify:
            g = g.to_crs(UTM)
            g["geometry"] = g.geometry.simplify(simplify)
            g = g.to_crs(4326)
        g = g[cols + ["geometry"]]
        js = json.loads(g.to_json(drop_id=True))
        def rnd(c):
            if isinstance(c, (list, tuple)):
                return [rnd(x) for x in c]
            return round(c, prec) if isinstance(c, float) else c
        for f in js["features"]:
            f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
            for kk, v in list(f["properties"].items()):
                if isinstance(v, float):
                    f["properties"][kk] = None if np.isnan(v) else round(v, 2)
        return js

    paths = gpd.read_file(GPKG, layer="path_revised")
    vil = gpd.read_file(GPKG, layer="villages_exposure")
    vil = vil[(vil.lon.between(94.9, 95.5)) & (vil.lat.between(17.8, 18.3))]
    st = gpd.read_file(GPKG, layer="routing_stations")
    st = st[(st.chain_m % 1000) == 0]
    ts = gpd.read_file(GPKG, layer="townships")
    ts = ts[ts.TS.isin(["Myanaung", "Ingapu", "Hinthada", "Kyangin", "Lemyethna", "Henzada", "Zalun"])]
    bundle = {
        "layers": layers,
        "paths": gj(paths, ["branch", "name", "length_km"], simplify=25),
        "user_path": gj(gpd.read_file(GPKG, layer="path_user"), ["name"]),
        "townships": gj(ts, ["TS", "TS_MMR"], simplify=150, prec=4),
        "reservoir": gj(gpd.read_file(GPKG, layer="reservoirs_mimu").iloc[[2]], ["Type"]),
        "catchment": gj(gpd.read_file(GPKG, layer="dam_catchment"), ["name"], simplify=60),
        "rivers": gj(gpd.read_file(GPKG, layer="rivers_mimu"), ["Name"], simplify=100, prec=4),
        "villages": gj(vil, ["village", "village_mm", "vt", "ts", "vlg_pcode", "priority", "S1_depth_m", "S1_arrival_h",
                             "S2_depth_m", "S2_arrival_h", "S3_depth_m", "S3_arrival_h", "pop_est_worldpop",
                             "dist_from_dam_km", "nearest_branch", "chainage_km"]),
        "facilities": gj(gpd.read_file(GPKG, layer="facilities_at_risk"),
                         ["facility_type", "name", "S1_depth_m", "S1_arrival_h"]),
        "stations": gj(st, ["branch", "chain_m", "S1_arrival_h", "S1_peak_h", "S1_Qpeak", "S1_rise_m", "S2_arrival_h",
                            "S2_Qpeak", "S3_arrival_h", "S3_Qpeak"]),
        "dam": [18.10800225032718, 95.07266541338852],
    }
    # charts: hydrographs and along-path profiles
    z = np.load(DATA / "routing_timeseries.npz")
    th = z["t_hydro"] / 3600
    sel = th <= 12
    bundle["hydro"] = {"t": np.round(th[sel][::6], 3).tolist(),
                       **{k: np.round(z[f"inflow_{k}"][sel][::6], 0).tolist() for k in ["S1", "S2", "S3"]}}
    stations = pd.read_csv(OUT / "routing_stations.csv")
    prof = {}
    for b in ["A", "B"]:
        s = stations[stations.branch == b].iloc[::2]
        prof[b] = {"km": (s.chain_m / 1000).round(1).tolist(),
                   **{f"{k}_{v}": s[f"{k}_{v}"].round(2).tolist() for k in ["S1", "S2", "S3"]
                      for v in ["arrival_h", "Qpeak", "rise_m"]}}
    bundle["profile"] = prof
    for name in ["breach_scenarios", "exposure_summary", "path_comparison", "inundation_volume_check", "roads_flooded_km"]:
        bundle[name] = json.loads(pd.read_csv(OUT / f"{name}.csv").to_json(orient="records"))
    bundle["vt_summary"] = json.loads(pd.read_csv(OUT / "village_tract_summary.csv").to_json(orient="records"))
    bundle["facilities_table"] = json.loads(pd.read_csv(OUT / "facilities_at_risk.csv").to_json(orient="records"))
    txt = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    (WEB / "bundle.json").write_text(txt, encoding="utf-8")
    print(f"web bundle {len(txt)/1e6:.2f} MB")


# ------------------------------------------------------------------------------------------------
def figures():
    plt.rcParams.update({"font.family": ["Segoe UI", "Myanmar Text", "DejaVu Sans"], "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                         "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6})
    import contextily as cx
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    vil = gpd.read_file(GPKG, layer="villages_exposure").to_crs(3857)
    paths = gpd.read_file(GPKG, layer="path_revised").to_crs(3857)
    user = gpd.read_file(GPKG, layer="path_user").to_crs(3857)
    ts = gpd.read_file(GPKG, layer="townships").to_crs(3857)
    dam = gpd.read_file(GPKG, layer="dam").to_crs(3857)
    rivers = gpd.read_file(GPKG, layer="rivers_mimu").to_crs(3857)
    ww = gpd.read_file(GPKG, layer="osm_waterways").to_crs(3857)
    ww = ww[ww.waterway.isin(["river", "canal"])]
    towns = gpd.read_file(GPKG, layer="towns").to_crs(3857)
    zz = to_mercator(DATA / "dem_utm.tif", CROP, res=30, resampling=Resampling.bilinear)
    zz = np.nan_to_num(np.where(zz < -1000, np.nan, zz), nan=0)
    gy, gx = np.gradient(zz * 3.0, 30)
    sl = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
    hs = np.clip(np.sin(np.radians(45)) * np.cos(sl) + np.cos(np.radians(45)) * np.sin(sl) * np.cos(np.radians(315) - asp), 0, 1)
    hs = 0.72 + 0.28 * hs  # light relief
    for k, title in [("S1", "Scenario S1: full breach at FSL (best estimate)"), ("S3", "Scenario S3: upper envelope")]:
        fig, ax = plt.subplots(figsize=(10, 10.5))
        d = to_mercator(DATA / f"depth_{k}_env.tif", CROP)
        from rasterio.warp import transform_bounds
        b = transform_bounds("EPSG:4326", "EPSG:3857", *CROP)
        d = np.where(d > 0.05, d, np.nan)
        ax.imshow(d, extent=(b[0], b[2], b[1], b[3]), cmap=ListedColormap(DEPTH_HEX),
                  norm=BoundaryNorm(DEPTH_BINS + [99], 4), alpha=.85, zorder=3)
        ts.boundary.plot(ax=ax, color="#555", lw=.7, ls="--", zorder=4)
        user.plot(ax=ax, color="#f2b705", lw=3, zorder=5)
        paths[paths.branch == "A"].plot(ax=ax, color="#c0262d", lw=1.6, zorder=6)
        paths[paths.branch == "B"].plot(ax=ax, color="#ff7f0e", lw=1.3, linestyle="--", zorder=6)
        v0 = vil[vil.priority == 0]
        ax.scatter(v0.geometry.x, v0.geometry.y, s=4, c="#8a8f96", zorder=7, lw=0)
        for p in [4, 3, 2, 1]:
            v = vil[vil.priority == p]
            ax.scatter(v.geometry.x, v.geometry.y, s=26 if p < 3 else 16, c=PRIO_HEX[p], edgecolor="white", lw=.6,
                       zorder=8)
        ax.scatter(dam.geometry.x, dam.geometry.y, marker="^", s=180, c="black", edgecolor="white", zorder=9)
        ax.annotate("Ma Mya Dam", (dam.geometry.x.iloc[0], dam.geometry.y.iloc[0]), xytext=(8, 8),
                    textcoords="offset points", fontsize=10, weight="bold", zorder=10)
        for nm in ["Myanaung", "Ingapu", "Kyangin", "Hinthada"]:
            g = ts[ts.TS == nm]
            if len(g):
                c = g.geometry.iloc[0].representative_point()
                if b[0] < c.x < b[2] and b[1] < c.y < b[3]:
                    ax.text(c.x, c.y, nm.upper(), fontsize=9, color="#444", ha="center", style="italic", zorder=10)
        ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
        ax.imshow(hs, extent=(b[0], b[2], b[1], b[3]), cmap="gray", vmin=0, vmax=1, zorder=1)
        rivers.plot(ax=ax, color="#6f9fcf", lw=2.2, zorder=2)
        ww.plot(ax=ax, color="#9cc0e3", lw=.6, zorder=2)
        for _, t in towns.iterrows():
            if b[0] < t.geometry.x < b[2] and b[1] < t.geometry.y < b[3]:
                ax.scatter(t.geometry.x, t.geometry.y, s=40, marker="s", c="white", edgecolor="black", zorder=9)
                ax.annotate(t.Town, (t.geometry.x, t.geometry.y), xytext=(6, -10), textcoords="offset points",
                            fontsize=9, weight="bold", zorder=10)
        ax.set_axis_off()
        leg = [Patch(color=c, label=l) for c, l in zip(DEPTH_HEX, ["< 0.5 m", "0.5-1.5 m", "1.5-3 m", "> 3 m"])]
        leg += [Line2D([], [], color="#f2b705", lw=3, label="Original projected path (user KML)"),
                Line2D([], [], color="#c0262d", lw=1.6, label="Revised path A: overland (DEM)"),
                Line2D([], [], color="#ff7f0e", lw=1.3, ls="--", label="Revised path B: Ma Mya Chaung channel")]
        leg += [Line2D([], [], marker="o", ls="", color=PRIO_HEX[p], markeredgecolor="white", label=PRIO_TXT[p], ms=7)
                for p in [1, 2, 3, 4]]
        ax.legend(handles=leg, loc="lower left", fontsize=7.5, framealpha=.92, title="Flood depth / paths / villages",
                  title_fontsize=8)
        ax.set_title(f"Ma Mya Dam break: modelled flood depth and village priority\n{title}", fontsize=12, loc="left")
        fig.text(0.01, 0.01, "Rapid desk model (Copernicus GLO-30 DEM, storage routing, Manning n=0.06). Not an official "
                 "warning. Data: MIMU, ODM, OSM, WorldPop. 10 Oct 2026.", fontsize=7, color="#555")
        fig.savefig(OUT / "figures" / f"map_flood_{k}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    # hydrographs
    z = np.load(DATA / "routing_timeseries.npz")
    th = z["t_hydro"] / 3600
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, c, lab in [("S3", "#4a3aa7", "S3 upper envelope"), ("S1", "#2a78d6", "S1 best estimate"),
                      ("S2", "#1baf7a", "S2 partial breach")]:
        ax.plot(th, z[f"inflow_{k}"], color=c, lw=2, label=lab)
    ax.set_xlim(0, 10); ax.set_xlabel("Hours after breach starts"); ax.set_ylabel("Outflow at dam (m³/s)")
    ax.legend(frameon=False); ax.set_title("Breach outflow hydrographs", loc="left")
    fig.savefig(OUT / "figures" / "breach_hydrographs.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    st = pd.read_csv(OUT / "routing_stations.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for b, c, lab in [("A", "#c0262d", "Branch A (overland)"), ("B", "#ff7f0e", "Branch B (channel)")]:
        s = st[st.branch == b]
        axs[0].plot(s.chain_m / 1000, s.S1_arrival_h, color=c, lw=2, label=lab)
        axs[1].plot(s.chain_m / 1000, s.S1_Qpeak, color=c, lw=2, label=lab)
    axs[0].set_xlabel("Distance below dam along path (km)"); axs[0].set_ylabel("Wave arrival (h)")
    axs[0].set_title("S1: flood wave arrival time", loc="left")
    axs[1].set_xlabel("Distance below dam along path (km)"); axs[1].set_ylabel("Peak discharge (m³/s)")
    axs[1].set_title("S1: peak discharge attenuation", loc="left")
    axs[0].legend(frameon=False)
    fig.savefig(OUT / "figures" / "along_path_profiles.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    write_kml()
    web_bundle()
    figures()
