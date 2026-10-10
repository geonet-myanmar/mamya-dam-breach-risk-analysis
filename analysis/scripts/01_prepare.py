"""Step 1 - Load and harmonise all input datasets into one GeoPackage + a UTM DEM.

Inputs : user KMLs, MIMU WFS layers, OSM Overpass JSON, Copernicus GLO-30 DEM tiles, WorldPop 2020.
Outputs: data/work.gpkg (vector layers), data/dem_utm.tif (30 m, EPSG:32646), data/pop_utm.tif
"""
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import LineString, Point, Polygon

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis" / "data"
GPKG = DATA / "work.gpkg"
UTM = "EPSG:32646"
BBOX = (94.85, 17.75, 95.60, 18.45)  # lon/lat study window
DAM = Point(95.07266541338852, 18.10800225032718)


def parse_village_kml(path):
    txt = path.read_text(encoding="utf-8")
    rows = []
    for pm in re.findall(r"<Placemark.*?</Placemark>", txt, re.S):
        desc = re.search(r"<description>(.*?)</description>", pm, re.S).group(1)
        rec = {}
        for line in desc.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                rec[k.strip()] = v.strip()
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["Longitude"] = df["Longitude"].astype(float)
    df["Latitude"] = df["Latitude"].astype(float)
    df["Village Pcode"] = df["Village Pcode"].str.replace(".0", "", regex=False)
    keep = {
        "Village Pcode": "vlg_pcode", "Village ENG": "village", "Village MMR": "village_mm",
        "Village Tract ENG": "vt", "Village Tract MMR": "vt_mm", "Village Tract Pcode": "vt_pcode",
        "Township ENG": "ts", "Longitude": "lon", "Latitude": "lat",
    }
    df = df[list(keep)].rename(columns=keep)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")


def parse_path_kml(path):
    txt = path.read_text(encoding="utf-8")
    coords = re.search(r"<coordinates>(.*?)</coordinates>", txt, re.S).group(1).split()
    pts = [tuple(map(float, c.split(",")[:2])) for c in coords]
    name = re.search(r"<Placemark.*?<name>(.*?)</name>", txt, re.S).group(1)
    return gpd.GeoDataFrame({"name": [name], "source": ["user KML (ယူဆ ရေလမ်းကြောင်း)"]},
                            geometry=[LineString(pts)], crs="EPSG:4326")


def osm_layers(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    water, roads, lakes, places = [], [], [], []
    for el in data["elements"]:
        t = el.get("tags", {})
        if el["type"] == "node":
            places.append({"name": t.get("name:en") or t.get("name"), "name_mm": t.get("name"),
                           "place": t.get("place"), "geometry": Point(el["lon"], el["lat"])})
            continue
        g = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(g) < 2:
            continue
        if "waterway" in t:
            water.append({"osm_id": el["id"], "waterway": t["waterway"], "name": t.get("name:en") or t.get("name"),
                          "name_mm": t.get("name"), "geometry": LineString(g)})
        elif "highway" in t:
            roads.append({"osm_id": el["id"], "highway": t["highway"], "name": t.get("name:en") or t.get("name"),
                          "geometry": LineString(g)})
        elif t.get("natural") == "water" and len(g) >= 4 and g[0] == g[-1]:
            lakes.append({"osm_id": el["id"], "name": t.get("name:en") or t.get("name"), "geometry": Polygon(g)})
    mk = lambda r: gpd.GeoDataFrame(r, geometry="geometry", crs="EPSG:4326")
    return mk(water), mk(roads), mk(lakes), mk(places)


def build_dem():
    tiles = sorted((DATA / "dem").glob("cop_*.tif"))
    srcs = [rasterio.open(t) for t in tiles]
    mosaic, tr = merge(srcs, bounds=BBOX)
    prof = srcs[0].profile.copy()
    prof.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=tr, driver="GTiff", compress="deflate")
    for s in srcs:
        s.close()
    dst_tr, w, h = calculate_default_transform(prof["crs"], UTM, prof["width"], prof["height"], *BBOX, resolution=30)
    out = np.full((h, w), -9999, dtype="float32")
    reproject(mosaic[0], out, src_transform=tr, src_crs=prof["crs"], dst_transform=dst_tr, dst_crs=UTM,
              resampling=Resampling.bilinear, dst_nodata=-9999)
    prof.update(crs=UTM, transform=dst_tr, width=w, height=h, nodata=-9999, dtype="float32")
    with rasterio.open(DATA / "dem_utm.tif", "w", **prof) as dst:
        dst.write(out, 1)
    return dst_tr, w, h, prof


def build_pop(dst_tr, w, h, prof):
    """WorldPop counts -> sum-preserving resample onto the DEM grid (30 m)."""
    with rasterio.open(DATA / "worldpop" / "mmr_ppp_2020_constrained.tif") as src:
        win = rasterio.windows.from_bounds(*BBOX, transform=src.transform)
        arr = src.read(1, window=win).astype("float32")
        arr[arr < 0] = 0
        wtr = src.window_transform(win)
        # density per m2 -> reproject -> multiply by 900 m2 keeps totals approx.
        cell_m2 = (abs(wtr.a) * 111320 * np.cos(np.radians(18.1))) * (abs(wtr.e) * 110574)
        dens = arr / cell_m2
        out = np.zeros((h, w), dtype="float32")
        reproject(dens, out, src_transform=wtr, src_crs=src.crs, dst_transform=dst_tr, dst_crs=UTM,
                  resampling=Resampling.nearest, dst_nodata=0)
        out *= 900.0
        print(f"WorldPop in window: {arr.sum():,.0f}  resampled: {out.sum():,.0f}")
    p = prof.copy(); p.update(nodata=0)
    with rasterio.open(DATA / "pop_utm.tif", "w", **p) as dst:
        dst.write(out, 1)


def main():
    if GPKG.exists():
        GPKG.unlink()
    vil = parse_village_kml(ROOT / "မြန်အောင်မြို့နယ်.kml")
    path = parse_path_kml(ROOT / "ယူဆ ရေလမ်းကြောင်း.kml")
    print("villages (user KML):", len(vil))

    mimu = DATA / "mimu"
    ady_v = gpd.read_file(mimu / "mmr_ady_pplp2_250k_mimu.geojson")
    # all MIMU villages in Myanaung + Ingapu (+ neighbours) so downstream Ingapu villages are covered
    ady_v = ady_v.rename(columns={"VILLAGE": "village", "VLG_MMR": "village_mm", "VT": "vt", "VT_MMR": "vt_mm",
                                  "VT_PCODE": "vt_pcode", "TS": "ts", "VLG_PCODE": "vlg_pcode",
                                  "Longitude": "lon", "Latitude": "lat"})
    ady_v["vlg_pcode"] = ady_v["vlg_pcode"].astype(str)
    ady_v = ady_v[["vlg_pcode", "village", "village_mm", "vt", "vt_pcode", "ts", "lon", "lat", "geometry"]]
    ady_v["vt_mm"] = None
    # prefer user's Myanaung KML (v9.5); add MIMU villages of other townships
    extra = ady_v[~ady_v.vlg_pcode.isin(vil.vlg_pcode)]
    villages = pd.concat([vil.assign(src="user KML (MIMU v9.5)"), extra.assign(src="MIMU WFS pplp2")], ignore_index=True)
    villages = gpd.GeoDataFrame(villages, geometry="geometry", crs="EPSG:4326")
    print("villages combined:", len(villages), villages.ts.value_counts().to_dict())

    layers = {
        "villages": villages,
        "path_user": path,
        "townships": gpd.read_file(mimu / "mmr_polbnda_adm3_250k_mimu_1.geojson"),
        "village_tracts": pd.concat([gpd.read_file(mimu / "mmr_ady_polbnda_adm4_mimu_250k.geojson"),
                                     gpd.read_file(mimu / "mmr_bgo_polbnda_adm4_mimu_250k_1.geojson")]),
        "wards": gpd.read_file(mimu / "mmr_polbnda_adm5_mimu_v9_4.geojson"),
        "towns": gpd.read_file(mimu / "mmr_pplp1_mimu250k.geojson"),
        "reservoirs_mimu": gpd.read_file(mimu / "mmr_dam_lake_2021.geojson"),
        "rivers_mimu": gpd.read_file(mimu / "myanmar_river_network_250k.geojson"),
        "roads_mimu": gpd.read_file(mimu / "mmr_rdsl_mimu_250k.geojson"),
        "schools": gpd.read_file(mimu / "formal_sector_school_location_lowermyanmar_2019.geojson"),
        "health": gpd.read_file(mimu / "health_facilities_myanmar2020_v20241016.geojson"),
        "bridges": gpd.read_file(mimu / "mm_bridges_pt.geojson"),
        "dam": gpd.GeoDataFrame({"name": ["Ma Mya Dam"], "storage_mcm": [86.38], "fsl_area_km2": [5.38],
                                 "crest_len_m": [3596.5], "completed": [2007],
                                 "source": ["ODM myanmar_dams (DoI/MoAI data); location from user"]},
                                geometry=[DAM], crs="EPSG:4326"),
    }
    w, r, l, p = osm_layers(DATA / "osm" / "osm.json")
    layers.update({"osm_waterways": w, "osm_roads": r, "osm_lakes": l, "osm_places": p})
    for name, g in layers.items():
        g = g.to_crs("EPSG:4326").drop(columns=["id"], errors="ignore")  # WFS 'id' clashes with 'Id' in GPKG
        for c in g.columns:
            if c != "geometry" and g[c].dtype == object:
                g[c] = g[c].astype(str).replace({"None": None, "nan": None})
        g.to_file(GPKG, layer=name, driver="GPKG")
        print(f"  {name:16s} {len(g):5d}")

    tr, w_, h_, prof = build_dem()
    print("DEM grid:", w_, "x", h_)
    build_pop(tr, w_, h_, prof)


if __name__ == "__main__":
    main()
