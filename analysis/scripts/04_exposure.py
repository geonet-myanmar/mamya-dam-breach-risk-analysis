"""Step 4 - Exposure & evacuation priority.

Villages: max flood depth within 150 m of the village point (compensates for the Copernicus DSM, which sees the
village tree canopy / roofs rather than the ground), wave arrival time, chainage from the dam.
Population: WorldPop 2020 (constrained) inside the flood extent (+150 m buffer for the same DSM reason).
Facilities: MIMU schools, health facilities, bridges; OSM roads.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis" / "data"
OUT = ROOT / "analysis" / "outputs"
GPKG = DATA / "work.gpkg"
UTM = 32646
SCEN = ["S1", "S2", "S3"]
R_CELLS = 5  # 150 m search radius on the 30 m grid


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def main():
    ds = rasterio.open(DATA / "dem_utm.tif")
    tr, shp = ds.transform, (ds.height, ds.width)
    fp = disk(R_CELLS)

    depth = {k: rasterio.open(DATA / f"depth_{k}_env.tif").read(1) for k in SCEN}
    dmax150 = {k: ndimage.maximum_filter(v, footprint=fp) for k, v in depth.items()}
    arrival = {}
    for k in SCEN:
        a = np.fmin(rasterio.open(DATA / f"arrival_{k}_A.tif").read(1), rasterio.open(DATA / f"arrival_{k}_B.tif").read(1))
        a = np.where(np.isnan(a), 1e6, a)
        arrival[k] = ndimage.minimum_filter(a, footprint=fp)
    vel = np.maximum(rasterio.open(DATA / "vel_S1_A.tif").read(1), rasterio.open(DATA / "vel_S1_B.tif").read(1))
    vel150 = ndimage.maximum_filter(vel, footprint=fp)

    # distance (m) to S3 extent for the alert band
    wet3 = depth["S3"] > 0.05
    dist3 = ndimage.distance_transform_edt(~wet3) * 30.0

    st = gpd.read_file(GPKG, layer="routing_stations").to_crs(UTM)

    def sample(gdf):
        g = gdf.to_crs(UTM).copy()
        g["geometry"] = g.geometry.representative_point()  # MIMU schools are MultiPoint
        rc = [ds.index(p.x, p.y) for p in g.geometry]
        ok = [(0 <= r < shp[0] and 0 <= c < shp[1]) for r, c in rc]
        g = g[ok].copy()
        rc = [x for x, o in zip(rc, ok) if o]
        R = np.array([r for r, _ in rc]); C = np.array([c for _, c in rc])
        for k in SCEN:
            g[f"{k}_depth_m"] = dmax150[k][R, C].round(2)
            a = arrival[k][R, C]
            g[f"{k}_arrival_h"] = np.where(a > 1e5, np.nan, a).round(1)
        g["S1_velocity_ms"] = vel150[R, C].round(2)
        g["dist_to_S3_extent_m"] = dist3[R, C].round(0)
        nn = st.sindex.nearest(g.geometry, return_all=False)[1]
        g["nearest_branch"] = st.branch.values[nn]
        g["chainage_km"] = (st.chain_m.values[nn] / 1000).round(1)
        g["dist_from_dam_km"] = (g.distance(gpd.read_file(GPKG, layer="dam").to_crs(UTM).geometry.iloc[0]) / 1000).round(1)
        return g

    def priority(r):
        if r.S1_depth_m > 0.05:
            a = r.S1_arrival_h
            if (a <= 3) or (r.S1_depth_m >= 1.5 and a <= 6):
                return 1
            if a <= 12:
                return 2
            return 3
        if r.S3_depth_m > 0.05:
            return 3
        if r.dist_to_S3_extent_m <= 1000:
            return 4
        return 0

    plabel = {1: "P1 - Evacuate immediately on breach warning", 2: "P2 - Evacuate (wave 3-12 h)",
              3: "P3 - Prepare / evacuate (wave >12 h or worst case only)", 4: "Alert - within 1 km of worst-case extent",
              0: "Outside modelled flood zone"}

    vil = sample(gpd.read_file(GPKG, layer="villages"))
    vil["priority"] = vil.apply(priority, axis=1)
    vil["priority_label"] = vil.priority.map(plabel)
    vil["S1_depth_class"] = pd.cut(vil.S1_depth_m, [-1, 0.05, 0.5, 1.5, 3, 99],
                                   labels=["dry", "< 0.5 m", "0.5-1.5 m", "1.5-3 m", "> 3 m"]).astype(str)

    # village population from WorldPop: assign each pop cell to its nearest village (Voronoi) within 2 km
    pop = rasterio.open(DATA / "pop_utm.tif").read(1)
    rr, cc = np.nonzero(pop > 0)
    xs = tr.c + (cc + 0.5) * tr.a
    ys = tr.f + (rr + 0.5) * tr.e
    from scipy.spatial import cKDTree
    allv = vil.copy()
    tree = cKDTree(np.column_stack([allv.geometry.x, allv.geometry.y]))
    d, nn = tree.query(np.column_stack([xs, ys]))
    m = d <= 2000
    vp = np.bincount(nn[m], weights=pop[rr[m], cc[m]], minlength=len(allv))
    vil["pop_est_worldpop"] = np.round(vp, 0)

    at_risk = vil[vil.priority > 0].sort_values(["priority", "S1_arrival_h", "dist_from_dam_km"])
    cols = ["priority", "priority_label", "vlg_pcode", "village", "village_mm", "vt", "vt_mm", "ts", "lon", "lat",
            "pop_est_worldpop", "dist_from_dam_km", "nearest_branch", "chainage_km",
            "S1_depth_m", "S1_depth_class", "S1_arrival_h", "S1_velocity_ms", "S2_depth_m", "S2_arrival_h",
            "S3_depth_m", "S3_arrival_h", "dist_to_S3_extent_m", "src"]
    at_risk[cols].to_csv(OUT / "villages_at_risk.csv", index=False, encoding="utf-8-sig")
    vil.to_crs(4326).to_file(GPKG, layer="villages_exposure", driver="GPKG")
    print("Villages by priority:\n", vil.groupby(["priority", "ts"]).size().unstack(fill_value=0))

    # --- population exposed ----------------------------------------------------------------------
    prow = []
    for k in SCEN:
        wet = depth[k] > 0.05
        wetb = ndimage.binary_dilation(wet, structure=fp)
        prow.append(dict(scenario=k, flooded_km2=wet.sum() * 900 / 1e6,
                         pop_in_extent=pop[wet].sum(), pop_in_extent_plus150m=pop[wetb].sum(),
                         pop_within150m_of_depth_gt_1_5m=pop[ndimage.binary_dilation(depth[k] > 1.5, structure=fp)].sum(),
                         villages_flooded=int((vil[f"{k}_depth_m"] > 0.05).sum()),
                         villages_depth_gt_1_5m=int((vil[f"{k}_depth_m"] > 1.5).sum())))
    ptab = pd.DataFrame(prow).round(0)
    ptab.to_csv(OUT / "exposure_summary.csv", index=False)
    print(ptab.to_string(index=False))

    # --- facilities -------------------------------------------------------------------------------
    sch = sample(gpd.read_file(GPKG, layer="schools"))
    hf = sample(gpd.read_file(GPKG, layer="health"))
    br = sample(gpd.read_file(GPKG, layer="bridges"))
    fac = []
    for name, g, ncol in [("school", sch, "schoolname"), ("health facility", hf, "nmHsp_eng"), ("bridge", br, "nmEng")]:
        h = g[(g.S3_depth_m > 0.05)].copy()
        h["facility_type"] = name
        h["name"] = h[ncol]
        fac.append(h)
    fac = pd.concat(fac)
    fac["lon"], fac["lat"] = fac.to_crs(4326).geometry.x, fac.to_crs(4326).geometry.y
    fcols = ["facility_type", "name", "lon", "lat", "S1_depth_m", "S1_arrival_h", "S2_depth_m", "S3_depth_m",
             "chainage_km", "dist_from_dam_km"]
    fac[fcols].sort_values(["facility_type", "S1_arrival_h"]).to_csv(OUT / "facilities_at_risk.csv", index=False,
                                                                      encoding="utf-8-sig")
    gpd.GeoDataFrame(fac[fcols], geometry=fac.geometry, crs=UTM).to_crs(4326).to_file(GPKG, layer="facilities_at_risk",
                                                                                       driver="GPKG")
    print("Facilities in worst-case extent:", fac.facility_type.value_counts().to_dict(),
          "| in S1:", fac[fac.S1_depth_m > 0.05].facility_type.value_counts().to_dict())

    # --- roads ------------------------------------------------------------------------------------
    roads = gpd.read_file(GPKG, layer="osm_roads").to_crs(UTM)
    rrows = []
    for k in SCEN:
        ext = gpd.read_file(GPKG, layer="flood_extent").to_crs(UTM)
        e = ext[ext.scenario == k].geometry.iloc[0]
        cl = roads.clip(e)
        g = cl.assign(km=cl.length / 1000).groupby("highway").km.sum()
        rrows.append(g.rename(k))
    rtab = pd.concat(rrows, axis=1).fillna(0).round(1)
    rtab.loc["total"] = rtab.sum()
    rtab.to_csv(OUT / "roads_flooded_km.csv")
    print(rtab)

    # --- village tract summary -----------------------------------------------------------------
    vts = (vil[vil.priority > 0].groupby(["ts", "vt"])
           .agg(villages=("village", "size"), p1=("priority", lambda s: (s == 1).sum()),
                p2=("priority", lambda s: (s == 2).sum()), p3=("priority", lambda s: (s == 3).sum()),
                alert=("priority", lambda s: (s == 4).sum()), pop_est=("pop_est_worldpop", "sum"),
                earliest_arrival_h=("S1_arrival_h", "min"), max_depth_S1=("S1_depth_m", "max"))
           .reset_index().sort_values(["earliest_arrival_h", "p1"], ascending=[True, False]))
    vts.to_csv(OUT / "village_tract_summary.csv", index=False, encoding="utf-8-sig")
    print(vts.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
