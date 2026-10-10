"""Step 2 - DEM hydrology: conditioned DEM, D8 flow direction/accumulation, dam catchment,
and the DEM-derived downstream flow path from the dam toe (the "revised" flood path).

Two variants are traced:
  * Branch A - pure DEM steepest descent (overland route the breach wave takes once it leaves the channel)
  * Branch B - DEM with OSM 'Ma Mya Chaung' burned in 20 m (wave stays in the mapped creek channel)
Both traces stop at the Ayeyarwady or Ngawun (Pathein) River (MIMU 250k river lines).
"""
import heapq
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pysheds.grid import Grid
from pysheds.sview import Raster
from numba import njit
from rasterio.features import rasterize, shapes
from shapely.geometry import LineString, Point, shape

if not hasattr(np, "in1d"):  # pysheds still calls np.in1d (removed in NumPy 2.x)
    np.in1d = np.isin

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis" / "data"
GPKG = DATA / "work.gpkg"
UTM = 32646
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)  # N NE E SE S SW W NW (pysheds default)
OFFS = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1)}


_DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
_DC = np.array([0, 1, 1, 1, 0, -1, -1, -1])
_CODE = np.array(DIRMAP)


@njit(cache=True)
def priority_flood_eps(z, valid):
    """Priority-flood + epsilon (Barnes et al. 2014): every valid cell gets a strictly downhill route
    to the grid edge / nodata. Also returns the D8 receiver code of each cell (the cell that flooded it)."""
    nr, nc = z.shape
    out = z.copy()
    done = np.zeros((nr, nc), np.bool_)
    fdir = np.zeros((nr, nc), np.int16)
    heap = [(0.0, 0, 0, 0)]
    heap.pop()
    cnt = 0
    for r in range(nr):
        for c in range(nc):
            if not valid[r, c]:
                continue
            edge = r == 0 or c == 0 or r == nr - 1 or c == nc - 1
            if not edge:
                for k in range(8):
                    if not valid[r + _DR[k], c + _DC[k]]:
                        edge = True
                        break
            if edge:
                heapq.heappush(heap, (out[r, c], cnt, r, c))
                cnt += 1
                done[r, c] = True
    while len(heap) > 0:
        e, _, r, c = heapq.heappop(heap)
        for k in range(8):
            rr, cc = r + _DR[k], c + _DC[k]
            if rr < 0 or cc < 0 or rr >= nr or cc >= nc or done[rr, cc] or not valid[rr, cc]:
                continue
            done[rr, cc] = True
            v = out[rr, cc]
            if v <= e:
                v = np.nextafter(e, np.inf) + 1e-4
                out[rr, cc] = v
            fdir[rr, cc] = _CODE[(k + 4) % 8]  # neighbour drains back to the cell that reached it
            heapq.heappush(heap, (v, cnt, rr, cc))
            cnt += 1
    return out, fdir


def condition(dem_path, burn=None, depth=20.0):
    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))
    arr = np.asarray(dem, dtype="float64").copy()
    valid = arr > -1000
    if burn is not None:
        arr[burn & valid] -= depth
    filled, fd = priority_flood_eps(arr, valid)
    fdir = Raster(fd.astype("int64"), viewfinder=dem.viewfinder)
    fdir.nodata = 0
    acc = grid.accumulation(fdir, dirmap=DIRMAP)
    return grid, dem, filled, fdir, acc


def trace(fdir, r, c, stop=None, max_steps=200000):
    path = [(r, c)]
    f = np.asarray(fdir)
    for _ in range(max_steps):
        d = int(f[r, c])
        if d not in OFFS:
            break
        dr, dc = OFFS[d]
        r, c = r + dr, c + dc
        if not (0 <= r < f.shape[0] and 0 <= c < f.shape[1]):
            break
        path.append((r, c))
        if stop is not None and stop[r, c]:
            break
    return path


def main():
    dem_ds = rasterio.open(DATA / "dem_utm.tif")
    tr, shp = dem_ds.transform, (dem_ds.height, dem_ds.width)
    raw = dem_ds.read(1)

    res = gpd.read_file(GPKG, layer="reservoirs_mimu").to_crs(UTM)
    dam = gpd.read_file(GPKG, layer="dam").to_crs(UTM).geometry.iloc[0]
    mamya_res = res.loc[res.distance(dam).idxmin()].geometry
    res_mask = rasterize([(mamya_res, 1)], out_shape=shp, transform=tr).astype(bool)

    # --- reservoir & dam geometry from DEM -------------------------------------------------
    wse_res = float(np.median(raw[res_mask]))
    ring = dam.buffer(1200).difference(mamya_res.buffer(60))
    ring_mask = rasterize([(ring, 1)], out_shape=shp, transform=tr).astype(bool)
    toe_val = float(np.percentile(raw[ring_mask], 1))
    emb = mamya_res.buffer(150).difference(mamya_res).intersection(dam.buffer(1500))
    emb_mask = rasterize([(emb, 1)], out_shape=shp, transform=tr).astype(bool)
    crest = float(np.percentile(raw[emb_mask], 90))
    print(f"Reservoir WSE (DSM) {wse_res:.1f} m | embankment crest ~{crest:.1f} m | downstream toe ~{toe_val:.1f} m")
    print(f"Reservoir area (MIMU 2021) {mamya_res.area/1e6:.2f} km2")

    # --- burn vectors ---------------------------------------------------------------------------
    ww = gpd.read_file(GPKG, layer="osm_waterways").to_crs(UTM)
    mm = ww[ww.name_mm.fillna("").str.contains("မမြ")]
    ayr = gpd.read_file(GPKG, layer="rivers_mimu").to_crs(UTM)
    burn = rasterize([(g, 1) for g in list(mm.geometry) + list(ayr.geometry)], out_shape=shp, transform=tr,
                     all_touched=True).astype(bool)
    print("OSM Ma Mya Chaung segments:", len(mm), "length km:", round(mm.length.sum() / 1000, 1))

    river_stop = rasterize([(g.buffer(300), 1) for g in ayr.geometry], out_shape=shp, transform=tr).astype(bool)

    out = {}
    for tag, b in [("burned", burn), ("dem", None)]:
        grid, dem, inf, fdir, acc = condition(DATA / "dem_utm.tif", b)
        A = np.asarray(acc)
        # start: the highest-accumulation cell just outside the reservoir near the dam (= creek leaving the dam)
        cand = ring_mask & ~res_mask
        cand_idx = np.argwhere(cand)
        best = cand_idx[np.argmax(A[cand])]
        p = trace(fdir, best[0], best[1], stop=river_stop)
        xy = [rasterio.transform.xy(tr, r, c) for r, c in p]
        line = LineString(xy)
        elev = np.array([raw[r, c] for r, c in p])
        out[tag] = dict(line=line, cells=p, acc=A, fdir=np.asarray(fdir), elev=elev, start=best)
        print(f"[{tag}] start cell {tuple(best)} acc={A[tuple(best)]*900/1e6:.1f} km2 | path {line.length/1000:.1f} km "
              f"| elev {elev[0]:.0f}->{elev[-1]:.0f} m | end {rasterio.transform.xy(tr, *p[-1])}")
        if tag == "dem":
            # catchment of the dam (upstream area draining to the start cell)
            x0, y0 = rasterio.transform.xy(tr, *best)
            catch = grid.catchment(x=int(best[1]), y=int(best[0]), fdir=fdir, dirmap=DIRMAP, xytype="index")
            cm = np.asarray(catch).astype(np.uint8)
            polys = [shape(g) for g, v in shapes(cm, mask=cm == 1, transform=tr)]
            cat = gpd.GeoDataFrame({"name": ["Ma Mya Dam catchment (DEM)"]}, geometry=[max(polys, key=lambda g: g.area)],
                                   crs=UTM)
            print(f"Dam catchment area: {cat.area.iloc[0]/1e6:.1f} km2")
            cat.to_crs(4326).to_file(GPKG, layer="dam_catchment", driver="GPKG")
            # stream network (acc > 5 km2) for context
            streams = grid.extract_river_network(fdir, acc > (5e6 / 900), dirmap=DIRMAP)
            sg = gpd.GeoDataFrame.from_features(streams["features"], crs=UTM)
            sg.to_crs(4326).to_file(GPKG, layer="streams_dem", driver="GPKG")
            with rasterio.open(DATA / "acc.tif", "w", driver="GTiff", height=shp[0], width=shp[1], count=1,
                               dtype="float32", crs=UTM, transform=tr, compress="deflate") as d:
                d.write(A.astype("float32"), 1)
            with rasterio.open(DATA / "fdir.tif", "w", driver="GTiff", height=shp[0], width=shp[1], count=1,
                               dtype="int16", crs=UTM, transform=tr, compress="deflate") as d:
                d.write(np.asarray(fdir).astype("int16"), 1)

    # --- compare with the user's hand-drawn path ------------------------------------------------
    user = gpd.read_file(GPKG, layer="path_user").to_crs(UTM).geometry.iloc[0]
    rows = []
    for tag in ["dem", "burned"]:
        L = out[tag]["line"]
        # mean distance from user path vertices/densified points to the DEM path
        pts = [user.interpolate(d) for d in np.arange(0, user.length, 100)]
        dist = np.array([L.distance(p) for p in pts])
        covered = max(L.project(p) for p in pts)  # how far down the DEM path the user's line reaches
        rows.append(dict(path={"dem": "Branch A (DEM)", "burned": "Branch B (channel)"}[tag], length_km=L.length / 1000,
                         hausdorff_km=user.hausdorff_distance(L) / 1000, mean_offset_km=dist.mean() / 1000,
                         max_offset_km=dist.max() / 1000, user_line_reaches_km=covered / 1000,
                         share_covered_pct=100 * covered / L.length))
    import pandas as pd
    cmp = pd.DataFrame(rows)
    cmp.loc[len(cmp)] = dict(path="User KML", length_km=user.length / 1000)
    print(cmp.round(2).to_string(index=False))
    cmp.to_csv(ROOT / "analysis" / "outputs" / "path_comparison.csv", index=False)

    paths = gpd.GeoDataFrame(
        {"branch": ["A", "B"],
         "name": ["Branch A - overland steepest descent (DEM)", "Branch B - Ma Mya Chaung channel (OSM-constrained)"],
         "method": ["Copernicus GLO-30, priority-flood+epsilon, D8", "as A, OSM Ma Mya Chaung burned 20 m"],
         "length_km": [out["dem"]["line"].length / 1000, out["burned"]["line"].length / 1000]},
        geometry=[out["dem"]["line"], out["burned"]["line"]], crs=UTM)
    paths.to_crs(4326).to_file(GPKG, layer="path_revised", driver="GPKG")
    np.save(DATA / "path_cells_A.npy", np.array(out["dem"]["cells"]))
    np.save(DATA / "path_cells_B.npy", np.array(out["burned"]["cells"]))
    meta = dict(wse_res=wse_res, crest=crest, toe=toe_val, res_area_km2=mamya_res.area / 1e6)
    pd.Series(meta).to_csv(DATA / "dam_geometry.csv")


if __name__ == "__main__":
    main()
