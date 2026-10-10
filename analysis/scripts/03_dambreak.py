"""Step 3 - Rapid dam-break assessment for Ma Mya Dam.

1. Reservoir: FSL elevation from DEM hypsometry (area at FSL = 5.38 km2, ODM/DoI), head on breach.
2. Breach outflow: empirical peak-discharge equations (Froehlich 1995, MacDonald & Langridge-Monopolis 1984,
   USBR 1982, Webby 1996) + Froehlich (2008) breach width and failure time -> breach hydrographs for 3 scenarios.
3. Routing: cascade of non-linear storage reaches (500 m) along branch A and B; storage/discharge from
   DEM cross-sections (strip conveyance, Manning n) -> peak discharge, stage, velocity, arrival time per station.
4. Inundation: peak water surface at stations -> nearest-station planar surface, hydraulically connected
   to the flow path -> depth raster per scenario & branch (+ envelope).
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from numba import njit
from rasterio.features import rasterize, shapes
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis" / "data"
OUT = ROOT / "analysis" / "outputs"
GPKG = DATA / "work.gpkg"
UTM = 32646
G = 9.81

STORAGE_FSL = 86.38e6      # m3, ODM myanmar_dams (DoI/MoAI)
AREA_FSL = 5.38e6          # m2
MANNING_N = 0.06           # paddy fields, village groves, scrub (dam-break practice 0.05-0.10)
DX = 500.0                 # station spacing (m)
HALF_W = 6000.0            # half width of cross-sections / mapping corridor (m)
BASE_Q = 30.0              # assumed creek flow incl. ongoing spillway release (m3/s)
SIM_HOURS = 48
DT = 5.0


# ----------------------------------------------------------------------------------------------
def reservoir_fsl(dem, tr, res_poly, catch_poly):
    """Raise a level inside the dam catchment until the connected water area equals the FSL area."""
    shp = dem.shape
    res_m = rasterize([(res_poly, 1)], out_shape=shp, transform=tr).astype(bool)
    cat_m = rasterize([(catch_poly, 1)], out_shape=shp, transform=tr).astype(bool)
    wse0 = float(np.median(dem[res_m]))
    rows = []
    for z in np.arange(wse0, wse0 + 12.01, 0.25):
        lab, _ = ndimage.label((dem <= z) & cat_m)
        ids = np.unique(lab[res_m & (dem <= z)])
        ids = ids[ids > 0]
        wet = np.isin(lab, ids)
        rows.append((z, wet.sum() * 900.0, ((z - dem[wet]).clip(0)).sum() * 900.0))
    h = pd.DataFrame(rows, columns=["z", "area_m2", "vol_above_dsm_m3"])
    fsl = float(np.interp(AREA_FSL, h.area_m2, h.z))
    return wse0, fsl, h


def breach_parameters(vw, hw, hb, mode="piping"):
    """Peak outflow (m3/s) by several empirical equations + Froehlich (2008) geometry/time."""
    eq = {
        "Froehlich (1995)": 0.607 * vw ** 0.295 * hw ** 1.24,
        "MacDonald & Langridge-Monopolis (1984)": 1.154 * (vw * hw) ** 0.412,
        "USBR (1982)": 19.1 * hw ** 1.85,
        "Webby (1996)": 0.0443 * G ** 0.5 * vw ** 0.365 * hw ** 1.40,
    }
    ko = 1.3 if mode == "overtopping" else 1.0
    b_avg = 0.27 * ko * vw ** 0.32 * hb ** 0.04          # average breach width (m)
    tf = 63.2 * np.sqrt(vw / (G * hb ** 2))               # failure time (s)
    return eq, b_avg, tf


def breach_hydrograph(qp, vw, tf, t):
    """Linear rise to Qp over tf, then exponential recession holding the remaining volume."""
    q = np.where(t <= tf, qp * t / tf, 0.0)
    vrem = max(vw - 0.5 * qp * tf, 0.05 * vw)
    k = vrem / qp
    q = np.where(t > tf, qp * np.exp(-(t - tf) / k), q)
    return q


# ----------------------------------------------------------------------------------------------
def stations_from_cells(cells, tr, dem):
    xy = np.array([rasterio.transform.xy(tr, r, c) for r, c in cells])
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    # smooth for direction (approx 600 m window)
    k = 21
    pad = np.pad(xy, ((k // 2, k // 2), (0, 0)), mode="edge")
    sm = np.column_stack([np.convolve(pad[:, i], np.ones(k) / k, mode="valid") for i in range(2)])
    sdist = np.arange(0, seg[-1], DX)
    idx = np.searchsorted(seg, sdist).clip(0, len(seg) - 1)
    st = pd.DataFrame({"chain_m": sdist, "x": xy[idx, 0], "y": xy[idx, 1], "cell": idx})
    t = np.gradient(sm, axis=0)[idx]
    t /= np.linalg.norm(t, axis=1)[:, None]
    st["nx"], st["ny"] = -t[:, 1], t[:, 0]
    st["z_cell"] = [dem[cells[i][0], cells[i][1]] for i in idx]
    return st


def cross_sections(st, dem, tr):
    offs = np.arange(-HALF_W, HALF_W + 1, 30.0)
    inv = ~tr
    prof = np.empty((len(st), len(offs)), dtype="float64")
    for i, r in st.iterrows():
        xs, ys = r.x + offs * r.nx, r.y + offs * r.ny
        cols, rows = inv * (xs, ys)
        prof[i] = ndimage.map_coordinates(dem, [rows - 0.5, cols - 0.5], order=1, cval=np.nan)
    prof[prof < -1000] = np.nan
    prof = np.where(np.isnan(prof), 999.0, prof)
    return offs, prof


def thalweg(st, prof, offs):
    c = len(offs) // 2
    z = prof[:, c - 3:c + 4].min(axis=1)                   # channel within +-90 m
    z = pd.Series(z).rolling(5, center=True, min_periods=1).median().values
    z = np.minimum.accumulate(z)                            # non-increasing downstream
    s = np.empty_like(z)
    n = 5
    for i in range(len(z)):
        a, b = max(0, i - n), min(len(z) - 1, i + n)
        s[i] = (z[a] - z[b]) / max((b - a) * DX, DX)
    return z, np.clip(s, 1e-4, 0.02)


def rating_tables(prof, z0, slope, hgrid):
    """Strip-conveyance rating per station: A(h), Q(h), top width T(h)."""
    ns, nw = prof.shape
    c = nw // 2
    A = np.zeros((ns, len(hgrid)))
    Q = np.zeros_like(A)
    T = np.zeros_like(A)
    for i in range(ns):
        p = prof[i]
        wl = z0[i] + hgrid[:, None]                         # (nh,1)
        below = p[None, :] < wl                             # (nh,nw)
        # contiguity outward from the lowest point near the centre
        c0 = c - 5 + int(np.argmin(p[c - 5:c + 6]))
        right = np.cumprod(below[:, c0:], axis=1).astype(bool)
        left = np.cumprod(below[:, c0::-1], axis=1)[:, ::-1].astype(bool)
        wet = np.concatenate([left[:, :-1], right], axis=1)
        d = np.where(wet, wl - p[None, :], 0.0).clip(0)
        A[i] = d.sum(axis=1) * 30.0
        T[i] = wet.sum(axis=1) * 30.0
        Q[i] = (d ** (5 / 3)).sum(axis=1) * 30.0 * np.sqrt(slope[i]) / MANNING_N
    return A, Q, T


@njit(cache=True)
def route(qin, A, Q, hgrid, dx, dt, rec_every):
    ns, nt = A.shape[0], qin.shape[0]
    S = np.empty(ns)
    for i in range(ns):  # initial steady base flow
        S[i] = np.interp(qin[0], Q[i], A[i]) * dx
    nrec = (nt - 1) // rec_every + 1
    Qrec = np.zeros((nrec, ns))
    Hrec = np.zeros((nrec, ns))
    qout = np.empty(ns)
    for n in range(nt):
        for i in range(ns):
            a = S[i] / dx
            qout[i] = np.interp(a, A[i], Q[i])
        if n % rec_every == 0:
            k = n // rec_every
            for i in range(ns):
                Qrec[k, i] = qout[i]
                Hrec[k, i] = np.interp(S[i] / dx, A[i], hgrid)
        for i in range(ns):
            inflow = qin[n] if i == 0 else qout[i - 1]
            S[i] = max(S[i] + (inflow - qout[i]) * dt, 0.0)
    return Qrec, Hrec


# ----------------------------------------------------------------------------------------------
def main():
    ds = rasterio.open(DATA / "dem_utm.tif")
    dem, tr = ds.read(1).astype("float64"), ds.transform
    shp = dem.shape

    res = gpd.read_file(GPKG, layer="reservoirs_mimu").to_crs(UTM)
    dam = gpd.read_file(GPKG, layer="dam").to_crs(UTM).geometry.iloc[0]
    res_poly = res.loc[res.distance(dam).idxmin()].geometry
    catch = gpd.read_file(GPKG, layer="dam_catchment").to_crs(UTM).geometry.iloc[0]

    # --- 1. reservoir ------------------------------------------------------------------------
    wse_dsm, fsl_area_match, hyps = reservoir_fsl(dem, tr, res_poly, catch)
    # Area-matching is unreliable (tree canopy on the DSM shoreline inflates land heights, the connected area only
    # reaches 4.6 km2 at +12 m). The spillway is spilling (news, 10 Oct) so the pool is ~at FSL: adopt the DSM
    # water level + 2 m as FSL (DSM area 3.1-4.0 km2 < 5.38 km2 published FSL area).
    fsl = wse_dsm + 2.0
    cellsA = [tuple(x) for x in np.load(DATA / "path_cells_A.npy")]
    cellsB = [tuple(x) for x in np.load(DATA / "path_cells_B.npy")]
    toe = float(dem[cellsA[0]])
    hw_full = fsl - toe
    hb = hw_full + 3.0  # crest assumed ~3 m freeboard above FSL (no published crest level)
    print(f"(area-matched FSL would be {fsl_area_match:.1f} m - rejected, see comment)")
    print(f"Reservoir WSE in DSM {wse_dsm:.1f} m; FSL (area 5.38 km2) ~{fsl:.1f} m; breach invert (toe) {toe:.1f} m")
    print(f"hw = {hw_full:.1f} m, assumed dam height hb = {hb:.1f} m")
    hyps.to_csv(OUT / "reservoir_hypsometry_dem.csv", index=False)

    # --- 2. breach scenarios -----------------------------------------------------------------
    scen_def = {
        "S1": dict(label="Full breach at FSL (best estimate)", vw=STORAGE_FSL, hw=hw_full, pick="median", mode="piping"),
        "S2": dict(label="Partial breach (invert at mid-height)", vw=0.85 * STORAGE_FSL, hw=0.5 * hw_full,
                   pick="median", mode="piping"),
        "S3": dict(label="Full breach at FSL (upper envelope)", vw=STORAGE_FSL, hw=hw_full, pick="max",
                   mode="overtopping"),
    }
    t = np.arange(0, SIM_HOURS * 3600, DT)
    brows, hydro = [], {}
    for k, s in scen_def.items():
        eq, bw, tf = breach_parameters(s["vw"], s["hw"], hb if k != "S2" else 0.5 * hb + 3, s["mode"])
        vals = np.array(list(eq.values()))
        qp = float(np.median(vals) if s["pick"] == "median" else vals.max())
        s.update(qp=qp, bw=bw, tf=tf)
        hydro[k] = breach_hydrograph(qp, s["vw"], tf, t) + BASE_Q
        brows.append({"scenario": k, "description": s["label"], "Vw_Mm3": s["vw"] / 1e6, "hw_m": s["hw"],
                      **{f"Qp {n}": v for n, v in eq.items()}, "Qp_used_m3s": qp,
                      "breach_width_m (Froehlich 2008)": bw, "failure_time_h (Froehlich 2008)": tf / 3600})
    btab = pd.DataFrame(brows)
    btab.round(1).to_csv(OUT / "breach_scenarios.csv", index=False)
    print(btab.round(1).T.to_string())

    # --- 3. routing per branch ---------------------------------------------------------------
    hgrid = np.r_[np.arange(0, 2, 0.05), np.arange(2, 40.01, 0.2)]
    rec_every = int(60 / DT)
    st_all, ts_out, xsec = [], {}, {}
    for br, cells in [("A", cellsA), ("B", cellsB)]:
        st = stations_from_cells(cells, tr, dem)
        offs, prof = cross_sections(st, dem, tr)
        z0, slope = thalweg(st, prof, offs)
        A, Q, T = rating_tables(prof, z0, slope, hgrid)
        A[:, 0], Q[:, 0] = 0.0, 0.0
        # enforce strictly increasing tables for interpolation
        A = np.maximum.accumulate(A + np.arange(A.shape[1]) * 1e-6, axis=1)
        Q = np.maximum.accumulate(Q + np.arange(Q.shape[1]) * 1e-6, axis=1)
        st["branch"], st["z_thalweg"], st["slope"] = br, z0, slope
        xsec[br] = (st, offs, prof)
        for k, s in scen_def.items():
            Qr, Hr = route(hydro[k], A, Q, hgrid, DX, DT, rec_every)
            tmin = np.arange(Qr.shape[0])  # minutes
            h0 = Hr[0]
            pk = Qr.argmax(axis=0)
            st[f"{k}_Qpeak"] = Qr.max(axis=0)
            st[f"{k}_hpeak"] = Hr.max(axis=0)
            st[f"{k}_wse"] = z0 + st[f"{k}_hpeak"]
            st[f"{k}_rise_m"] = Hr.max(axis=0) - h0
            st[f"{k}_wse_base"] = z0 + h0
            a0 = np.array([np.interp(h0[i], hgrid, A[i]) for i in range(len(st))])
            amax = np.array([np.interp(Hr.max(axis=0)[i], hgrid, A[i]) for i in range(len(st))])
            st[f"{k}_dA"] = amax - a0
            apk = np.array([np.interp(Hr[pk[i], i], hgrid, A[i]) for i in range(len(st))])
            st[f"{k}_vel"] = st[f"{k}_Qpeak"] / np.maximum(apk, 1)
            st[f"{k}_width"] = [np.interp(st[f"{k}_hpeak"][i], hgrid, T[i]) for i in range(len(st))]
            arr = []
            for i in range(len(st)):
                over = np.where(Hr[:, i] - h0[i] >= 0.3)[0]
                arr.append(tmin[over[0]] / 60 if len(over) else np.nan)
            st[f"{k}_arrival_h"] = arr
            st[f"{k}_peak_h"] = tmin[pk] / 60
            ts_out[(br, k)] = (Qr[:, ::4], Hr[:, ::4])  # every 4th station for plots
        st_all.append(st)
        print(f"Branch {br}: {len(st)} stations; S1 Qpeak dam->end {st.S1_Qpeak.iloc[0]:.0f} -> {st.S1_Qpeak.iloc[-1]:.0f} m3/s;"
              f" arrival at end {st.S1_arrival_h.iloc[-1]:.1f} h")
    stations = pd.concat(st_all, ignore_index=True)
    stations.to_csv(OUT / "routing_stations.csv", index=False)
    gst = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.x, stations.y), crs=UTM)
    gst.to_crs(4326).to_file(GPKG, layer="routing_stations", driver="GPKG")
    np.savez_compressed(DATA / "routing_timeseries.npz",
                        **{f"{b}_{k}_{v}": arr[j] for (b, k), arr in ts_out.items() for j, v in enumerate("QH")},
                        t_hydro=t[::12], **{f"inflow_{k}": hydro[k][::12] for k in hydro})

    # --- 4. inundation -------------------------------------------------------------------------
    rr, cc = np.mgrid[0:shp[0], 0:shp[1]]
    X = tr.c + (cc + 0.5) * tr.a
    Y = tr.f + (rr + 0.5) * tr.e
    valid = dem > -1000
    res_m = rasterize([(res_poly.buffer(100), 1), (catch, 1)], out_shape=shp, transform=tr).astype(bool)
    prof = ds.profile.copy()
    prof.update(dtype="float32", nodata=0, compress="deflate")
    depth_env = {}
    vol_rows = []
    for k in scen_def:
        env = np.zeros(shp, dtype="float32")
        for br, cells in [("A", cellsA), ("B", cellsB)]:
            s = stations[stations.branch == br].reset_index(drop=True)
            _, offs, xprof = xsec[br]
            # wetted corridor = union of discs around each station, radius = wetted half-width of the routed
            # cross-section at peak stage (smoothed over 5 stations); cells must also lie below the local peak WSE
            # and be connected to the flow path, so the disc only bounds lateral spread (no transect fan artefacts)
            c = len(offs) // 2
            half = []
            for i, r in s.iterrows():
                p = xprof[i]
                c0 = c - 5 + int(np.argmin(p[c - 5:c + 6]))
                wl = r[f"{k}_wse"]
                R = c0
                while R + 1 < len(p) and p[R + 1] < wl:
                    R += 1
                L = c0
                while L - 1 >= 0 and p[L - 1] < wl:
                    L -= 1
                half.append(max(abs(offs[L]), abs(offs[R])) + 60)
            half = pd.Series(half).rolling(5, center=True, min_periods=1).median().values
            s[f"{k}_halfwidth"] = half
            stations.loc[stations.branch == br, f"{k}_halfwidth"] = half
            corr = unary_union([Point(xy).buffer(h) for xy, h in zip(s[["x", "y"]].values, half)])
            cm = rasterize([(corr, 1)], out_shape=shp, transform=tr).astype(bool) & valid & ~res_m
            idx = np.where(cm)
            tree = cKDTree(s[["x", "y"]].values)
            dd, nn = tree.query(np.column_stack([X[idx], Y[idx]]), k=4)
            wgt = 1.0 / np.maximum(dd, 1.0) ** 2
            wgt /= wgt.sum(axis=1, keepdims=True)
            wse = np.full(shp, np.nan)
            base = np.full(shp, np.nan)
            wse[idx] = (s[f"{k}_wse"].values[nn] * wgt).sum(axis=1)      # IDW of 4 nearest stations
            base[idx] = (s[f"{k}_wse_base"].values[nn] * wgt).sum(axis=1)
            d = np.zeros(shp, dtype="float32")
            d[idx] = (wse[idx] - np.maximum(dem[idx], base[idx])).clip(0)
            d[idx] = np.where(dem[idx] < wse[idx], d[idx], 0)
            lab, _ = ndimage.label(d > 0.05, structure=np.ones((3, 3)))
            seeds = np.unique([lab[r, c] for r, c in cells if lab[r, c] > 0])
            d = np.where(np.isin(lab, seeds), d, 0).astype("float32")
            env = np.maximum(env, d)
            arr_r = np.full(shp, np.nan, dtype="float32")
            arr_r[idx] = s[f"{k}_arrival_h"].values[nn[:, 0]]
            arr_r[d <= 0.05] = np.nan
            p2 = prof.copy(); p2.update(nodata=np.nan)
            with rasterio.open(DATA / f"arrival_{k}_{br}.tif", "w", **p2) as dst:
                dst.write(arr_r, 1)
            vel_r = np.zeros(shp, dtype="float32")
            vel_r[idx] = s[f"{k}_vel"].values[nn[:, 0]]
            with rasterio.open(DATA / f"vel_{k}_{br}.tif", "w", **prof) as dst:
                dst.write(np.where(d > 0.05, vel_r, 0).astype("float32"), 1)
            with rasterio.open(DATA / f"depth_{k}_{br}.tif", "w", **prof) as dst:
                dst.write(d, 1)
            vol_rows.append(dict(scenario=k, branch=br, area_km2=(d > 0.05).sum() * 900 / 1e6,
                                 volume_Mm3=d.sum() * 900 / 1e6,
                                 routing_envelope_volume_Mm3=(s[f"{k}_dA"] * DX).sum() / 1e6))
        with rasterio.open(DATA / f"depth_{k}_env.tif", "w", **prof) as dst:
            dst.write(env, 1)
        depth_env[k] = env
        vol_rows.append(dict(scenario=k, branch="A+B envelope", area_km2=(env > 0.05).sum() * 900 / 1e6,
                             volume_Mm3=env.sum() * 900 / 1e6))
    stations.to_csv(OUT / "routing_stations.csv", index=False)
    gst = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.x, stations.y), crs=UTM)
    gst.to_crs(4326).to_file(GPKG, layer="routing_stations", driver="GPKG")
    vt = pd.DataFrame(vol_rows)
    vt["reservoir_volume_Mm3"] = vt.scenario.map({k: v["vw"] / 1e6 for k, v in scen_def.items()})
    print(vt.round(1).to_string(index=False))
    vt.round(2).to_csv(OUT / "inundation_volume_check.csv", index=False)

    # flood extent polygons (depth classes) for mapping / KML
    polys = []
    for k, env in depth_env.items():
        cls = np.digitize(env, [0.05, 0.5, 1.5, 3.0]).astype("uint8")  # 0 dry, 1..4
        cls = ndimage.median_filter(cls, size=3)
        for geom, v in shapes(cls, mask=cls > 0, transform=tr):
            g = shape(geom)
            if g.area >= 900 * 4:
                polys.append({"scenario": k, "depth_class": int(v), "geometry": g})
    fp = gpd.GeoDataFrame(polys, crs=UTM)
    fp["depth_label"] = fp.depth_class.map({1: "< 0.5 m", 2: "0.5-1.5 m", 3: "1.5-3 m", 4: "> 3 m"})
    fp = fp.dissolve(by=["scenario", "depth_class", "depth_label"], as_index=False)
    fp["geometry"] = fp.geometry.simplify(15)
    fp.to_crs(4326).to_file(GPKG, layer="flood_depth_classes", driver="GPKG")
    ext = gpd.GeoDataFrame([{"scenario": k, "geometry": fp[fp.scenario == k].union_all()} for k in depth_env], crs=UTM)
    ext["area_km2"] = ext.area / 1e6
    ext.to_crs(4326).to_file(GPKG, layer="flood_extent", driver="GPKG")
    print(ext[["scenario", "area_km2"]])


if __name__ == "__main__":
    main()
