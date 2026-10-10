# Ma Mya Dam breach risk analysis

**Interactive report:** https://geonet-myanmar.github.io/mamya-dam-breach-risk-analysis/

This is a rapid desk assessment, made on 10 October 2026, of what would happen downstream if the Ma Mya Reservoir embankment (မမြရေလှောင်တမံ) failed. The reservoir is in Myin War Taung village tract, Myanaung Township, Ayeyarwady Region, Myanmar. On that morning local news reported subsidence of the earthen embankment. Spillway release began at 09:00, and no breach had been reported.

> **Disclaimer.** This is model output for preparedness planning, built from open data. It is not an official warning and has not been checked in the field. Follow instructions from township authorities and the Irrigation and Water Utilization Management Department (IWUMD).

## What the analysis covers

- **Revised flood path.** Flow traced over the Copernicus GLO-30 DEM from the dam to the Ngawun (Pathein) River. There are two branches: A follows the overland route and B follows the Ma Mya Chaung channel. Both are compared with the original hand-drawn path.
- **Breach scenarios.** Four empirical peak-discharge equations give three scenarios: S1 is the best estimate, S2 a partial breach and S3 the upper envelope.
- **Flood routing and mapping.** Storage routing runs along DEM cross-sections. Results are maps of flood depth and wave arrival time.
- **Exposure.** Evacuation priority for 1,224 MIMU village points in Myanaung, Ingapu, Kyangin and Hinthada townships. Also counts population (WorldPop 2020), schools, health facilities, bridges and roads.

## Repository layout

| Path | Contents |
|---|---|
| `index.html` | The interactive report, served by GitHub Pages |
| `analysis/outputs/` | KMZ/KML for Google Earth, CSV tables, PNG maps and charts |
| `analysis/scripts/` | Reproducible Python pipeline, run `01_prepare.py` to `06_build_report.py` in order |
| `analysis/web/` | Report template, data bundle and Leaflet CSS |
| `ယူဆ ရေလမ်းကြောင်း.kml` | Original projected flood path |
| `မြန်အောင်မြို့နယ်.kml` | Myanaung Township village points (MIMU Pcode 9.5, KML by GSM) |

The downloaded inputs total about 260 MB and are not committed. These are the DEM tiles, WorldPop, MIMU WFS layers, OSM and intermediate rasters. `01_prepare.py` expects them under `analysis/data/`; see `analysis/README.md` for details.

Python packages: geopandas, rasterio, pysheds, numba, scipy, scikit-image, shapely, matplotlib, pandas.

## Data sources and licences

- Village points, boundaries, schools, health facilities, bridges, reservoirs and rivers: [MIMU](https://themimu.info) / [MIMU GeoNode](https://geonode.themimu.info)
- Dam storage, area and crest length: [Open Development Mekong, Myanmar dams](https://data.opendevelopmentmekong.net/dataset/myanmar-dams) (Department of Irrigation data)
- Terrain: Copernicus DEM GLO-30, © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA
- Population: [WorldPop](https://www.worldpop.org) 2020 constrained, CC BY 4.0
- Waterways and roads: © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, ODbL
- Situation reports: SH News Facebook posts, 10 October 2026 (not redistributed here)
