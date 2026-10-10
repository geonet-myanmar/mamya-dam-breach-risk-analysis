# Ma Mya Dam break: rapid assessment (10 Oct 2026)

Run the scripts in `scripts/` in order (01 → 06). Downloaded inputs are in `data/`.

| Step | Script | What it does |
|---|---|---|
| 1 | `01_prepare.py` | Loads the KMLs, MIMU WFS layers, OSM, Copernicus GLO-30 DEM (UTM 46N, 30 m) and WorldPop 2020 into `data/work.gpkg` |
| 2 | `02_hydrology.py` | Priority-flood + D8 hydrology, the dam catchment, and revised flow paths A (overland) and B (Ma Mya Chaung channel) |
| 3 | `03_dambreak.py` | Breach peak equations, three scenarios, storage routing on DEM cross-sections, and depth/arrival rasters |
| 4 | `04_exposure.py` | Village depth, arrival time and priority; population, schools, health facilities, bridges and roads exposed |
| 5 | `05_export.py` | KML/KMZ for Google Earth, PNG maps and charts, and the web data bundle |
| 6 | `06_build_report.py` | Builds `index.html` at the repo root (GitHub Pages) and `outputs/MaMya_DamBreak_Report.html` (artifact copy) |

Key outputs in `outputs/`:

- `kml/MaMya_DamBreak_Assessment.kmz`: everything for Google Earth
- `kml/MaMya_revised_flood_path.kml`: revised paths only
- `villages_at_risk.csv`: priority villages
- `village_tract_summary.csv`: per-tract summary
- `facilities_at_risk.csv`, `roads_flooded_km.csv`, `breach_scenarios.csv`, `routing_stations.csv`
- `figures/`: static maps and charts

Key assumptions:

- Storage is 86.38 million m³ (ODM/DoI).
- Full supply level is the DSM water level of 86 m plus 2 m.
- The breach invert is at the dam toe (55.3 m).
- Manning n is 0.06.
- Each branch carries the full flow.

This is a desk model, not an official warning.
