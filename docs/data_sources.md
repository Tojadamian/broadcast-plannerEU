# Data sources and licenses

## Default teaching region

**Brussels–Antwerp corridor** (Belgium), bounding box in EPSG:3035:

- X: 4,720,000 – 4,820,000 m  
- Y: 2,260,000 – 2,360,000 m  

Approximate extent: ~100 km × 100 km mixing urban (Brussels, Antwerp), rural, and flat-to-rolling terrain.

## Digital elevation model

| Source | Resolution | License |
|--------|------------|---------|
| [Copernicus GLO-30](https://dataspace.copernicus.eu/) | 30 m | Copernicus free and open |
| EU-DEM (legacy) | 25 m | Copernicus / INSPIRE |

Download script: `broadcast_planner.data.download` (requires network).

Bundled fallback: `data/examples/dem.tif` (synthetic hill model for offline use).

## Land cover / clutter

| Source | Use | License |
|--------|-----|---------|
| [Copernicus Land Cover 2019](https://land.copernicus.eu/) | ITU clutter height R2 mapping | Copernicus free and open |

Classes mapped to ITU representative clutter heights: Rural 10 m, Suburban 15 m, Urban 20 m, Dense Urban 20 m.

## Population

| Source | Resolution | License |
|--------|------------|---------|
| [Eurostat GEOSTAT 1 km grid](https://ec.europa.eu/eurostat/web/gisco/geodata/reference-grids/population) | 1 km | Eurostat reuse policy |
| WorldPop (alternative) | 100 m | Creative Commons |

Bundled fallback: `data/examples/population.tif` (synthetic density grid).

## Transmitter sites

Example network in `data/examples/sites.yaml` — synthetic HPHT/MPMT parameters aligned with ITU 5G broadcast planning examples (ISD ~80 km / ~23 km).

## Citation

When publishing course results, cite Copernicus, Eurostat, and ITU-R P.1546 as applicable.
