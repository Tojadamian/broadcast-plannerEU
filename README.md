# EU Broadcast Network Planner

Educational mini broadcast network planner for telecommunications master classes. Mirrors core **HTZ Communications** workflows using Python, GIS, ITU-R P.1546 propagation, and parallel **DVB-T/T2** and **5G broadcast** planning modules.

## Quick start

```bash
cd broadcast-planner
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
prepare-examples          # synthetic DEM, clutter, population (offline labs)
streamlit run app/streamlit_app.py
```

Optional: download real EU geodata for the default Brussels–Antwerp case study:

```bash
download-data
```

## Project layout

| Path | Purpose |
|------|---------|
| `broadcast_planner/core/` | Sites, grids, CRS, regions |
| `broadcast_planner/propagation/` | P.1546 grid engine |
| `broadcast_planner/dvb_t/` | DVB-T COFDM, SFN, C/N+I |
| `broadcast_planner/fiveg_bc/` | 5G broadcast numerology, SINR |
| `broadcast_planner/planning/` | Site optimization, exports |
| `app/` | Streamlit + Folium UI |
| `notebooks/` | Master-class labs 01–06 |
| `data/examples/` | Bundled offline dataset |

## Capability matrix vs HTZ Communications

| HTZ capability | This project | Notes |
|----------------|--------------|-------|
| Coverage planning | Yes | Field-strength rasters via ITU-R P.1546 |
| Best-server / overlap | Yes | Per-pixel strongest server |
| Population analysis | Yes | % population above threshold |
| SFN launch delay | Partial | Pilot-based propagation delay |
| SFN guard / CP vs ISD | Partial | Analytic feasibility checks |
| C/N+I (COFDM) | Partial | Simplified t-LNM power sum |
| Automated site planning | Partial | Greedy candidate placement |
| Frequency assignment (AFP) | Light | Co-channel constraint checker |
| Model tuning vs measurements | Light | CSV import + RMSE |
| DVB-T / DVB-T2 | Yes | COFDM thresholds, guard interval |
| 5G broadcast / FeMBMS | Yes | Numerology, CP, SINR, ISD rules |
| Microwave backhaul | No | Out of scope v1 |
| GE06 / ICS coordination | No | Extension topic |

## Default case study

**Brussels–Antwerp corridor** (~80 km), CRS **EPSG:3035** for analysis, WGS84 for map display. See [docs/data_sources.md](docs/data_sources.md) for licenses.

## Labs

1. GIS setup and layer visualization  
2. Single-transmitter P.1546 coverage  
3. DVB-T SFN and guard interval  
4. 5G broadcast ISD and numerology  
5. Best-server and population KPIs  
6. Mini planner: optimize new sites  

## References

- ITU-R P.1546 — terrestrial broadcast propagation  
- EBU Tech 3348 — DVB-T planning  
- ETSI TR 36.976 / TS 103 720 — LTE-based 5G broadcast  
- [Py1546](https://github.com/eeveetza/Py1546) — ITU-R P.1546 Python implementation  
