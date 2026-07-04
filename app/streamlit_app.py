"""Streamlit UI for EU Broadcast Network Planner."""

from __future__ import annotations

import io
import tempfile
import logging
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# --- Core Domain Imports ---
from broadcast_planner.core.crs import to_display
from broadcast_planner.core.grids import align_layers_to_grid, make_grid, read_raster
from broadcast_planner.core.regions import REGIONS, get_region
from broadcast_planner.core.sites import Site, SiteCollection, load_sites, save_sites

# --- Telecommunications Logic Imports ---
from broadcast_planner.dvb_t.coverage import run_dvb_t_coverage
from broadcast_planner.dvb_t.models import DvbTConfig, GUARD_INTERVAL_US, MODULATION_CN
from broadcast_planner.dvb_t.sfn import assign_sfn_delays, sfn_delay_table
from broadcast_planner.fiveg_bc.coverage import run_fiveg_broadcast_coverage
from broadcast_planner.fiveg_bc.models import FiveGBroadcastConfig, NUMEROLOGY_PRESETS, TOPOLOGY_PRESETS

# --- Planning & Export Imports ---
from broadcast_planner.planning.exports import (
    export_coverage_png,
    export_geotiff,
    export_kpi_csv,
    export_pdf_report,
    export_sfn_table_csv,
    export_sites_geojson,
)
from broadcast_planner.planning.interference import cochannel_violations
from broadcast_planner.planning.optimizer import greedy_add_sites


# ==========================================
# LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"
OUTPUT_DIR = PROJECT_ROOT / "output"


# ==========================================
# STATE MANAGEMENT
# ==========================================
def init_session_state() -> None:
    """Initializes all necessary session state variables on app boot."""
    region = get_region(st.session_state.get("region_name", "brussels_antwerp"))
    default_states = {
        "region_name": "brussels_antwerp",
        "res_m": 250.0,  # Fixed default to match native data layers
        "sites": load_sites(region.sites_path()),
        "dem": None,
        "clutter": None,
        "population": None,
        "layer_grid": None,
        "dvb_result": None,
        "fiveg_result": None,
        "opt_result": None,
    }
    
    for key, default_val in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_val

    # Auto-load GIS data if not already loaded
    if st.session_state["dem"] is None:
        _load_gis_layers()


def _load_gis_layers() -> None:
    """Loads underlying GIS raster matrices into memory."""
    region = get_region(st.session_state["region_name"])
    st.session_state["dem"], st.session_state["layer_grid"] = read_raster(region.dem_path())
    st.session_state["clutter"], _ = read_raster(region.clutter_path())
    st.session_state["population"], _ = read_raster(region.population_path())
    # Reset simulation results when region changes
    st.session_state["dvb_result"] = None
    st.session_state["fiveg_result"] = None
    st.session_state["opt_result"] = None


# ==========================================
# HELPER COMPONENTS
# ==========================================
def make_map(sites: SiteCollection, coverage_array: np.ndarray, tech_name: str, test_points: list = None) -> folium.Map:
    """Generates an interactive Folium map perfectly framed around the data."""
    region = get_region(st.session_state["region_name"])
    minx, miny, maxx, maxy = region.bounds

    # 1. Transform all 4 corners to WGS84 (lon, lat)
    corners = [
        to_display(minx, miny),
        to_display(maxx, miny),
        to_display(maxx, maxy),
        to_display(minx, maxy)
    ]

    # 2. Extract Latitude and Longitude explicitly (to_display returns (lon, lat))
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]

    # 3. Compute strict South, North, West, East bounds
    south = min(lats)
    north = max(lats)
    west = min(lons)
    east = max(lons)

    # 4. Initialize the map perfectly in the center
    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)

    # 5. Add Site Markers safely (transmitters)
    for site in sites.sites:
        c1, c2 = to_display(site.x, site.y)
        lat, lon = (c1, c2) if c1 > c2 else (c2, c1)
        # Use getattr as a fallback for old cached data classes without the boolean
        color = "red" if getattr(site, 'is_pilot', False) else "blue"
        folium.Marker(
            [lat, lon],
            popup=f"TX: {site.name}<br>{site.erp_kw} kW",
            icon=folium.Icon(color=color, icon="broadcast"),
        ).add_to(m)

    # 5b. Add Test Points (receivers) if provided
    if test_points:
        for x, y, name in test_points:
            lon, lat = to_display(x, y)
            folium.Marker(
                [lat, lon],
                popup=f"RX: {name}",
                icon=folium.Icon(color="green", icon="check"),
            ).add_to(m)

    # 6. Build Leaflet-safe bounds that CANNOT wrap around the world
    bounds = [[south, west], [north, east]]
    
    folium.raster_layers.ImageOverlay(
        image=coverage_array,
        bounds=bounds,
        opacity=0.5,
        colormap=lambda x: (1, 1, 0, x),  # Yellow transparent overlay
        name=f"{tech_name} coverage",
    ).add_to(m)

    # 7. Force the camera to frame the grid perfectly
    m.fit_bounds(bounds)

    return m


# ==========================================
# PAGE VIEWS
# ==========================================
def page_setup():
    st.header("🌍 Region & Environment Setup")
    
    with st.form("setup_form"):
        col1, col2 = st.columns(2)
        with col1:
            region_name = st.selectbox("Case Study Region", list(REGIONS.keys()))
        with col2:
            # Locked resolution to prevent All-NaN shape mismatch crashes
            st.info("💡 Resolution locked to 250.0m to match native GIS layers.")
            res_m = 250.0
            
        submitted = st.form_submit_button("Load GIS Data")
        if submitted:
            # If region changed, load new region's default sites
            if st.session_state["region_name"] != region_name:
                new_region = get_region(region_name)
                st.session_state["sites"] = load_sites(new_region.sites_path())
            
            st.session_state["region_name"] = region_name
            st.session_state["res_m"] = res_m
            _load_gis_layers()
            st.success(f"Loaded '{region_name}' at 250.0m resolution with default transmitters.")


def page_sites():
    st.header("🗼 Transmitter Network Configuration")
    sites = st.session_state["sites"]
    region = st.session_state.get("region_name", "Unknown")
    
    # 1. Contextual Instructions Based on Region
    if region == "brussels_antwerp":
        st.info(
            "🇧🇪 **Brussels-Antwerp Setup:** The default tutorial sites loaded on boot are synthetic (mapped to Croatia). "
            "To plan a true Belgian network, clear the synthetic sites below and add new transmitters using standard "
            "Belgian EPSG:3035 coordinates."
        )
        default_x = 3927000.0
        default_y = 3082000.0
    else:
        st.info("ℹ️ **Synthetic Setup:** Ensure your transmitter coordinates match your selected region's bounding box.")
        default_x = 4750000.0
        default_y = 2300000.0

    # 2. Render Site Table & Management Controls
    if not sites.sites:
        st.warning("No transmitters currently configured. Add one below to begin your network plan!")
    else:
        df = pd.DataFrame([s.__dict__ for s in sites.sites])
        st.dataframe(df, width="stretch")
        
        if st.button("🗑️ Clear All Synthetic Sites", type="secondary"):
            st.session_state["sites"].sites = []
            st.rerun()

    # 3. Dynamic 'Add Site' Form
    with st.expander("➕ Add New Transmitter", expanded=(not sites.sites)):
        with st.form("add_site"):
            c1, c2, c3 = st.columns(3)
            sid = c1.text_input("ID", f"TX_BEL_{len(sites.sites) + 1}")
            name = c2.text_input("Name", "New Core TX")
            erp = c3.number_input("ERP (kW)", value=100.0)
            
            c4, c5, c6 = st.columns(3)
            x = c4.number_input("Easting (EPSG:3035)", value=default_x)
            y = c5.number_input("Northing (EPSG:3035)", value=default_y)
            height = c6.number_input("Antenna Height (m)", value=100.0)

            # Pilot Toggle (Defaults to True if it is the very first site added)
            is_first_site = len(sites.sites) == 0
            is_pilot = st.checkbox("👑 Set as Pilot Transmitter (Network Master Reference)", value=is_first_site)
            
            if st.form_submit_button("Add Site to Network"):
                
                # SFN Safety Rule: Auto-demote other pilots
                if is_pilot:
                    for s in st.session_state["sites"].sites:
                        s.is_pilot = False

                # Explicit kwargs to avoid TypeError
                new_site = Site(
                    id=sid, 
                    name=name, 
                    x=float(x), 
                    y=float(y), 
                    erp_kw=float(erp), 
                    antenna_height_m=float(height), 
                    topology="HPHT"
                )
                new_site.is_pilot = is_pilot
                
                st.session_state["sites"].sites.append(new_site)
                st.rerun()


def page_run():
    st.header("⚡ SFN Simulation & KPI Analysis")
    
    # 1. DVB-T Config Block
    st.subheader("DVB-T Profile")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        mod = st.selectbox("Modulation", list(MODULATION_CN.keys()), index=5)
    with d_col2:
        gi = st.selectbox("Guard Interval", list(GUARD_INTERVAL_US.keys()), index=0)
    dvb_config = DvbTConfig(modulation=mod, guard_interval=gi)

    # 2. 5G Broadcast Config Block
    st.subheader("5G Broadcast Profile")
    f_col1, _ = st.columns(2)
    with f_col1:
        num = st.selectbox("Numerology (SCS)", NUMEROLOGY_PRESETS, index=0)
    fiveg_config = FiveGBroadcastConfig(numerology=num)

    # 3. Execution Engine with Logging and Guards
    if st.button("🚀 Run Dual Simulation", type="primary"):
        logger.info("--- STARTING NEW SIMULATION ---")
        
        region = get_region(st.session_state["region_name"])
        sites = st.session_state["sites"]
        
        # SAFETY GUARD: Check if sites actually exist
        if not sites.sites:
            logger.error("Simulation aborted: No sites in network.")
            st.error("❌ Cannot run simulation: Your network has no transmitters!")
            return
            
        # SAFETY GUARD: Check if sites are geographically inside the map bounds
        minx, miny, maxx, maxy = region.bounds
        for s in sites.sites:
            if not (minx <= s.x <= maxx and miny <= s.y <= maxy):
                err_msg = f"Tower '{s.id}' at ({s.x}, {s.y}) is OUTSIDE the loaded map boundaries ({minx} to {maxx}, {miny} to {maxy})!"
                logger.error(err_msg)
                st.error(f"❌ **Geographic Error:** {err_msg} Please adjust your coordinates on the Sites page.")
                return

        with st.spinner("Processing P.1546 radio propagation and interference matrices..."):
            try:
                logger.info(f"Building grid at {st.session_state['res_m']}m resolution...")
                grid = make_grid(region, resolution_m=st.session_state["res_m"])
                
                layer_grid = st.session_state.get("layer_grid")
                dem, clutter, population = align_layers_to_grid(
                    grid,
                    st.session_state["dem"],
                    st.session_state["clutter"],
                    st.session_state["population"],
                    source_grid=layer_grid,
                )

                logger.info("Assigning SFN delays...")
                sites = assign_sfn_delays(sites)
                st.session_state["sites"] = sites

                logger.info("Running DVB-T Engine...")
                st.session_state["dvb_result"] = run_dvb_t_coverage(
                    sites, grid, dem, clutter, population, dvb_config
                )

                logger.info("Running 5G Broadcast Engine...")
                st.session_state["fiveg_result"] = run_fiveg_broadcast_coverage(
                    sites, grid, dem, clutter, population, fiveg_config
                )
                
                logger.info("--- SIMULATION SUCCESSFUL ---")
                st.success("Simulation Complete! Navigate to 'Compare' to view KPIs.")
                
            except Exception as e:
                logger.exception("CRITICAL ENGINE FAILURE during simulation loop!")
                st.error(f"💥 **Engine Crash:** {str(e)}\n\nCheck your terminal logs for the full traceback.")


def page_compare():
    st.header("📊 Technology Comparison")
    
    if st.session_state.get("dvb_result") is None or st.session_state.get("fiveg_result") is None:
        st.warning("⚠️ No data available. Please run the simulation first.")
        return

    dvb = st.session_state["dvb_result"]
    g5 = st.session_state["fiveg_result"]
    
    df = pd.DataFrame([
        {"Technology": "DVB-T", **dvb.population_stats},
        {"Technology": "5G Broadcast", **g5.population_stats},
    ])
    st.dataframe(df, width="stretch")

    region = get_region(st.session_state["region_name"])
    test_points = list(getattr(region, 'test_points', ())) if getattr(region, 'test_points', ()) else None

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("DVB-T Footprint")
        st_folium(
            make_map(st.session_state["sites"], dvb.service_coverage.astype(float), "DVB-T", test_points=test_points),
            width=500, height=450
        )
    with col2:
        st.subheader("5G Broadcast Footprint")
        st_folium(
            make_map(st.session_state["sites"], g5.service_coverage.astype(float), "5G", test_points=test_points),
            width=500, height=450
        )


def page_opt():
    st.header("🤖 Network Optimizer")
    st.write("Automatically deploy new HPHT transmitters to maximize population coverage.")

    col1, col2 = st.columns(2)
    with col1:
        n_add = st.slider("New sites budget", 1, 10, 3)
    with col2:
        topology = st.selectbox("Site Type", ["HPHT", "MPMT", "LPLT"], index=0)

    if st.button("Start Greedy Optimization Loop", type="primary"):
        with st.spinner("Analyzing candidate grid locations..."):
            region = get_region(st.session_state["region_name"])
            grid = make_grid(region, resolution_m=st.session_state["res_m"])
            
            # Align layers to grid (CRITICAL to match dimensions)
            layer_grid = st.session_state.get("layer_grid")
            dem, clutter, population = align_layers_to_grid(
                grid,
                st.session_state["dem"],
                st.session_state["clutter"],
                st.session_state["population"],
                source_grid=layer_grid,
            )
            
            res = greedy_add_sites(
                region,
                grid,
                dem,
                clutter,
                population,
                st.session_state["sites"],
                n_add=n_add,
                topology=topology,
            )
            st.session_state["opt_result"] = res
            st.session_state["opt_grid"] = grid
            st.session_state["opt_dem"] = dem
            st.session_state["opt_clutter"] = clutter
            st.session_state["opt_population"] = population
            st.session_state["sites"] = res.selected_sites
        
        st.rerun()

    # Display results if they exist (persists across reruns)
    if "opt_result" in st.session_state:
        res = st.session_state["opt_result"]
        st.success(f"Optimization finished! Final Coverage: {res.final_coverage_percent}%")
        st.write("Incremental Population Gains:", res.incremental_coverage)
        
        # Show coverage map after optimization
        st.subheader("📊 Coverage After Optimization")
        with st.spinner("Computing final coverage map..."):
            grid = st.session_state["opt_grid"]
            dem = st.session_state["opt_dem"]
            clutter = st.session_state["opt_clutter"]
            population = st.session_state["opt_population"]
            
            dvb_config = DvbTConfig()
            opt_coverage = run_dvb_t_coverage(
                res.selected_sites, grid, dem, clutter, population, dvb_config
            )
            
            # Display map
            region = get_region(st.session_state["region_name"])
            test_points = list(getattr(region, 'test_points', ())) if getattr(region, 'test_points', ()) else None
            st_folium(
                make_map(res.selected_sites, opt_coverage.service_coverage.astype(float), "DVB-T (Optimized)", test_points=test_points),
                width=1000, height=500
            )


def page_exports():
    st.header("💾 Export Engine")
    
    if not st.session_state.get("dvb_result"):
        st.warning("⚠️ Run simulation first before exporting datasets.")
        return
        
    st.write(f"Outputs are saved to your local `{OUTPUT_DIR}` directory.")
    if st.button("Generate Complete Export Package"):
        with st.spinner("Writing rasters and PDFs..."):
            region = get_region(st.session_state["region_name"])
            grid = make_grid(region, resolution_m=st.session_state["res_m"])
            dvb = st.session_state["dvb_result"]
            sites = st.session_state["sites"]

            export_sites_geojson(sites, OUTPUT_DIR / "sites.geojson")
            export_sfn_table_csv(sites, OUTPUT_DIR / "sfn_delays.csv")
            export_kpi_csv(dvb.population_stats, OUTPUT_DIR / "dvb_kpis.csv")
            export_geotiff(grid, dvb.best_field_dbuv_m, OUTPUT_DIR / "best_field.tif")
            export_geotiff(grid, dvb.service_coverage.astype(np.float32), OUTPUT_DIR / "coverage.tif")
            
            st.success(f"Data successfully exported to `{OUTPUT_DIR}`")


def page_receivers():
    st.header("📡 Test Receiver Points (Coverage Monitoring)")
    
    region = get_region(st.session_state["region_name"])
    st.info(f"ℹ️ These are default monitoring points for {region.name} to track coverage quality.")
    
    test_points = getattr(region, 'test_points', ())
    if not test_points:
        st.warning("No test receivers defined for this region.")
        return
    
    # Display current test points
    st.subheader("Default Test Receivers")
    test_data = []
    for x, y, name in test_points:
        lon, lat = to_display(x, y)
        test_data.append({
            "Name": name,
            "Latitude": f"{lat:.4f}°N",
            "Longitude": f"{lon:.4f}°E",
            "X (EPSG:3035)": f"{x:.0f}",
            "Y (EPSG:3035)": f"{y:.0f}",
        })
    
    df_receivers = pd.DataFrame(test_data)
    st.dataframe(df_receivers, width="stretch", use_container_width=True)
    
    st.write(f"**Total test points:** {len(test_points)}")
    st.caption("These receivers are displayed as 🟢 green markers on the coverage maps in the 'Compare' page.")


# ==========================================
# MAIN ROUTER
# ==========================================
def main():
    st.set_page_config(page_title="EU Broadcast Planner", layout="wide", page_icon="📡")
    
    # Enforce safe state initialization
    init_session_state()

    st.sidebar.title("📡 Broadcast Planner")
    st.sidebar.markdown("---")
    
    # Navigation mapping
    pages = {
        "Setup Region": page_setup,
        "Sites": page_sites,
        "Receivers": page_receivers,
        "Simulate": page_run,
        "Compare": page_compare,
        "Optimize": page_opt,
        "Export": page_exports,
    }
    
    choice = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Execute selected page
    pages[choice]()


if __name__ == "__main__":
    main()