"""
SIH26168 - Stage C8: OpenStreetMap (OSM) Network Ingestion and Parsing
Script: src/map/osm_parser.py

Features:
- Fetches raw OSM vector map via official OpenStreetMap API:
  https://api.openstreetmap.org/api/0.6/map?bbox=min_lon,min_lat,max_lon,max_lat
- Automatically partitions large bounding boxes into small sub-tiles to respect
  OSM's 50,000-node query limit, and merges nodes and ways seamlessly.
- Caches locally in `data/raw/maps/{name}.osm` for 100% reproducible offline execution.
- Filters drivable vehicle road ways (motorway, trunk, primary, secondary, tertiary, residential, etc.).
- Converts geodetic (lat, lon) coordinates to local Cartesian East-North-Up (ENU) meters.
- Generates vectorized directed road segment primitives with tangent headings and road metadata.
"""

import os
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import requests
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.orientation import geodetic_to_enu

MAP_CACHE_DIR = REPO_ROOT / "data" / "raw" / "maps"
MAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DRIVABLE_HIGHWAYS = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
    'unclassified', 'residential', 'service',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'
}


def _fetch_single_tile(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> bytes:
    """Fetches a single bounding box tile from OpenStreetMap API."""
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}"
    headers = {'User-Agent': 'SIH26168-IDR-Research/1.0 (dead-reckoning map matching audit; contact: sih@example.com)'}

    for attempt in range(3):
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.content
        elif response.status_code == 429:  # Rate limited
            time.sleep(2.0 * (attempt + 1))
        else:
            time.sleep(1.0)

    raise RuntimeError(f"Failed to fetch tile [{min_lon:.5f},{min_lat:.5f},{max_lon:.5f},{max_lat:.5f}]: HTTP {response.status_code}")


def download_osm_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                      cache_name: str, force_download: bool = False,
                      max_tile_span_deg: float = 0.035) -> Path:
    """
    Downloads OSM vector data for a bounding box from OpenStreetMap API,
    automatically tiling if the area is large to stay below the 50,000 node limit.
    Saves/loads from local cache file.
    """
    cache_path = MAP_CACHE_DIR / f"{cache_name}.osm"
    if cache_path.exists() and not force_download:
        print(f"[OSM Parser] Loaded cached map: {cache_path} ({cache_path.stat().st_size / 1024:.1f} KB)")
        return cache_path

    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    n_lat_tiles = max(1, int(np.ceil(lat_span / max_tile_span_deg)))
    n_lon_tiles = max(1, int(np.ceil(lon_span / max_tile_span_deg)))

    print(f"[OSM Parser] Fetching OSM region '{cache_name}' (BBox: Lat [{min_lat:.4f}, {max_lat:.4f}], Lon [{min_lon:.4f}, {max_lon:.4f}])")
    print(f"[OSM Parser] Grid partitioning: {n_lat_tiles}x{n_lon_tiles} = {n_lat_tiles * n_lon_tiles} tiles...")

    # Dictionary deduplicating nodes and ways
    unique_nodes = {}  # id -> element
    unique_ways = {}   # id -> element

    d_lat = lat_span / n_lat_tiles
    d_lon = lon_span / n_lon_tiles

    tile_count = 0
    total_tiles = n_lat_tiles * n_lon_tiles

    for i in range(n_lat_tiles):
        t_min_lat = min_lat + i * d_lat
        t_max_lat = min_lat + (i + 1) * d_lat
        for j in range(n_lon_tiles):
            tile_count += 1
            t_min_lon = min_lon + j * d_lon
            t_max_lon = min_lon + (j + 1) * d_lon

            print(f"  Fetching tile {tile_count}/{total_tiles} (Lat [{t_min_lat:.4f}, {t_max_lat:.4f}], Lon [{t_min_lon:.4f}, {t_max_lon:.4f}])...")
            tile_bytes = _fetch_single_tile(t_min_lon, t_min_lat, t_max_lon, t_max_lat)

            root = ET.fromstring(tile_bytes)
            for node in root.findall('node'):
                nid = int(node.attrib['id'])
                if nid not in unique_nodes:
                    unique_nodes[nid] = node

            for way in root.findall('way'):
                wid = int(way.attrib['id'])
                if wid not in unique_ways:
                    unique_ways[wid] = way

            time.sleep(0.5)  # Be polite to OSM API

    print(f"[OSM Parser] Merging {len(unique_nodes)} unique nodes and {len(unique_ways)} unique ways...")

    # Build merged XML
    merged_root = ET.Element('osm', version="0.6", generator="SIH26168-IDR-Tiler")
    for node in unique_nodes.values():
        merged_root.append(node)
    for way in unique_ways.values():
        merged_root.append(way)

    merged_tree = ET.ElementTree(merged_root)
    merged_tree.write(cache_path, encoding='utf-8', xml_declaration=True)
    print(f"[OSM Parser] Saved merged map ({cache_path.stat().st_size / 1024:.1f} KB) to {cache_path}")

    return cache_path


def parse_osm_network(osm_file: Path, lat0: float, lon0: float) -> dict:
    """
    Parses an OSM XML file, filters drivable vehicle roads,
    converts coordinates to local ENU relative to (lat0, lon0),
    and builds structured directed segments.
    """
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # 1. Parse all nodes
    nodes = {}
    for node_elem in root.findall('node'):
        node_id = int(node_elem.attrib['id'])
        lat = float(node_elem.attrib['lat'])
        lon = float(node_elem.attrib['lon'])
        nodes[node_id] = (lat, lon)

    # 2. Parse ways and filter drivable roads
    segments = []
    ways_summary = []

    for way_elem in root.findall('way'):
        way_id = int(way_elem.attrib['id'])
        tags = {}
        for tag in way_elem.findall('tag'):
            tags[tag.attrib['k']] = tag.attrib['v']

        highway_type = tags.get('highway', None)
        if highway_type not in DRIVABLE_HIGHWAYS:
            continue

        name = tags.get('name', 'Unnamed Road')
        oneway_tag = tags.get('oneway', 'no')
        is_oneway = oneway_tag in ('yes', '1', 'true') or highway_type in ('motorway', 'motorway_link')
        is_reverse_oneway = oneway_tag == '-1'
        lanes = int(tags.get('lanes', 1)) if tags.get('lanes', '').isdigit() else 1

        nd_refs = [int(nd.attrib['ref']) for nd in way_elem.findall('nd') if int(nd.attrib['ref']) in nodes]
        if len(nd_refs) < 2:
            continue

        ways_summary.append({
            'way_id': way_id,
            'name': name,
            'highway': highway_type,
            'is_oneway': is_oneway,
            'lanes': lanes,
            'node_count': len(nd_refs)
        })

        # Process node pairs into segments
        coords_enu = []
        for nd_id in nd_refs:
            lat, lon = nodes[nd_id]
            e, n, u = geodetic_to_enu(lat, lon, lat0, lon0)
            coords_enu.append((e, n))

        coords_enu = np.array(coords_enu)

        for i in range(len(coords_enu) - 1):
            p1 = coords_enu[i]
            p2 = coords_enu[i + 1]
            diff = p2 - p1
            length = float(np.linalg.norm(diff))
            if length < 0.2:  # Skip degenerate sub-meter node duplicates
                continue

            tangent = diff / length
            # Azimuth angle in radians clockwise from North: atan2(East, North)
            # In navigation ENU: East is X, North is Y -> heading psi = atan2(dx, dy)
            heading_rad = float(np.arctan2(diff[0], diff[1]))

            u_node = nd_refs[i]
            v_node = nd_refs[i + 1]

            # Primary directed segment
            segments.append({
                'way_id': way_id,
                'name': name,
                'highway': highway_type,
                'is_oneway': is_oneway,
                'lanes': lanes,
                'p1': p1,
                'p2': p2,
                'u_node': u_node,
                'v_node': v_node,
                'length': length,
                'tangent': tangent,
                'heading_rad': heading_rad
            })

            # If two-way, add opposite directed segment
            if not is_oneway and not is_reverse_oneway:
                segments.append({
                    'way_id': way_id,
                    'name': name,
                    'highway': highway_type,
                    'is_oneway': False,
                    'lanes': lanes,
                    'p1': p2,
                    'p2': p1,
                    'u_node': v_node,
                    'v_node': u_node,
                    'length': length,
                    'tangent': -tangent,
                    'heading_rad': float(np.arctan2(-diff[0], -diff[1]))
                })

    return {
        'segments': segments,
        'ways': ways_summary,
        'total_segments': len(segments),
        'total_ways': len(ways_summary),
        'lat0': lat0,
        'lon0': lon0
    }
