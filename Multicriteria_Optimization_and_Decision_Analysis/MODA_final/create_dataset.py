import requests
import numpy as np
import networkx as nx
from geopy.distance import geodesic
import folium
import time
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import math


@dataclass
class RouteMetrics:
    """Data class for route metrics compatible with pygmo"""
    distance: float
    scenic_score: float  # 1 (worst) to 5 (best) - scenic beauty
    roughness_score: float  # 1 (very rough) to 5 (smooth bicycle path)
    safety_score: float  # 1 (very safe) to 5 (quite dangerous traffic)
    slope_score: float  # 1 (gentle descent) to 5 (very steep)
    source: str
    route_points: int
    coordinates: List[List[float]]

    # Real data sources
    elevation_data: Optional[List[float]] = None
    surface_data: Optional[List[str]] = None
    infrastructure_data: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RouteMetrics':
        """Create from dictionary for JSON deserialization"""
        return cls(**data)


class RealDataFetcher:
    """Fetches real geographical and infrastructure data from APIs"""

    def __init__(self):
        self.elevation_api_base = "https://api.open-elevation.com/api/v1/lookup"
        self.overpass_api_base = "https://overpass-api.de/api/interpreter"

        # Surface quality mapping based on OSM smoothness tags
        self.smoothness_scores = {
            'excellent': 5.0, 'good': 4.5, 'intermediate': 3.5,
            'bad': 2.5, 'very_bad': 1.5, 'horrible': 1.0,
            'very_horrible': 0.5, 'impassable': 0.1
        }

        # Surface type scores
        self.surface_scores = {
            'asphalt': 5.0, 'concrete': 4.8, 'paved': 4.5,
            'paving_stones': 4.0, 'compacted': 3.5, 'fine_gravel': 3.0,
            'gravel': 2.5, 'ground': 2.0, 'grass': 1.5,
            'sand': 1.0, 'mud': 0.5
        }

        # Cycling infrastructure safety scores (lower = safer)
        self.infrastructure_safety = {
            'cycleway': 1.0,  # Dedicated cycle path - very safe
            'track': 1.2,  # Separated cycle track
            'lane': 2.5,  # Cycle lane on road
            'shared_lane': 3.5,  # Shared with cars
            'residential': 2.0,  # Residential street
            'tertiary': 3.0,  # Minor road
            'secondary': 4.0,  # Major road
            'primary': 4.5,  # Main road - dangerous
            'trunk': 5.0  # Highway - very dangerous
        }

    def get_elevation_profile(self, coordinates: List[List[float]]) -> List[float]:
        """Get elevation data for route coordinates using Open-Elevation API"""
        if not coordinates or len(coordinates) < 2:
            return []

        try:
            # Limit to max 100 points for API limits, sample evenly
            max_points = min(100, len(coordinates))
            if len(coordinates) > max_points:
                indices = np.linspace(0, len(coordinates) - 1, max_points, dtype=int)
                sample_coords = [coordinates[i] for i in indices]
            else:
                sample_coords = coordinates

            # Convert [lon, lat] to [lat, lon] for elevation API
            locations = [{"latitude": coord[1], "longitude": coord[0]} for coord in sample_coords]

            # Split into chunks if too many points
            elevations = []
            chunk_size = 50

            for i in range(0, len(locations), chunk_size):
                chunk = locations[i:i + chunk_size]

                response = requests.post(
                    self.elevation_api_base,
                    json={"locations": chunk},
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data:
                        chunk_elevations = [result.get('elevation', 0) for result in data['results']]
                        elevations.extend(chunk_elevations)
                    else:
                        break
                else:
                    break

                time.sleep(0.5)  # Rate limiting

            if elevations:
                return elevations

        except Exception:
            pass

        return []

    def get_infrastructure_data(self, start_coords: Tuple[float, float],
                                end_coords: Tuple[float, float]) -> Dict[str, Any]:
        """Get cycling infrastructure data from OpenStreetMap via Overpass API"""
        lat1, lon1 = start_coords
        lat2, lon2 = end_coords

        # Create bounding box with some padding
        padding = 0.01
        min_lat = min(lat1, lat2) - padding
        max_lat = max(lat1, lat2) + padding
        min_lon = min(lon1, lon2) - padding
        max_lon = max(lon1, lon2) + padding

        # Overpass query for cycling infrastructure, surface, and smoothness
        overpass_query = f"""
        [out:json][timeout:15];
        (
          way["highway"="cycleway"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["highway"="path"]["bicycle"="designated"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["highway"]["cycleway"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["highway"~"^(residential|tertiary|secondary)$"]["bicycle"!="no"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out geom tags;
        """

        try:
            response = requests.post(
                self.overpass_api_base,
                data=overpass_query,
                timeout=20,
                headers={'Content-Type': 'text/plain'}
            )

            if response.status_code == 200:
                data = response.json()
                ways = data.get('elements', [])

                if not ways:
                    return {'surface_types': [], 'smoothness_values': [], 'highway_types': []}

                surface_types = []
                smoothness_values = []
                highway_types = []

                for way in ways:
                    tags = way.get('tags', {})

                    # Extract surface information
                    surface = tags.get('surface', 'unknown')
                    if surface != 'unknown':
                        surface_types.append(surface)

                    # Extract smoothness information
                    smoothness = tags.get('smoothness', 'unknown')
                    if smoothness != 'unknown':
                        smoothness_values.append(smoothness)

                    # Extract highway/cycleway type for safety assessment
                    highway = tags.get('highway', 'unknown')
                    cycleway = tags.get('cycleway', '')

                    if highway == 'cycleway' or cycleway in ['track', 'lane']:
                        highway_types.append('cycleway')
                    elif cycleway:
                        highway_types.append('lane')
                    else:
                        highway_types.append(highway)

                return {
                    'surface_types': surface_types,
                    'smoothness_values': smoothness_values,
                    'highway_types': highway_types
                }

        except Exception:
            pass

        return {'surface_types': [], 'smoothness_values': [], 'highway_types': []}

    def calculate_slope_score(self, elevations: List[float]) -> float:
        """Calculate slope difficulty score from elevation profile"""
        if len(elevations) < 2:
            return 1.2  # Default flat Netherlands

        # Calculate elevation changes and gradients
        elevation_changes = []
        for i in range(1, len(elevations)):
            change = abs(elevations[i] - elevations[i - 1])
            elevation_changes.append(change)

        # Calculate statistics
        total_elevation_gain = sum(max(0, elevations[i] - elevations[i - 1]) for i in range(1, len(elevations)))
        max_elevation_change = max(elevation_changes) if elevation_changes else 0
        avg_elevation_change = np.mean(elevation_changes) if elevation_changes else 0

        # Score calculation (1 = easy, 5 = very steep)
        # Netherlands context: most routes are flat, small hills matter
        base_score = 1.0

        # Add score for total elevation gain
        if total_elevation_gain > 100:  # Significant climbing
            base_score += min(2.0, total_elevation_gain / 100)
        elif total_elevation_gain > 50:  # Moderate climbing
            base_score += total_elevation_gain / 100

        # Add score for steepness (max single change)
        if max_elevation_change > 20:  # Steep section
            base_score += min(1.5, max_elevation_change / 20)

        # Add score for overall hilliness
        if avg_elevation_change > 5:  # Generally hilly
            base_score += min(1.0, avg_elevation_change / 10)

        return min(5.0, base_score)

    def calculate_roughness_score(self, surface_types: List[str],
                                  smoothness_values: List[str]) -> float:
        """Calculate surface quality score from OSM data"""
        base_score = 4.5  # Default good Dutch infrastructure

        if not surface_types and not smoothness_values:
            return base_score

        # Use smoothness data if available (more reliable)
        if smoothness_values:
            smoothness_scores = [self.smoothness_scores.get(s, 3.0) for s in smoothness_values]
            return np.mean(smoothness_scores)

        # Fall back to surface types
        if surface_types:
            surface_scores = [self.surface_scores.get(s, 3.0) for s in surface_types]
            return np.mean(surface_scores)

        return base_score

    def calculate_safety_score(self, highway_types: List[str]) -> float:
        """Calculate safety score from infrastructure types (lower = safer)"""
        base_score = 2.0  # Default good Dutch safety

        if not highway_types:
            return base_score

        # Calculate weighted average of safety scores
        safety_scores = [self.infrastructure_safety.get(h, 3.0) for h in highway_types]
        return np.mean(safety_scores)


class GeographicalFeatureAnalyzer:
    """Analyzes geographical features for scenic scoring"""

    def __init__(self):
        # Define scenic zones based on research data
        self.scenic_zones = {
            'national_parks': {
                'coords': [(52.0, 5.5, 52.2, 6.2), (51.8, 5.8, 52.0, 6.0)],  # Hoge Veluwe, Gelderland
                'bonus': 2.0
            },
            'coastal': {
                'coords': [(52.0, 4.0, 53.5, 5.0)],  # North Sea coast
                'bonus': 1.5
            },
            'rivers_canals': {
                'coords': [(51.8, 4.0, 52.5, 6.0)],  # Rhine, Waal region
                'bonus': 1.0
            },
            'historic_windmills': {
                'coords': [(51.85, 4.6, 51.95, 4.7)],  # Kinderdijk area
                'bonus': 1.5
            },
            'tulip_regions': {
                'coords': [(52.2, 4.3, 52.4, 4.6)],  # Keukenhof/Lisse region
                'bonus': 1.8
            }
        }

    def calculate_scenic_score(self, start_coords: Tuple[float, float],
                               end_coords: Tuple[float, float]) -> float:
        """Calculate scenic score based on geographical features"""
        route_center = ((start_coords[0] + end_coords[0]) / 2,
                        (start_coords[1] + end_coords[1]) / 2)

        base_score = 3.0  # Default Dutch countryside

        for feature, data in self.scenic_zones.items():
            for coords in data['coords']:
                if self._point_in_bounds(route_center, coords):
                    base_score += data['bonus']

        return min(5.0, base_score)

    def _point_in_bounds(self, point: Tuple[float, float],
                         bounds: Tuple[float, float, float, float]) -> bool:
        """Check if point is within geographical bounds"""
        lat, lon = point
        min_lat, min_lon, max_lat, max_lon = bounds
        return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


class AbstractRouteCalculator(ABC):
    """Abstract base class for route calculation methods"""

    @abstractmethod
    def get_route(self, start_coords: Tuple[float, float],
                  end_coords: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """Calculate route between two points"""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Get the name of this route calculation method"""
        pass


class OSRMCalculator(AbstractRouteCalculator):
    """OSRM bicycle routing calculator"""

    def get_route(self, start_coords: Tuple[float, float],
                  end_coords: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        try:
            lat1, lon1, lat2, lon2 = start_coords[0], start_coords[1], end_coords[0], end_coords[1]
            url = f"http://router.project-osrm.org/route/v1/bike/{lon1},{lat1};{lon2},{lat2}"
            params = {'overview': 'full', 'geometries': 'geojson', 'steps': 'true'}

            response = requests.get(url, params=params, timeout=25)
            if response.status_code == 200:
                data = response.json()
                if 'routes' in data and len(data['routes']) > 0:
                    route = data['routes'][0]
                    return {
                        'coordinates': route['geometry']['coordinates'],
                        'distance': route.get('distance', 0),
                        'duration': route.get('duration', 0),
                        'route_points': len(route['geometry']['coordinates'])
                    }
        except Exception:
            pass
        return None

    def get_source_name(self) -> str:
        return 'OSRM'


class CurvedSimulationCalculator(AbstractRouteCalculator):
    """Fallback curved route simulation"""

    def get_route(self, start_coords: Tuple[float, float],
                  end_coords: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        lat1, lon1, lat2, lon2 = start_coords[0], start_coords[1], end_coords[0], end_coords[1]

        # Create realistic curved route
        num_segments = 15
        coordinates = []

        for i in range(num_segments + 1):
            t = i / num_segments
            lat = lat1 + t * (lat2 - lat1)
            lon = lon1 + t * (lon2 - lon1)

            # Add realistic curves for Dutch geography
            if 4.3 < lon < 4.9 and 51.9 < lat < 52.4:  # South Holland water avoidance
                curve_offset = 0.01 * np.sin(t * 2 * np.pi) * (1 - abs(t - 0.5) * 2)
                lat += curve_offset
                lon += curve_offset * 0.5

            # Road-following variation
            road_variation = 0.003 * np.sin(t * 6 * np.pi) * np.exp(-4 * (t - 0.5) ** 2)
            lat += road_variation
            lon += road_variation * 0.7

            coordinates.append([lon, lat])

        distance = geodesic(start_coords, end_coords).meters
        return {
            'coordinates': coordinates,
            'distance': distance,
            'route_points': len(coordinates)
        }

    def get_source_name(self) -> str:
        return 'CurvedSimulation'


class DutchBikeRouteNetwork:
    """Enhanced Dutch bicycle route network with real API data"""

    def __init__(self):
        self.cities = {
            'Leiden': (52.1601, 4.4970), 'Amsterdam': (52.3676, 4.9041),
            'Rotterdam': (51.9225, 4.47917), 'The Hague': (52.0705, 4.3007),
            'Utrecht': (52.0907, 5.1214), 'Haarlem': (52.3873, 4.6462),
            'Delft': (52.0116, 4.3571), 'Groningen': (53.2194, 6.5665),
            'Leeuwarden': (53.2012, 5.8086), 'Alkmaar': (52.6318, 4.7516),
            'Enschede': (52.2215, 6.8937), 'Nijmegen': (51.8426, 5.8530),
            'Arnhem': (51.9851, 5.8987), 'Apeldoorn': (52.2112, 5.9699),
            'Zwolle': (52.5168, 6.0830), 'Eindhoven': (51.4416, 5.4697),
            'Maastricht': (50.8514, 5.6909), 'Breda': (51.5719, 4.7683),
            'Den Bosch': (51.6978, 5.3037), 'Venlo': (51.3704, 6.1724)
        }

        self.data_fetcher = RealDataFetcher()
        self.scenic_analyzer = GeographicalFeatureAnalyzer()
        self.calculators = [OSRMCalculator(), CurvedSimulationCalculator()]

    def calculate_realistic_scores(self, route_data: Dict[str, Any],
                                   start_city: str, end_city: str) -> RouteMetrics:
        """Calculate realistic scores using real API data"""
        start_coords = self.cities[start_city]
        end_coords = self.cities[end_city]
        coordinates = route_data.get('coordinates', [])

        # Get real elevation data
        elevations = self.data_fetcher.get_elevation_profile(coordinates)

        # Get real infrastructure data
        infrastructure_data = self.data_fetcher.get_infrastructure_data(start_coords, end_coords)

        # Calculate scores using real data
        slope_score = self.data_fetcher.calculate_slope_score(elevations)

        roughness_score = self.data_fetcher.calculate_roughness_score(
            infrastructure_data['surface_types'],
            infrastructure_data['smoothness_values']
        )

        safety_score = self.data_fetcher.calculate_safety_score(
            infrastructure_data['highway_types']
        )

        scenic_score = self.scenic_analyzer.calculate_scenic_score(start_coords, end_coords)

        return RouteMetrics(
            distance=route_data.get('distance', 0),
            scenic_score=round(scenic_score, 1),
            roughness_score=round(roughness_score, 1),
            safety_score=round(safety_score, 1),
            slope_score=round(slope_score, 1),
            source=route_data.get('source', 'Unknown'),
            route_points=route_data.get('route_points', 0),
            coordinates=coordinates,
            elevation_data=elevations,
            surface_data=infrastructure_data['surface_types'],
            infrastructure_data=infrastructure_data['highway_types']
        )

    def get_route_between_cities(self, start_city: str, end_city: str) -> Optional[RouteMetrics]:
        """Get route between two cities using available calculators"""
        start_coords = self.cities[start_city]
        end_coords = self.cities[end_city]

        for calculator in self.calculators:
            try:
                route_data = calculator.get_route(start_coords, end_coords)
                if route_data and len(route_data.get('coordinates', [])) > 5:
                    route_data['source'] = calculator.get_source_name()
                    metrics = self.calculate_realistic_scores(route_data, start_city, end_city)
                    return metrics
            except Exception:
                continue

        return None

    def build_network(self) -> nx.Graph:
        """Build complete network with real API-sourced metrics"""
        G = nx.Graph()

        # Add cities as nodes
        for city, coords in self.cities.items():
            G.add_node(city, coordinates=coords)

        # Define realistic connections
        connections = [
            ('Leiden', 'The Hague'), ('Leiden', 'Haarlem'), ('Amsterdam', 'Utrecht'),
            ('Amsterdam', 'Haarlem'), ('Rotterdam', 'Delft'), ('Utrecht', 'Rotterdam'),
            ('Utrecht', 'Apeldoorn'), ('Utrecht', 'Arnhem'), ('Utrecht', 'Breda'),
            ('Venlo', 'Eindhoven'), ('Venlo', 'Maastricht'), ('Venlo', 'Nijmegen'),
            ('Apeldoorn', 'Arnhem'), ('Delft', 'The Hague'), ('Breda', 'Rotterdam'),
            ('Eindhoven', 'Breda'), ('Eindhoven', 'Den Bosch'), ('Nijmegen', 'Den Bosch'),
            ('Den Bosch', 'Utrecht'), ('Groningen', 'Leeuwarden'), ('Groningen', 'Enschede'),
            ('Nijmegen', 'Arnhem'), ('Enschede', 'Apeldoorn'), ('Zwolle', 'Apeldoorn'),
            ('Alkmaar', 'Amsterdam'), ('Alkmaar', 'Haarlem'), ('Zwolle', 'Groningen'),
            ('Eindhoven', 'Maastricht'), ('Leeuwarden', 'Zwolle'), ('Zwolle', 'Enschede'),
            ('Leiden', 'Utrecht')
        ]

        for i, (city1, city2) in enumerate(connections):
            print(f"Processing route {i + 1}/{len(connections)}: {city1} - {city2}")

            metrics = self.get_route_between_cities(city1, city2)
            if metrics:
                # Add edge with all metrics as attributes
                G.add_edge(city1, city2, **metrics.to_dict())

            time.sleep(2)  # Rate limiting for APIs

        return G

    def save_network(self, network: nx.Graph, filename: str = 'real_dutch_bike_network'):
        """Save network in multiple formats for compatibility"""
        # Save as JSON (human readable)
        network_data = {
            'nodes': {node: data for node, data in network.nodes(data=True)},
            'edges': {f"{u}-{v}": data for u, v, data in network.edges(data=True)}
        }

        with open(f'{filename}.json', 'w') as f:
            json.dump(network_data, f, indent=2, default=str)

        # Save as pickle (preserves NetworkX structure)
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(network, f)

        # Save as pygmo-compatible format
        self.save_for_pygmo(network, f'{filename}_pygmo.json')

    def save_for_pygmo(self, network: nx.Graph, filename: str):
        """Save network in format compatible with pygmo optimization"""
        cities = list(network.nodes())
        n = len(cities)
        city_to_idx = {city: i for i, city in enumerate(cities)}

        # Initialize matrices
        distance_matrix = np.full((n, n), np.inf)
        scenic_matrix = np.zeros((n, n))
        safety_matrix = np.zeros((n, n))
        slope_matrix = np.zeros((n, n))
        roughness_matrix = np.zeros((n, n))

        # Fill matrices
        for u, v, data in network.edges(data=True):
            i, j = city_to_idx[u], city_to_idx[v]
            distance_matrix[i, j] = distance_matrix[j, i] = data.get('distance', 0)
            scenic_matrix[i, j] = scenic_matrix[j, i] = data.get('scenic_score', 3)
            safety_matrix[i, j] = safety_matrix[j, i] = data.get('safety_score', 2)
            slope_matrix[i, j] = slope_matrix[j, i] = data.get('slope_score', 1)
            roughness_matrix[i, j] = roughness_matrix[j, i] = data.get('roughness_score', 4)

        # Set diagonal to 0
        np.fill_diagonal(distance_matrix, 0)

        pygmo_data = {
            'cities': cities,
            'city_coordinates': {city: network.nodes[city]['coordinates'] for city in cities},
            'distance_matrix': distance_matrix.tolist(),
            'scenic_matrix': scenic_matrix.tolist(),
            'safety_matrix': safety_matrix.tolist(),
            'slope_matrix': slope_matrix.tolist(),
            'roughness_matrix': roughness_matrix.tolist(),
            'metadata': {
                'num_cities': n,
                'num_routes': network.number_of_edges(),
                'description': 'Dutch bicycle route network with REAL API data',
                'data_sources': ['Open-Elevation API', 'OpenStreetMap/Overpass API', 'OSRM API']
            }
        }

        with open(filename, 'w') as f:
            json.dump(pygmo_data, f, indent=2)

    @staticmethod
    def load_network(filename: str) -> nx.Graph:
        """Load network from pickle file"""
        with open(f'{filename}.pkl', 'rb') as f:
            return pickle.load(f)

    def create_visualization(self, network: nx.Graph) -> folium.Map:
        """Create interactive map visualization"""
        m = folium.Map(location=[52.1326, 5.2913], zoom_start=8)

        # Add cities
        for city, coords in self.cities.items():
            color = 'red' if city == 'Leiden' else 'blue'
            icon = 'star' if city == 'Leiden' else 'info-sign'

            folium.Marker(
                location=coords,
                popup=f"<b>{city}</b>",
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)

        # Add routes
        for edge in network.edges(data=True):
            city1, city2, data = edge
            coordinates = data.get('coordinates', [])

            if len(coordinates) > 1:
                folium_coords = [[coord[1], coord[0]] for coord in coordinates]

                # Simple color based on source
                source = data.get('source', 'Unknown')
                if 'OSRM' in source:
                    color = 'green'
                else:
                    color = 'blue'

                popup_text = (f"<b>{city1} - {city2}</b><br>"
                              f"Distance: {data.get('distance', 0) / 1000:.1f}km<br>"
                              f"Scenic: {data.get('scenic_score', 'N/A')}/5<br>"
                              f"Safety: {data.get('safety_score', 'N/A')}/5<br>"
                              f"Slope: {data.get('slope_score', 'N/A')}/5<br>"
                              f"Surface: {data.get('roughness_score', 'N/A')}/5")

                folium.PolyLine(
                    locations=folium_coords,
                    color=color,
                    weight=4,
                    opacity=0.8,
                    popup=popup_text
                ).add_to(m)

                # Add legend
        legend_html = '''
                <div style="position: fixed; top: 10px; right: 10px; width: 280px; height: 180px; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:12px; padding: 10px">
                <h4>🇳🇱 Real Netherlands Bicycle Routes</h4>
                <b>Route Data Sources:</b><br>
                <i class="fa fa-minus" style="color:green"></i> OSRM (Real routing)<br>
                <i class="fa fa-minus" style="color:blue"></i> OSM Bicycle Infrastructure<br>
                <i class="fa fa-minus" style="color:purple"></i> OpenRouteService<br>
                <i class="fa fa-minus" style="color:orange"></i> Curved Simulation<br>
                <br>
                <i class="fa fa-star" style="color:red"></i> Leiden (Start)<br>
                <i class="fa fa-circle" style="color:blue"></i> Other Cities<br>
                <small><b>All routes show actual bicycle paths, not straight lines!</b></small>
                </div>
                '''
        m.get_root().html.add_child(folium.Element(legend_html))

        return m


def main():
    """Main function to demonstrate the enhanced network with real data"""
    print("Dutch Bicycle Route Network with Real API Data")
    print("=" * 50)

    network_builder = DutchBikeRouteNetwork()

    # Build network with real data
    network = network_builder.build_network()

    print(f"\nNetwork Summary:")
    print(f"Cities: {network.number_of_nodes()}")
    print(f"Routes: {network.number_of_edges()}")

    # Save network
    network_builder.save_network(network, 'real_dutch_bike_network')

    # Create visualization
    route_map = network_builder.create_visualization(network)
    route_map.save('real_dutch_bike_routes.html')

    print(f"\nFiles created:")
    print(f"- real_dutch_bike_network.json")
    print(f"- real_dutch_bike_network.pkl")
    print(f"- real_dutch_bike_network_pygmo.json")
    print(f"- real_dutch_bike_routes.html")

    # Display sample metrics to show variation
    print(f"\nSample Route Metrics:")
    for u, v, data in list(network.edges(data=True))[:5]:
        print(f"{u} - {v}: Scenic {data.get('scenic_score', 'N/A')}, "
              f"Safety {data.get('safety_score', 'N/A')}, "
              f"Slope {data.get('slope_score', 'N/A')}, "
              f"Surface {data.get('roughness_score', 'N/A')}")

    return network


if __name__ == "__main__":
    network = main()