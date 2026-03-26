"""
File created at: 2026-03-25 17:53:54
Author: Sam Ghalayini
meteor/data/ground_stations.py
"""

from dataclasses import dataclass
from typing import List

@dataclass
class Gateway:
    id : str
    lat : float
    lon : float
    has_server : bool = False
    mu_s : float = 0.0 # CPU cycles / sec

def get_global_gateways() -> List[Gateway]:
    """
    returns subsset of ground stations.
    a subset of these have a server attached with an associated compute
    """
    standard_mu = 10e9

    return [
        # need to add more gateways but this is fine for now
        # North American cities by population
        Gateway("gw_seattle", 47.6062, -122.3321, has_server=True, mu_s=standard_mu),
        Gateway("gw_new_york", 40.7128, -74.0060, has_server=True, mu_s=standard_mu),
        Gateway("gw_texas", 31.9686, -99.9018, has_server=False),
        
        # Europe by famous cities
        Gateway("gw_london", 51.5074, -0.1278, has_server=True, mu_s=standard_mu),
        Gateway("gw_frankfurt", 50.1109, 8.6821, has_server=False),
        
        # Asia & Oceania by famous cities
        Gateway("gw_tokyo", 35.6762, 139.6503, has_server=True, mu_s=standard_mu),
        Gateway("gw_singapore", 1.3521, 103.8198, has_server=False),
        Gateway("gw_sydney", -33.8688, 151.2093, has_server=True, mu_s=standard_mu),
        
        # South America & Africa by famous cities
        Gateway("gw_sao_paulo", -23.5505, -46.6333, has_server=False),
        Gateway("gw_cape_town", -33.9249, 18.4241, has_server=False)
    ]