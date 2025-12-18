import numpy as np
from skyfield.api import load, wgs84, Star
import time
import threading


def update_physics(plot_data, ts, eph, stars, LAT, LON, stop_event=None, trail_minutes=20, trail_points=30):
    earth = eph['earth']

    # 1. Setup Objects
    bright_df = stars[stars['magnitude'] <= 5]
    star_objects = Star.from_dataframe(bright_df)

    # Map common names to DE421 keys
    # Note: We use 'barycenter' for major planets in DE421
    PLANET_LOOKUP = {
        'Mercury': 'mercury barycenter',
        'Venus': 'venus barycenter',
        'Mars': 'mars barycenter',
        'Jupiter': 'jupiter barycenter',
        'Saturn': 'saturn barycenter',
        'Uranus': 'uranus barycenter',
        'Neptune': 'neptune barycenter'
    }

    # 2. Setup Satellites
    try:
        print("Downloading Satellite Orbits...")
        weather_sats = load.tle_file(
            'https://celestrak.org/NORAD/elements/weather.txt')
        station_sats = load.tle_file(
            'https://celestrak.org/NORAD/elements/stations.txt')
        sats = weather_sats + station_sats
    except Exception:
        print("Internet Unavailable. Using Cached TLE Data.")
        sats = []

    # 3. Setup Observer
    my_loc_geo = wgs84.latlon(LAT, LON)
    my_loc_bary = earth + my_loc_geo

    print("Physics Engine Running...")
    if stop_event is None:
        stop_event = threading.Event()

    while not stop_event.is_set():
        t_now = ts.now()

        # --- A. STARS ---
        # Observer (Barycentric) -> Target (Stars)
        astrometric = my_loc_bary.at(t_now).observe(star_objects)
        alt, az, _ = astrometric.apparent().altaz()

        visible_mask = alt.degrees > 0
        s_theta = np.radians(az.degrees[visible_mask])
        s_r = 90.0 - alt.degrees[visible_mask]
        s_size = 20 * 10 ** (bright_df['magnitude'][visible_mask] / -2.5)

        # --- PREPARE TIME VECTOR FOR TRAILS ---
        # 0 to -20 minutes
        offsets = np.linspace(0, -trail_minutes, trail_points) / (24 * 60)
        t_vec = ts.tt_jd(t_now.tt + offsets)

        # --- B. SATELLITES ---
        sat_theta = []
        sat_r = []
        sat_names = []
        sat_trails = []

        for sat in sats:
            # Satellites are Earth-Centered, so we subtract Geo location directly
            diff = sat - my_loc_geo
            topocentric = diff.at(t_vec)
            alt_s, az_s, _ = topocentric.altaz()

            # Current Position (Index 0 is 'now')
            if alt_s.degrees[0] > 0:
                sat_theta.append(np.radians(az_s.degrees[0]))
                sat_r.append(90 - alt_s.degrees[0])
                sat_names.append(sat.name.strip())

                # Trails
                mask = alt_s.degrees > 0
                if np.any(mask):
                    tr_theta = np.radians(az_s.degrees[mask])
                    tr_r = 90 - alt_s.degrees[mask]
                    sat_trails.append(np.column_stack([tr_theta, tr_r]))

        # --- C. PLANETS ---
        planet_theta = []
        planet_r = []
        planet_names_visible = []
        planet_trails = []

        for name, key in PLANET_LOOKUP.items():
            target = eph[key]

            # FIXED: Use 'observe()' instead of subtraction
            # Planets are Barycentric, so we observe them from 'my_loc_bary'
            astrometric = my_loc_bary.at(t_vec).observe(target)
            alt_p, az_p, _ = astrometric.apparent().altaz()

            # Current Position
            if alt_p.degrees[0] > 0:
                planet_theta.append(np.radians(az_p.degrees[0]))
                planet_r.append(90 - alt_p.degrees[0])
                planet_names_visible.append(name)  # Store the visible name

                # Trails
                mask = alt_p.degrees > 0
                if np.any(mask):
                    tr_theta = np.radians(az_p.degrees[mask])
                    tr_r = 90 - alt_p.degrees[mask]
                    planet_trails.append(np.column_stack([tr_theta, tr_r]))

        # --- UPDATE PLOT DATA ---
        plot_data['stars_theta'] = s_theta
        plot_data['stars_r'] = s_r
        plot_data['stars_size'] = s_size

        plot_data['sats_theta'] = sat_theta
        plot_data['sats_r'] = sat_r
        plot_data['sat_names'] = sat_names
        plot_data['sats_trails'] = sat_trails

        plot_data['planets_theta'] = planet_theta
        plot_data['planets_r'] = planet_r
        # Make sure to add this key to plot_data init!
        plot_data['planet_names'] = planet_names_visible
        plot_data['planet_trails'] = planet_trails

        time.sleep(0.2)
