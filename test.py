from datetime import datetime
from geopy import Nominatim
from timezonefinder import TimezoneFinder
from pytz import timezone, utc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from skyfield.api import load, wgs84, Star
from skyfield.data import hipparcos
import threading
import queue

import utils.physics as physics
import utils.radio as radio
import style.particle as particle

import matplotlib as mpl
import matplotlib.font_manager as fm
import random
import time
import matplotlib.patheffects as pe

plt.style.use('style/gruvbox.mplstyle')
available = {f.name for f in fm.fontManager.ttflist}
preferred_serif = ['Palatino', 'DejaVu Serif', 'Times New Roman']
preferred_sans = ['DejaVu Sans', 'Arial', 'Liberation Sans']


def pick_font(preferred_list):
    for name in preferred_list:
        if name in available:
            return name
    return None


font_choice = pick_font(preferred_serif) or pick_font(preferred_sans) or (
    next(iter(available)) if available else 'sans-serif')
mpl.rcParams['font.family'] = font_choice
if font_choice in preferred_serif:
    mpl.rcParams['font.serif'] = [font_choice]
else:
    mpl.rcParams['font.sans-serif'] = [font_choice]


# --- CONFIGURATION ---
LAT = 33.16658302128232
LON = -117.18180825488281
FM_FREQ = 102.7e6
GAIN = 30
SAMPLE_RATE = 2.048e6
TRAIL_MINUTES = 2
TRAIL_POINTS = 10

# Global State
data_queue = queue.Queue()
plot_data = {
    "stars_theta": [], "stars_r": [], "stars_size": [],
    "sats_theta": [], "sats_r": [], "sat_names": [],
    "sats_trails": [], "planets_theta": [], "planets_r": [],
    "planet_trails": [], "planet_names": []
}
meteor_buffer = {
    'meteors': [],
    'flash': False,
    'last_snr': 0
}
planet_scale = {
    'Mercury': 0.38,
    'Venus': 0.95,
    'Mars': 0.53,
    'Jupiter': 11.21,
    'Saturn': 9.45,
    'Uranus': 4.01,
    'Neptune': 3.88
}

stop_event = threading.Event()

# Load Physics Data
eph = load('de421.bsp')
with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)
ts = load.timescale()

# --- PARTICLE SYSTEM CLASS ---


def run_dashboard():
    fig = plt.figure(figsize=(10, 10), facecolor="#282828")
    ax = fig.add_subplot(111, projection='polar', facecolor='#32302f')

    border = plt.Circle((0, 0), 1, color='#504945', fill=True)
    ax.add_patch(border)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rticks([30, 60, 90])
    ax.set_yticklabels([])
    ax.grid(color='#504945', alpha=0.3)

    # Plot Objects
    star_dots = ax.scatter([], [], marker='s', c='white',
                           alpha=0.95, s=[], edgecolors='none', zorder=4)
    star_glow = ax.scatter([], [], marker='s', c='#b8bb26',
                           alpha=0.25, s=[], edgecolors='none', zorder=3)

    trail_collection = LineCollection(
        [], colors='#8ec07c', linewidths=1.25, alpha=0.5, zorder=5)
    ax.add_collection(trail_collection)
    trail_glow = LineCollection(
        [], colors='#fabd2f', linewidths=3, alpha=0.12, zorder=4)
    ax.add_collection(trail_glow)

    sat_glow, = ax.plot([], [], marker='s', linestyle='None',
                        color='#fabd2f', markersize=8, alpha=0.18, zorder=6)
    sat_dots, = ax.plot([], [], marker='s', linestyle='None',
                        color='#8ec07c', markersize=4, alpha=1.0, zorder=7)
    pulse, = ax.plot([], [], color='#fabd2f', linewidth=4, alpha=0, zorder=8)
    sat_labels = []

    planet_glow = ax.scatter([], [], marker='s', c='#fabd2f',
                             alpha=0.2, s=[], edgecolors='none', zorder=6)
    planet_dots = ax.scatter([], [], marker='s', c='#d3869b',
                             alpha=1.0, s=[], edgecolors='none', zorder=7)
    planet_labels = []

    # --- PARTICLE EFFECT OBJECTS ---
    particle_sys = particle.ParticleSystem()
    particles_scat = ax.scatter(
        [], [], marker='s', c='#fb4934', s=[], edgecolors='none', zorder=9)

    def animate(frame):
        # 1. Update Physics (Stars/Sats)
        if len(plot_data['stars_theta']) > 0:
            offsets = np.c_[plot_data['stars_theta'], plot_data['stars_r']]
            star_dots.set_offsets(offsets)
            star_dots.set_sizes(plot_data['stars_size'])
            star_glow.set_offsets(offsets)
            star_glow.set_sizes([max(6, s * 3)
                                for s in plot_data['stars_size']])

        if plot_data['sats_trails']:
            trail_collection.set_segments(plot_data['sats_trails'])
            trail_glow.set_segments(plot_data['sats_trails'])
        else:
            trail_collection.set_segments([])
            trail_glow.set_segments([])

        sat_dots.set_data(plot_data['sats_theta'], plot_data['sats_r'])
        sat_glow.set_data(plot_data['sats_theta'], plot_data['sats_r'])

        for label in sat_labels:
            label.remove()
        sat_labels.clear()
        for theta, r, name in zip(plot_data['sats_theta'], plot_data['sats_r'], plot_data['sat_names']):
            label = ax.text(theta, r - 6, name, color='#fbf1c7',
                            fontsize=4, ha='center')
            label.set_path_effects(
                [pe.withStroke(linewidth=1, foreground='black'), pe.Normal()])
            sat_labels.append(label)

        # --- FIXED PLANET SECTION ---
        if len(plot_data['planets_theta']) > 0:
            planet_dots.set_offsets(
                np.c_[plot_data['planets_theta'], plot_data['planets_r']])
            planet_glow.set_offsets(
                np.c_[plot_data['planets_theta'], plot_data['planets_r']])
            
            p_names = plot_data['planet_names']
            
            sizes = [planet_scale.get(name, 1.0) * 2 for name in p_names]
            
            planet_dots.set_sizes(sizes)
            planet_glow.set_sizes([s * 3 for s in sizes])
        else:
            planet_dots.set_offsets(np.zeros((0, 2)))
            planet_glow.set_offsets(np.zeros((0, 2)))

        for label in planet_labels:
            label.remove()
        planet_labels.clear()
        for theta, r, name in zip(plot_data['planets_theta'], plot_data['planets_r'], plot_data['planet_names']):
            label = ax.text(theta, r - 6, name, color='#fbf1c7',
                            fontsize=4, ha='center')
            label.set_path_effects(
                [pe.withStroke(linewidth=1, foreground='black'), pe.Normal()])
            planet_labels.append(label)

        if meteor_buffer['flash']:
            est_size = meteor_buffer['meteors'][-1]['estimated_size']
            print(
                f"Meteor Detected! Estimated Size: {est_size} m | SNR: {meteor_buffer['last_snr']:.2f} dB")
            particle_sys.explode(meteor_buffer['last_snr'])
            meteor_buffer['flash'] = False  # Reset trigger

        particle_sys.update()
        p_theta, p_r, p_sizes, p_alphas = particle_sys.get_data()

        if len(p_theta) > 0:
            particles_scat.set_offsets(np.c_[p_theta, p_r])
            particles_scat.set_sizes(p_sizes)
            colors = np.zeros((len(p_theta), 4))
            colors[:, 0] = 0.98  # R (fb4934)
            colors[:, 1] = 0.28  # G
            colors[:, 2] = 0.20  # B
            colors[:, 3] = p_alphas  # Alpha channel
            particles_scat.set_color(colors)
        else:
            particles_scat.set_offsets(np.zeros((0, 2)))

        if len(p_theta) > 0:
            pulse.set_data(np.linspace(0, 2*np.pi, 100), [90]*100)
            pulse.set_alpha(max(0, np.mean(p_alphas) * 0.5))
        else:
            pulse.set_alpha(0)

        return star_dots, star_glow, sat_glow, sat_dots, trail_collection, trail_glow, pulse, particles_scat, planet_dots, planet_glow

    # Start Threads
    t1 = threading.Thread(target=physics.update_physics, args=(
        plot_data, ts, eph, stars, LAT, LON, stop_event, TRAIL_MINUTES, TRAIL_POINTS), daemon=True)
    t2 = threading.Thread(target=radio.radio_monitor, args=(
        meteor_buffer, data_queue, stop_event, SAMPLE_RATE, FM_FREQ, GAIN), daemon=True)

    t1.start()
    t2.start()

    # 50ms interval for smoother particle animation (20 FPS)
    ani = FuncAnimation(fig, animate, interval=50, cache_frame_data=False)
    plt.show()

    stop_event.set()
    t1.join(timeout=1)
    t2.join(timeout=1)


if __name__ == "__main__":
    run_dashboard()
