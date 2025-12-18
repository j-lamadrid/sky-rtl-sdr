from shiny import App, ui, reactive, render
import threading
import time
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend to avoid creating GUI windows from worker threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
try:
    from PIL import Image, ImageFilter, ImageOps
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
from matplotlib.collections import LineCollection

import utils.physics as physics
from skyfield.api import load
from skyfield.data import hipparcos

# Load ephemeris and star catalog once
eph = load('de421.bsp')
with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)
ts = load.timescale()

# Shared state
plot_data = {
    "stars_theta": [], "stars_r": [], "stars_size": [],
    "sats_theta": [], "sats_r": [], "sat_names": [],
    "sats_trails": []
}
stop_event = threading.Event()
worker_thread = None
radio_thread = None
spec_buffer = {"spec": None}

# UI layout
app_ui = ui.page_fluid(
    ui.h2("Sky RTL-SDR — Interactive Dashboard"),
    ui.row(
        ui.column(3,
            ui.panel_well(
                ui.input_text("lat", "Latitude", value="34.05"),
                ui.input_text("lon", "Longitude", value="-118.25"),
                ui.input_text("freq", "FM Frequency (Hz)", value="98.5e6"),
                ui.input_numeric("gain", "Gain", value=40, min=0, max=100),
                ui.input_numeric("srate", "Sample Rate", value=2.048e6, step=1e5),
                ui.input_checkbox("radio_enable", "Enable Radio (dummy)", value=False),
                ui.input_action_button("run", "Run"),
                ui.input_action_button("stop", "Stop"),
                ui.br(),
                ui.p("Trail length (min):"),
                ui.input_numeric("trail_min", "Trail Minutes", value=5, min=1, max=60),
                ui.input_numeric("trail_pts", "Trail Points", value=10, min=2, max=60)
            )
        ),
        ui.column(9,
            ui.output_plot("sky_plot", width="100%", height="420px"),
            ui.h4("Satellites in view"),
            ui.output_table("sat_table"),
            ui.h4("Radio Spectrogram (simulated)"),
            ui.output_plot("spectrogram", width="100%", height="240px")
        )
    )
)

# Helper to start physics worker
def start_worker(lat, lon, trail_minutes, trail_points, radio_enable=False, freq=98.5e6, gain=40, srate=2.048e6):
    global worker_thread, stop_event, radio_thread, spec_buffer
    if worker_thread and worker_thread.is_alive():
        return
    stop_event.clear()
    worker_thread = threading.Thread(
        target=physics.update_physics,
        args=(plot_data, ts, eph, stars, float(lat), float(lon), stop_event, int(trail_minutes), int(trail_points)),
        daemon=True,
    )
    worker_thread.start()

    # start simulated radio worker when requested
    if radio_enable and (radio_thread is None or not radio_thread.is_alive()):
        spec_buffer['spec'] = np.zeros((128, 256))
        radio_thread = threading.Thread(target=_radio_simulator, args=(spec_buffer, stop_event, float(freq), int(gain), float(srate)), daemon=True)
        radio_thread.start()

# Helper to stop worker
def stop_worker():
    global worker_thread, stop_event, radio_thread
    stop_event.set()
    if worker_thread:
        worker_thread.join(timeout=2)
        worker_thread = None
    if radio_thread:
        radio_thread.join(timeout=2)
        radio_thread = None


def _radio_simulator(spec_buffer, stop_event, freq, gain, srate):
    """Simulated radio worker producing a dummy spectrogram until real SDR is available."""
    rng = np.random.default_rng()
    t = 0
    while not stop_event.is_set():
        S = 0.2 * rng.standard_normal((128, 256))
        center = int((np.sin(t / 10.0) * 0.4 + 0.5) * S.shape[1])
        for i in range(S.shape[0]):
            idx = (center + (i % 7) - 3) % S.shape[1]
            S[i, idx] += 6.0 * np.exp(-((i - S.shape[0]/2)**2)/(2*(S.shape[0]/8)**2))
        S = np.abs(S)
        spec_buffer['spec'] = S
        t += 1
        time.sleep(0.5)

# Server
def server(input, output, session):
    # Use reactive.invalidate_later inside render functions for periodic updates

    # Start when Run button is pressed
    @reactive.Effect
    @reactive.event(input.run)
    def _():
        lat = input.lat()
        lon = input.lon()
        trail_min = input.trail_min()
        trail_pts = input.trail_pts()
        radio_enable = input.radio_enable()
        freq = input.freq()
        gain = input.gain()
        srate = input.srate()
        start_worker(lat, lon, trail_min, trail_pts, radio_enable, freq, gain, srate)

    # Stop when Stop pressed or session ends
    @reactive.Effect
    @reactive.event(input.stop)
    def _():
        stop_worker()

    # Note: older/newer shiny versions may not provide a session.on_session_ended hook.
    # The Stop button will stop workers; process shutdown should also terminate daemon threads.

    @output
    @render.plot()
    def sky_plot():
        # refresh every 1000 ms
        reactive.invalidate_later(1000)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='polar', facecolor='navy')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rticks([30, 60, 90])
        ax.set_yticklabels([])
        ax.grid(color='#444444', alpha=0.3)

        # Stars
        if len(plot_data['stars_theta']) > 0:
            offsets = np.c_[plot_data['stars_theta'], plot_data['stars_r']]
            ax.scatter(offsets[:,0], offsets[:,1], c='white', alpha=0.75, s=plot_data['stars_size'], edgecolors='none')

        # Trails
        for seg in plot_data.get('sats_trails', []):
            if len(seg) > 1:
                thetas = seg[:,0]
                rs = seg[:,1]
                ax.plot(thetas, rs, color='cyan', linewidth=1, alpha=0.5)

        # Satellites
        if plot_data['sats_theta']:
            ax.plot(plot_data['sats_theta'], plot_data['sats_r'], 'o', color='gray', markersize=6, alpha=1.0)

        # Labels
        for theta, r, name in zip(plot_data['sats_theta'], plot_data['sats_r'], plot_data['sat_names']):
            ax.text(theta, r - 6, name, color='gray', fontsize=6, ha='center', fontweight='bold')

        # Post-process to give an 8-bit / retro pixelated look if Pillow is available
        if PIL_AVAILABLE:
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')

            # Pixelate: downscale then upscale using nearest neighbor
            pixel_scale = 0.08  # fraction of original size for pixelation
            small_size = (max(8, int(img.width * pixel_scale)), max(8, int(img.height * pixel_scale)))
            img_small = img.resize(small_size, resample=Image.NEAREST)
            img_pix = img_small.resize(img.size, resample=Image.NEAREST)

            # Posterize to reduce color depth (8-bit-ish)
            img_post = ImageOps.posterize(img_pix, bits=3)

            # Add faint scanlines for CRT effect
            scan = Image.new('L', img_post.size, color=0)
            px = scan.load()
            for y in range(0, img_post.size[1], 2):
                for x in range(img_post.size[0]):
                    px[x, y] = 30  # darken every other line slightly
            img_scan = Image.composite(Image.new('RGB', img_post.size, 'black'), img_post, scan)

            # Slight blur to simulate phosphor glow
            img_final = img_scan.filter(ImageFilter.GaussianBlur(radius=0.6))

            # Render the processed image into a new Matplotlib figure
            fig2 = plt.figure(figsize=(6, 6))
            ax2 = fig2.add_subplot(111)
            ax2.imshow(img_final)
            ax2.axis('off')
            return fig2

        return fig

    @output
    @render.table
    def sat_table():
        reactive.invalidate_later(1000)
        names = plot_data['sat_names']
        thetas = plot_data['sats_theta']
        rs = plot_data['sats_r']
        if not names:
            return pd.DataFrame(columns=['name','az_deg','alt_deg'])
        az_deg = [np.degrees(t) for t in thetas]
        alt_deg = [90 - r for r in rs]
        df = pd.DataFrame({
            'name': names,
            'az_deg': az_deg,
            'alt_deg': alt_deg
        })
        return df

    @output
    @render.plot()
    def spectrogram():
        reactive.invalidate_later(1000)
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.set_facecolor('black')
        S = spec_buffer.get('spec')
        if S is None:
            S = np.zeros((128, 256))
        ax.imshow(20 * np.log10(S + 1e-9), aspect='auto', origin='lower', cmap='inferno')
        ax.set_xlabel('Frequency bins')
        ax.set_ylabel('Time')
        plt.tight_layout()
        return fig

app = App(app_ui, server)

if __name__ == '__main__':
    from shiny import run_app
    run_app(app, port=8000)
