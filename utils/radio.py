from rtlsdr import RtlSdr
import numpy as np
from scipy.signal import welch
from scipy.fft import fftshift
from datetime import datetime, timedelta

# --- CONSTANTS ---
METEOR_THRESHOLD_DB = 8
METEOR_MIN_PERSISTENCE = 2
METEOR_MAX_SILENCE = 1


class MeteorScatterEvent:

    def __init__(self, detection_time, peak_power_db, duration_ms, bandwidth_hz,
                 frequency_center_hz, snr_db, estimated_size=None):
        """_summary_

        Args:
            detection_time (_type_): _description_
            peak_power_db (_type_): _description_
            duration_ms (_type_): _description_
            bandwidth_hz (_type_): _description_
            frequency_center_hz (_type_): _description_
            snr_db (_type_): _description_
            estimated_size (_type_, optional): _description_. Defaults to None.
        """
        self.detection_time = detection_time
        self.peak_power_db = peak_power_db
        self.duration_ms = duration_ms
        self.bandwidth_hz = bandwidth_hz
        self.frequency_center_hz = frequency_center_hz
        self.snr_db = snr_db
        self.estimated_size = estimated_size

    def to_dict(self):
        return {
            'time': self.detection_time.isoformat(),
            'peak_power_db': float(self.peak_power_db),
            'duration_ms': float(self.duration_ms),
            'bandwidth_hz': float(self.bandwidth_hz),
            'center_freq_hz': float(self.frequency_center_hz),
            'snr_db': float(self.snr_db),
            'estimated_size': self.estimated_size
        }


def configure_sdr(sample_rate=2.048e6, center_freq=98.5e6, gain=40):
    """_summary_

    Args:
        sample_rate (_type_, optional): _description_. Defaults to 2.048e6.
        center_freq (_type_, optional): _description_. Defaults to 98.5e6.
        gain (int, optional): _description_. Defaults to 40.

    Returns:
        _type_: _description_
    """
    try:
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.center_freq = center_freq
        sdr.gain = gain
        return sdr
    except Exception as e:
        print(f"Error initializing SDR: {e}")
        return None


def close_sdr(sdr):
    if sdr:
        sdr.close()


def compute_psd(samples, sample_rate, nfft=1024):
    """_summary_

    Args:
        samples (_type_): _description_
        sample_rate (_type_): _description_
        nfft (int, optional): _description_. Defaults to 1024.

    Returns:
        _type_: _description_
    """
    f, Pxx = welch(samples, fs=sample_rate,
                   nperseg=nfft, return_onesided=False)
    f = fftshift(f)
    Pxx = fftshift(Pxx)

    return f, Pxx


def _estimate_meteor_size(snr_db, duration_ms):
    """_summary_

    Args:
        snr_db (_type_): _description_
        duration_ms (_type_): _description_

    Returns:
        _type_: _description_
    """
    score = snr_db + (duration_ms / 100)

    if score < 15:
        return 'Tiny'
    if score < 25:
        return 'Small'
    if score < 40:
        return 'Medium'
    return 'Large'


class MeteorScatterAnalyzer:

    def __init__(self, meteor_buffer):
        """_summary_

        Args:
            meteor_buffer (_type_): _description_
        """
        self.meteor_buffer = meteor_buffer
        self.meteor_history = []
        self.max_history = 50

        # State Machine
        self.active_event = False
        self.start_time = None
        self.peak_snr = 0
        self.max_bandwidth = 0
        self.frames_seen = 0
        self.silence_counter = 0

        # Adaptive Baseline Stats ---
        self.avg_snr = 0.0
        self.calibrated = False
        self.calibration_frames = 0
        self.calibration_target = 100

    def update(self, f, Pxx, sample_rate, frame_time_s=None):
        """_summary_

        Args:
            f (_type_): _description_
            Pxx (_type_): _description_
            sample_rate (_type_): _description_
            frame_time_s (_type_, optional): _description_. Defaults to None.
        """
        Pxx_db = 10 * np.log10(Pxx + 1e-12)

        center_idx = len(Pxx_db) // 2
        Pxx_db[center_idx - 5: center_idx + 5] = -120.0

        noise_floor = np.median(Pxx_db)
        peak_val = np.max(Pxx_db)
        raw_snr = peak_val - noise_floor
        if frame_time_s is not None:
            self.last_frame_time = frame_time_s
        else:
            try:
                self.last_frame_time = len(Pxx) / float(sample_rate)
            except Exception:
                self.last_frame_time = 0.1

        # Calibration (Learning Mode)
        if not self.calibrated:
            self.calibration_frames += 1
            # Fast learning (Average the current SNR into the baseline)
            if self.avg_snr == 0:
                self.avg_snr = raw_snr
            else:
                self.avg_snr = 0.9 * self.avg_snr + 0.1 * raw_snr

            if self.calibration_frames > self.calibration_target:
                self.calibrated = True
                print(
                    f"Radio Analyzer Calibrated. Baseline SNR: {self.avg_snr:.2f} dB")
            return

        # Run Detector
        snr_diff = raw_snr - self.avg_snr

        # Slowly update baseline to handle drifting signals (e.g. night vs day)
        if not self.active_event:
            self.avg_snr = 0.98 * self.avg_snr + 0.02 * raw_snr

        signal_present = snr_diff > METEOR_THRESHOLD_DB

        # State Machine Logic
        if signal_present:
            thresh_idx = np.where(
                Pxx_db > (noise_floor + METEOR_THRESHOLD_DB))[0]
            if thresh_idx.size > 0:
                bw = float(np.abs(f[thresh_idx[-1]] - f[thresh_idx[0]]))
                center_freq = float(f[thresh_idx].mean())
            else:
                bw = 0.0
                center_freq = float(f[np.argmax(Pxx_db)])

            if not self.active_event:
                self.active_event = True
                self.start_time = datetime.now()
                self.peak_snr = snr_diff
                self.max_bandwidth = bw
                self.frames_seen = 1
                self.silence_counter = 0
                self.center_freq_start = center_freq
            else:
                self.frames_seen += 1
                self.silence_counter = 0
                old_peak = self.peak_snr
                self.peak_snr = max(self.peak_snr, snr_diff)
                self.max_bandwidth = max(self.max_bandwidth, bw)
        else:
            if self.active_event and self.frames_seen >= METEOR_MIN_PERSISTENCE:
                self._finalize_event()

    def _finalize_event(self):
        """
        _summary_
        """
        if self.frames_seen >= METEOR_MIN_PERSISTENCE:
            if getattr(self, 'last_frame_time', None) and self.last_frame_time > 0:
                duration = self.frames_seen * self.last_frame_time * 1000.0
            else:
                duration = (datetime.now() -
                            self.start_time).total_seconds() * 1000
            est_size = _estimate_meteor_size(self.peak_snr, duration)

            event = MeteorScatterEvent(
                detection_time=self.start_time,
                peak_power_db=self.peak_snr,
                duration_ms=duration,
                bandwidth_hz=self.max_bandwidth,
                frequency_center_hz=self.center_freq_start,
                snr_db=self.peak_snr,
                estimated_size=est_size
            )

            self.meteor_history.append(event)
            if len(self.meteor_history) > self.max_history:
                self.meteor_history.pop(0)

            self.meteor_buffer['meteors'] = [m.to_dict()
                                             for m in self.meteor_history]
            self.meteor_buffer['flash'] = True
            self.meteor_buffer['last_snr'] = self.peak_snr

        self.active_event = False
        self.frames_seen = 0
        self.silence_counter = 0


def radio_monitor(buffer, queue_ref, stop_evt, sample_rate, center_freq, gain):
    # Initialize Hardware
    sdr = configure_sdr(sample_rate=sample_rate,
                        center_freq=center_freq, gain=gain)

    if sdr is None:
        print("SDR Hardware not found. Running in Simulation Mode.")
        return

    # Initialize Analyzer
    analyzer = MeteorScatterAnalyzer(buffer)

    while not stop_evt.is_set():
        try:
            block_len = 256 * 1024
            samples = sdr.read_samples(block_len)

            f, Pxx = compute_psd(samples, sample_rate)
            frame_time_s = float(len(samples)) / float(sample_rate)
            analyzer.update(f, Pxx, sample_rate, frame_time_s=frame_time_s)

            if buffer.get('flash'):
                try:
                    queue_ref.put("FLASH")
                except Exception:
                    pass

        except Exception as e:
            print(f"Radio Error: {e}")
            break

    close_sdr(sdr)
