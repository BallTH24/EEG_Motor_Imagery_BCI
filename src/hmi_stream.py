"""
Continuous Asynchronous HMI Streaming Engine (Phase 3)
Brain Code Camp (BCC) 2026 - Track: Neurotechnology & Computational Neuroscience

Implements a pseudo-online streaming pipeline with:
- Circular Ring Buffer for sliding temporal windowing (2.0s window, 100ms step).
- 2-Stage Hierarchical Decoder:
    Stage 1: Intent Gatekeeper (posterior probability thresholding for IDLE rejection).
    Stage 2: Directional Decoder (CSP with Ledoit-Wolf shrinkage + LDA).
- Stage 3: Temporal Dwell Debouncing (K=3 agreeing windows) + 1.0s Refractory Lockout.
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np


class CircularRingBuffer:
    """
    Fixed-size FIFO Ring Buffer for continuous multi-channel EEG streaming.
    Maintains a rolling window of T samples across C channels.
    """

    def __init__(self, n_channels: int = 22, window_samples: int = 500) -> None:
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.buffer = np.zeros((n_channels, window_samples), dtype=np.float64)
        self.total_samples_pushed = 0

    def push(self, chunk: np.ndarray) -> None:
        """
        Pushes a new chunk of shape (n_channels, n_samples) into the circular buffer.
        """
        n_ch, n_samples = chunk.shape
        if n_ch != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {n_ch}")

        if n_samples >= self.window_samples:
            # Chunk is larger than or equal to buffer; take the most recent window
            self.buffer = chunk[:, -self.window_samples:].astype(np.float64)
        else:
            # Shift buffer left and append new samples
            self.buffer[:, :-n_samples] = self.buffer[:, n_samples:]
            self.buffer[:, -n_samples:] = chunk.astype(np.float64)

        self.total_samples_pushed += n_samples

    def is_full(self) -> bool:
        """Returns True if the buffer contains at least window_samples samples."""
        return self.total_samples_pushed >= self.window_samples

    def get_window(self) -> np.ndarray:
        """
        Returns the current window of shape (n_channels, window_samples).
        """
        return self.buffer.copy()

    def reset(self) -> None:
        """Resets buffer contents and sample counter."""
        self.buffer.fill(0.0)
        self.total_samples_pushed = 0


class ContinuousHMIEngine:
    """
    Real-Time Asynchronous HMI State Machine and Decision Engine.
    Executes a 2-stage hierarchical decoding pipeline with temporal dwell debouncing.
    """

    def __init__(
        self,
        stage1_gatekeeper: Any,
        stage2_decoder: Any,
        class_labels: List[str],
        sfreq: float = 250.0,
        window_sec: float = 2.0,
        step_sec: float = 0.1,
        prob_threshold: float = 0.50,
        dwell_count: int = 3,
        refractory_sec: float = 1.0,
    ) -> None:
        """
        Parameters:
        - stage1_gatekeeper: Binary classifier (0=REST, 1=ACTIVE).
        - stage2_decoder: Multiclass decoder for direction (e.g. 4-class CSP-LDA).
        - class_labels: List of class names, e.g., ['left_hand', 'right_hand', 'feet', 'tongue'].
        - sfreq: Sampling frequency in Hz (default 250.0).
        - window_sec: Sliding window length in seconds (default 2.0s = 500 samples).
        - step_sec: Step update interval in seconds (default 0.1s = 25 samples).
        - prob_threshold: Minimum ACTIVE probability for Gatekeeper (default 0.50).
        - dwell_count: Number of consecutive agreeing windows required to trigger (default 3 = 300ms).
        - refractory_sec: Lockout refractory duration post-trigger (default 1.0s = 10 steps).
        """
        self.stage1 = stage1_gatekeeper
        self.stage2 = stage2_decoder
        self.class_labels = class_labels
        self.sfreq = sfreq
        self.window_samples = int(round(window_sec * sfreq))
        self.step_samples = int(round(step_sec * sfreq))
        self.prob_threshold = prob_threshold
        self.dwell_count = dwell_count
        self.refractory_steps = int(round(refractory_sec / step_sec))

        self.ring_buffer = CircularRingBuffer(
            n_channels=22, window_samples=self.window_samples
        )

        # State tracking
        self.dwell_counter = 0
        self.dwell_class: Optional[str] = None
        self.refractory_counter = 0

    def reset(self) -> None:
        """Resets internal state machine and ring buffer."""
        self.ring_buffer.reset()
        self.dwell_counter = 0
        self.dwell_class = None
        self.refractory_counter = 0

    def step(
        self, chunk: np.ndarray, current_timestamp: float = 0.0
    ) -> Dict[str, Any]:
        """
        Ingests a new step chunk (22, step_samples), runs hierarchical decoding,
        and evaluates temporal debouncing state.

        Returns a dictionary containing step diagnostics and event flags.
        """
        t0 = time.perf_counter()

        self.ring_buffer.push(chunk)

        # Check if circular buffer is filled
        if not self.ring_buffer.is_full():
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "timestamp": current_timestamp,
                "state": "BUFFERING",
                "pred_class": None,
                "max_prob": 0.0,
                "raw_probs": None,
                "triggered": False,
                "triggered_class": None,
                "dwell_count": 0,
                "refractory_remaining": 0,
                "latency_ms": latency_ms,
            }

        # Retrieve window: shape (1, n_channels, window_samples)
        window = self.ring_buffer.get_window()[np.newaxis, :, :]

        # Stage 1: Intent Gatekeeper (REST vs ACTIVE)
        gate_probs = self.stage1.predict_proba(window)[0]
        prob_active = float(gate_probs[1])

        if prob_active < self.prob_threshold:
            gated_state = "IDLE"
            raw_pred_class = None
            max_prob = prob_active
            probs = np.zeros(len(self.class_labels))
        else:
            # Stage 2: Directional Decoder
            probs = self.stage2.predict_proba(window)[0]
            max_prob = float(np.max(probs))
            argmax_idx = int(np.argmax(probs))
            raw_pred_class = self.class_labels[argmax_idx]
            gated_state = raw_pred_class

        # Decrement refractory counter if active
        if self.refractory_counter > 0:
            self.refractory_counter -= 1

        # Stage 3: Temporal Dwell Debouncing
        triggered = False
        triggered_class = None

        if gated_state != "IDLE":
            if gated_state == self.dwell_class:
                self.dwell_counter += 1
            else:
                self.dwell_class = gated_state
                self.dwell_counter = 1

            # Fire trigger if dwell condition met and not in refractory period
            if (
                self.dwell_counter >= self.dwell_count
                and self.refractory_counter == 0
            ):
                triggered = True
                triggered_class = self.dwell_class
                self.refractory_counter = self.refractory_steps
                self.dwell_counter = 0  # Reset dwell count after firing
        else:
            self.dwell_class = None
            self.dwell_counter = 0

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "timestamp": current_timestamp,
            "state": gated_state,
            "pred_class": raw_pred_class,
            "max_prob": max_prob,
            "raw_probs": probs,
            "triggered": triggered,
            "triggered_class": triggered_class,
            "dwell_count": self.dwell_counter,
            "refractory_remaining": self.refractory_counter,
            "latency_ms": latency_ms,
        }


class StreamSimulator:
    """
    Simulates continuous chronological streaming from raw EEG recordings,
    evaluating asynchronous real-time HMI performance.
    """

    def __init__(self, engine: ContinuousHMIEngine) -> None:
        self.engine = engine

    def simulate_continuous_recording(
        self,
        continuous_data: np.ndarray,
        annotations: List[Dict[str, Any]],
        sfreq: float = 250.0,
    ) -> Dict[str, Any]:
        """
        Streams continuous multi-channel EEG data through the engine.

        Parameters:
        - continuous_data: Array of shape (22, n_samples)
        - annotations: List of dicts with 'onset', 'duration', 'description'
        - sfreq: Sampling frequency (250 Hz)

        Returns:
        - Evaluation summary including onset latencies, FPR in rest, and step execution latencies.
        """
        self.engine.reset()
        step_samples = self.engine.step_samples
        total_samples = continuous_data.shape[1]

        step_records: List[Dict[str, Any]] = []
        triggers: List[Tuple[float, str]] = []

        # Chronological playback loop
        sample_idx = 0
        while sample_idx + step_samples <= total_samples:
            chunk = continuous_data[:, sample_idx : sample_idx + step_samples]
            current_time = sample_idx / sfreq

            res = self.engine.step(chunk, current_timestamp=current_time)
            step_records.append(res)

            if res["triggered"]:
                triggers.append((current_time, res["triggered_class"]))

            sample_idx += step_samples

        # Analyze Triggers vs. Ground Truth Annotations
        metrics = self._compute_hmi_metrics(
            step_records, triggers, annotations, total_duration_sec=total_samples / sfreq
        )
        return metrics

    @staticmethod
    def _compute_hmi_metrics(
        step_records: List[Dict[str, Any]],
        triggers: List[Tuple[float, str]],
        annotations: List[Dict[str, Any]],
        total_duration_sec: float,
    ) -> Dict[str, Any]:
        """Computes onset latencies, FPR during rest, and execution times."""
        # Separate active trial windows from rest intervals
        active_windows: List[Tuple[float, float, str]] = []
        for ann in annotations:
            active_windows.append((ann["onset"], ann["onset"] + ann["duration"], ann["description"]))

        # Sort active windows
        active_windows.sort(key=lambda x: x[0])

        # Compute rest intervals
        rest_intervals: List[Tuple[float, float]] = []
        last_end = 0.0
        for onset, end, _ in active_windows:
            if onset > last_end:
                rest_intervals.append((last_end, onset))
            last_end = max(last_end, end)
        if last_end < total_duration_sec:
            rest_intervals.append((last_end, total_duration_sec))

        total_rest_duration_sec = sum(end - start for start, end in rest_intervals)
        total_rest_duration_min = max(total_rest_duration_sec / 60.0, 1e-6)

        # 1. Onset Latencies for Active Trials
        onset_latencies_sec: List[float] = []
        detected_trials = 0
        correct_detected_trials = 0

        for trial_onset, trial_end, true_label in active_windows:
            # Find first trigger in [trial_onset, trial_end]
            trial_triggers = [
                (t_time, t_label)
                for t_time, t_label in triggers
                if trial_onset <= t_time <= trial_end
            ]
            if trial_triggers:
                detected_trials += 1
                first_trig_time, first_trig_label = trial_triggers[0]
                latency = first_trig_time - trial_onset
                onset_latencies_sec.append(latency)
                if first_trig_label == true_label:
                    correct_detected_trials += 1

        total_trials = len(active_windows)
        trial_detection_rate = (detected_trials / total_trials) * 100.0 if total_trials > 0 else 0.0
        trial_accuracy_rate = (
            (correct_detected_trials / detected_trials) * 100.0 if detected_trials > 0 else 0.0
        )

        mean_onset_latency_ms = (
            float(np.mean(onset_latencies_sec) * 1000.0) if onset_latencies_sec else float("nan")
        )
        median_onset_latency_ms = (
            float(np.median(onset_latencies_sec) * 1000.0) if onset_latencies_sec else float("nan")
        )

        # 2. False Positives in Rest Intervals
        rest_triggers_count = 0
        for t_time, _ in triggers:
            is_in_rest = any(start <= t_time < end for start, end in rest_intervals)
            if is_in_rest:
                rest_triggers_count += 1

        fpr_rest_per_min = rest_triggers_count / total_rest_duration_min

        # 3. Execution Latencies
        step_latencies = [rec["latency_ms"] for rec in step_records if rec["state"] != "BUFFERING"]
        median_step_lat_ms = float(np.median(step_latencies)) if step_latencies else 0.0
        p95_step_lat_ms = float(np.percentile(step_latencies, 95)) if step_latencies else 0.0
        max_step_lat_ms = float(np.max(step_latencies)) if step_latencies else 0.0

        return {
            "total_trials": total_trials,
            "detected_trials": detected_trials,
            "correct_detected_trials": correct_detected_trials,
            "trial_detection_rate_pct": trial_detection_rate,
            "trial_accuracy_rate_pct": trial_accuracy_rate,
            "mean_onset_latency_ms": mean_onset_latency_ms,
            "median_onset_latency_ms": median_onset_latency_ms,
            "rest_triggers_count": rest_triggers_count,
            "total_rest_duration_min": total_rest_duration_min,
            "fpr_rest_per_min": fpr_rest_per_min,
            "step_latency_median_ms": median_step_lat_ms,
            "step_latency_p95_ms": p95_step_lat_ms,
            "step_latency_max_ms": max_step_lat_ms,
            "total_triggers": len(triggers),
            "step_records": step_records,
            "triggers": triggers,
        }


# ==============================================================================
# Phase 4: Cyber-Physical Deployment & Assistive Applications
# Deterministic Binary UART Protocol & ESP32 Watchdog Emulator
# ==============================================================================

CMD_MAP: Dict[str, int] = {
    "IDLE": 0x00,
    "REST": 0x00,
    "left_hand": 0x01,
    "right_hand": 0x02,
    "feet": 0x03,
    "tongue": 0x04,
}

REVERSE_CMD_MAP: Dict[int, str] = {
    0x00: "IDLE",
    0x01: "left_hand",
    0x02: "right_hand",
    0x03: "feet",
    0x04: "tongue",
}


def compute_crc8(data: bytes) -> int:
    """
    Computes CRC-8-CCITT checksum over input bytes.
    Polynomial: x^8 + x^2 + x + 1 (0x07), Init: 0x00, Final XOR: 0x00.
    """
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def encode_uart_frame(cmd_id: int, confidence: float) -> bytes:
    """
    Encodes command and confidence into a deterministic 6-byte binary UART frame.

    Frame Structure:
    [ Byte 0 ] 0xAA      : Frame Synchronization Preamble (High Byte)
    [ Byte 1 ] 0x55      : Frame Synchronization Preamble (Low Byte)
    [ Byte 2 ] CMD_ID    : Actuator Command (0x00=IDLE, 0x01=LEFT, 0x02=RIGHT, 0x03=FEET, 0x04=TONGUE/BRAKE)
    [ Byte 3 ] CONF_BYTE : Quantized Confidence (floor(confidence * 255) in [0, 255])
    [ Byte 4 ] CRC8      : CRC-8 over [CMD_ID, CONF_BYTE] with polynomial 0x07
    [ Byte 5 ] 0x0D      : Carriage Return End-of-Frame Terminator ('\\r')
    """
    cmd = int(cmd_id) & 0xFF
    conf_byte = int(np.clip(np.floor(confidence * 255.0), 0, 255))
    crc = compute_crc8(bytes([cmd, conf_byte]))
    return bytes([0xAA, 0x55, cmd, conf_byte, crc, 0x0D])


class ESP32WatchdogEmulator:
    """
    Hardware-in-the-Loop (HIL) state machine emulator for ESP32 microcontroller receiver.
    Parses continuous UART byte streams, validates CRC-8 integrity, and enforces
    an autonomous fail-safe watchdog timer (500 ms).
    """

    STATE_WAIT_HEADER_1 = 0
    STATE_WAIT_HEADER_2 = 1
    STATE_READ_CMD = 2
    STATE_READ_CONF = 3
    STATE_READ_CRC = 4
    STATE_WAIT_TAIL = 5

    def __init__(self, watchdog_timeout_ms: float = 500.0) -> None:
        self.watchdog_timeout_ms = watchdog_timeout_ms
        self.parser_state = self.STATE_WAIT_HEADER_1
        self.temp_cmd = 0
        self.temp_conf = 0
        self.temp_crc = 0

        self.last_valid_cmd = 0
        self.last_valid_conf = 0.0
        self.last_valid_packet_ms = -float("inf")
        self.actuator_state = 0  # 0x00 = IDLE / BRAKE

        self.valid_frames_received = 0
        self.corrupted_frames_rejected = 0
        self.watchdog_trips = 0

    def reset(self) -> None:
        """Resets emulator parser state and counters."""
        self.parser_state = self.STATE_WAIT_HEADER_1
        self.temp_cmd = 0
        self.temp_conf = 0
        self.temp_crc = 0
        self.last_valid_cmd = 0
        self.last_valid_conf = 0.0
        self.last_valid_packet_ms = -float("inf")
        self.actuator_state = 0
        self.valid_frames_received = 0
        self.corrupted_frames_rejected = 0
        self.watchdog_trips = 0

    def check_watchdog(self, current_time_ms: float) -> bool:
        """
        Evaluates the watchdog timer. If elapsed time reaches or exceeds timeout,
        forces actuators to 0x00 (IDLE/EMERGENCY_BRAKE).
        Returns True if watchdog tripped.
        """
        if self.actuator_state != 0x00:
            if (current_time_ms - self.last_valid_packet_ms) >= self.watchdog_timeout_ms:
                self.actuator_state = 0x00
                self.watchdog_trips += 1
                return True
        return False

    def ingest_byte(self, byte_val: int, current_time_ms: float) -> bool:
        """
        Processes a single incoming byte through the FSM parser.
        Returns True if a valid frame was successfully completed and validated.
        """
        self.check_watchdog(current_time_ms)
        frame_accepted = False

        if self.parser_state == self.STATE_WAIT_HEADER_1:
            if byte_val == 0xAA:
                self.parser_state = self.STATE_WAIT_HEADER_2
        elif self.parser_state == self.STATE_WAIT_HEADER_2:
            if byte_val == 0x55:
                self.parser_state = self.STATE_READ_CMD
            elif byte_val == 0xAA:
                self.parser_state = self.STATE_WAIT_HEADER_2
            else:
                self.parser_state = self.STATE_WAIT_HEADER_1
        elif self.parser_state == self.STATE_READ_CMD:
            self.temp_cmd = byte_val
            self.parser_state = self.STATE_READ_CONF
        elif self.parser_state == self.STATE_READ_CONF:
            self.temp_conf = byte_val
            self.parser_state = self.STATE_READ_CRC
        elif self.parser_state == self.STATE_READ_CRC:
            self.temp_crc = byte_val
            self.parser_state = self.STATE_WAIT_TAIL
        elif self.parser_state == self.STATE_WAIT_TAIL:
            if byte_val == 0x0D:
                expected_crc = compute_crc8(bytes([self.temp_cmd, self.temp_conf]))
                if self.temp_crc == expected_crc:
                    self.last_valid_cmd = self.temp_cmd
                    self.last_valid_conf = self.temp_conf / 255.0
                    self.last_valid_packet_ms = current_time_ms
                    self.actuator_state = self.temp_cmd
                    self.valid_frames_received += 1
                    frame_accepted = True
                else:
                    self.corrupted_frames_rejected += 1
            else:
                self.corrupted_frames_rejected += 1
            self.parser_state = self.STATE_WAIT_HEADER_1

        return frame_accepted

    def ingest_bytes(self, data: bytes, current_time_ms: float) -> int:
        """
        Ingests a stream of bytes. Returns the count of valid frames decoded.
        """
        accepted_count = 0
        for b in data:
            if self.ingest_byte(b, current_time_ms):
                accepted_count += 1
        return accepted_count

    def get_actuator_state(self, current_time_ms: float) -> int:
        """Returns the current fail-safe actuator state after checking watchdog."""
        self.check_watchdog(current_time_ms)
        return self.actuator_state

