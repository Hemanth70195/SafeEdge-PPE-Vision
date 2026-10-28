"""
SafeEdge PPE Vision - Environmental Hazard Simulator Module.
Core simulation engine and threshold evaluation logic for construction site environmental parameters.

Provides:
- Parameter configuration loading and validation
- Isolated hazard status evaluation (SAFE, WARNING, DANGER)
- Manual reading updates and bounds checking
- Automatic realistic simulation step generation (bounded random walk)
- Overall site hazard classification based on maximum hazard severity

DISCLAIMER: SIMULATED DATA — NOT LIVE SENSOR READINGS.
All values are generated for demonstration and simulation purposes.
"""

import copy
import json
import math
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Standard disclaimer required across all simulated environmental data readings
SIMULATED_DATA_DISCLAIMER_SHORT = "SIMULATED DATA — NOT LIVE SENSOR READINGS"

# Default project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "environmental_thresholds.json"


class HazardStatus(Enum):
    """Hazard status classifications ordered by severity."""
    SAFE = "SAFE"
    WARNING = "WARNING"
    DANGER = "DANGER"

    @property
    def severity(self) -> int:
        """Numeric severity rank: 0 (SAFE) < 1 (WARNING) < 2 (DANGER)."""
        if self == HazardStatus.SAFE:
            return 0
        elif self == HazardStatus.WARNING:
            return 1
        return 2

    @property
    def color_label(self) -> str:
        """Human-readable primary color designation."""
        if self == HazardStatus.SAFE:
            return "GREEN"
        elif self == HazardStatus.WARNING:
            return "ORANGE"
        return "RED"

    @property
    def color_hex(self) -> str:
        """Hexadecimal color string for UI indicators and badges."""
        if self == HazardStatus.SAFE:
            return "#10B981"  # Emerald Green
        elif self == HazardStatus.WARNING:
            return "#F59E0B"  # Amber Orange
        return "#EF4444"  # Crimson Red

    @property
    def background_hex(self) -> str:
        """Light tinted background hex for cards and alerts."""
        if self == HazardStatus.SAFE:
            return "#ECFDF5"
        elif self == HazardStatus.WARNING:
            return "#FFFBEB"
        return "#FEF2F2"

    @property
    def border_hex(self) -> str:
        """Subtle border hex for cards."""
        if self == HazardStatus.SAFE:
            return "#A7F3D0"
        elif self == HazardStatus.WARNING:
            return "#FDE68A"
        return "#FECACA"

    def __lt__(self, other: "HazardStatus") -> bool:
        if not isinstance(other, HazardStatus):
            return NotImplemented
        return self.severity < other.severity

    def __le__(self, other: "HazardStatus") -> bool:
        if not isinstance(other, HazardStatus):
            return NotImplemented
        return self.severity <= other.severity

    def __gt__(self, other: "HazardStatus") -> bool:
        if not isinstance(other, HazardStatus):
            return NotImplemented
        return self.severity > other.severity

    def __ge__(self, other: "HazardStatus") -> bool:
        if not isinstance(other, HazardStatus):
            return NotImplemented
        return self.severity >= other.severity


@dataclass(frozen=True)
class EnvironmentalReading:
    """Immutable snapshot of an evaluated environmental parameter reading."""
    parameter_key: str
    display_name: str
    value: float
    unit: str
    status: HazardStatus
    status_message: str
    min_value: float
    max_value: float
    step: float
    description: str

    @property
    def is_safe(self) -> bool:
        return self.status == HazardStatus.SAFE

    @property
    def is_warning(self) -> bool:
        return self.status == HazardStatus.WARNING

    @property
    def is_danger(self) -> bool:
        return self.status == HazardStatus.DANGER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_key": self.parameter_key,
            "display_name": self.display_name,
            "value": round(self.value, 2),
            "unit": self.unit,
            "status": self.status.value,
            "status_color": self.status.color_label,
            "status_message": self.status_message,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "description": self.description,
            "disclaimer": SIMULATED_DATA_DISCLAIMER_SHORT,
        }


class EnvironmentalConfigError(Exception):
    """Raised when the environmental thresholds configuration is missing, unreadable, or invalid."""
    pass


# Default fallback configuration in the event that external JSON is missing or inaccessible.
BUILTIN_DEFAULT_CONFIG: Dict[str, Any] = {
    "_metadata": {
        "version": "1.0.0",
        "regulatory_disclaimer": (
            "DISCLAIMER: SIMULATED DATA — NOT LIVE SENSOR READINGS. "
            "Default demonstration thresholds. Calibrate per applicable safety regulations."
        ),
    },
    "simulation": {
        "default_mode": "manual",
        "update_interval_seconds": 2.0,
        "random_walk_enabled": True,
    },
    "parameters": {
        "temperature": {
            "display_name": "Temperature",
            "unit": "°C",
            "description": "Ambient site dry-bulb temperature.",
            "min_value": -10.0,
            "max_value": 55.0,
            "step": 0.5,
            "default_value": 24.0,
            "simulation_drift_max": 1.5,
            "thresholds": {
                "type": "range",
                "safe_min": 15.0,
                "safe_max": 30.0,
                "warning_min": 5.0,
                "warning_max": 38.0,
            },
        },
        "humidity": {
            "display_name": "Humidity",
            "unit": "%",
            "description": "Relative atmospheric moisture percentage.",
            "min_value": 0.0,
            "max_value": 100.0,
            "step": 1.0,
            "default_value": 45.0,
            "simulation_drift_max": 3.0,
            "thresholds": {
                "type": "range",
                "safe_min": 30.0,
                "safe_max": 60.0,
                "warning_min": 20.0,
                "warning_max": 80.0,
            },
        },
        "dust": {
            "display_name": "Dust Concentration",
            "unit": "µg/m³",
            "description": "Respirable particulate matter (PM10 / silica dust concentration).",
            "min_value": 0.0,
            "max_value": 500.0,
            "step": 1.0,
            "default_value": 28.0,
            "simulation_drift_max": 10.0,
            "thresholds": {
                "type": "upper",
                "warning_threshold": 50.0,
                "danger_threshold": 150.0,
            },
        },
        "noise": {
            "display_name": "Noise Level",
            "unit": "dB",
            "description": "Acoustic sound pressure level from equipment and construction activities.",
            "min_value": 30.0,
            "max_value": 130.0,
            "step": 0.5,
            "default_value": 65.0,
            "simulation_drift_max": 3.5,
            "thresholds": {
                "type": "upper",
                "warning_threshold": 70.0,
                "danger_threshold": 85.0,
            },
        },
        "gas": {
            "display_name": "Gas Concentration",
            "unit": "ppm",
            "description": "Hazardous airborne gas concentration index (e.g., CO, H2S, flammable vapors).",
            "min_value": 0.0,
            "max_value": 200.0,
            "step": 0.5,
            "default_value": 8.0,
            "simulation_drift_max": 4.0,
            "thresholds": {
                "type": "upper",
                "warning_threshold": 25.0,
                "danger_threshold": 50.0,
            },
        },
    },
}


class EnvironmentalSimulator:
    """
    Simulator engine for construction site environmental hazard monitoring.
    Manages parameter readings, evaluates safety thresholds, generates dynamic simulation ticks,
    and aggregates overall site safety state.
    """

    REQUIRED_PARAMETERS = ["temperature", "humidity", "dust", "noise", "gas"]

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        allow_fallback_defaults: bool = True,
    ):
        """
        Initialize the simulator.

        Args:
            config_path: Optional file path to environmental_thresholds.json.
            config_dict: Optional pre-loaded dictionary configuration.
            allow_fallback_defaults: If True, uses BUILTIN_DEFAULT_CONFIG when config_path is not found.
        """
        self.config_path = Path(config_path) if config_path else None
        self.allow_fallback_defaults = allow_fallback_defaults

        if config_dict is not None:
            self._config = copy.deepcopy(config_dict)
            self._validate_config(self._config)
        elif self.config_path is not None:
            self._config = self._load_config_file(self.config_path)
        else:
            if DEFAULT_CONFIG_PATH.exists():
                self._config = self._load_config_file(DEFAULT_CONFIG_PATH)
            elif Path("config/environmental_thresholds.json").exists():
                self._config = self._load_config_file(Path("config/environmental_thresholds.json"))
            elif allow_fallback_defaults:
                self._config = copy.deepcopy(BUILTIN_DEFAULT_CONFIG)
            else:
                raise EnvironmentalConfigError(f"Configuration file not found at: {DEFAULT_CONFIG_PATH}")

        self.parameters_meta: Dict[str, Dict[str, Any]] = self._config.get("parameters", {})
        self._validate_config(self._config)

        self._readings: Dict[str, float] = {}
        self._is_running: bool = False
        self._is_paused: bool = False
        self._tick_count: int = 0
        from core.hazard_history import HazardHistoryTracker
        self.history = HazardHistoryTracker()

        self.reset()

    def _load_config_file(self, path: Path) -> Dict[str, Any]:
        """Loads and validates a JSON configuration file."""
        if not path.exists():
            if self.allow_fallback_defaults:
                return copy.deepcopy(BUILTIN_DEFAULT_CONFIG)
            raise EnvironmentalConfigError(f"Threshold configuration file does not exist at: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._validate_config(data)
            return data
        except json.JSONDecodeError as e:
            raise EnvironmentalConfigError(f"Failed to parse configuration JSON at {path}: {e}") from e
        except Exception as e:
            if isinstance(e, EnvironmentalConfigError):
                raise
            raise EnvironmentalConfigError(f"Error reading configuration at {path}: {e}") from e

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validates configuration schema, parameters presence, bounds, and thresholds."""
        if not isinstance(config, dict):
            raise EnvironmentalConfigError("Configuration root must be a dictionary.")

        params = config.get("parameters")
        if not isinstance(params, dict):
            raise EnvironmentalConfigError("Configuration must contain a 'parameters' dictionary.")

        for req in self.REQUIRED_PARAMETERS:
            if req not in params:
                raise EnvironmentalConfigError(f"Configuration missing required parameter: '{req}'")

            pdata = params[req]
            for field in ["min_value", "max_value", "default_value", "thresholds", "unit"]:
                if field not in pdata:
                    raise EnvironmentalConfigError(
                        f"Parameter '{req}' is missing required field '{field}'."
                    )

            if pdata["min_value"] >= pdata["max_value"]:
                raise EnvironmentalConfigError(
                    f"Parameter '{req}' has invalid min/max range ({pdata['min_value']} >= {pdata['max_value']})."
                )

            thresh = pdata["thresholds"]
            ttype = thresh.get("type")
            if ttype == "range":
                for tf in ["safe_min", "safe_max", "warning_min", "warning_max"]:
                    if tf not in thresh:
                        raise EnvironmentalConfigError(
                            f"Parameter '{req}' range threshold missing '{tf}'."
                        )
                if not (thresh["warning_min"] <= thresh["safe_min"] < thresh["safe_max"] <= thresh["warning_max"]):
                    raise EnvironmentalConfigError(
                        f"Parameter '{req}' range thresholds must satisfy: warning_min <= safe_min < safe_max <= warning_max."
                    )
            elif ttype == "upper":
                for tf in ["warning_threshold", "danger_threshold"]:
                    if tf not in thresh:
                        raise EnvironmentalConfigError(
                            f"Parameter '{req}' upper threshold missing '{tf}'."
                        )
                if thresh["warning_threshold"] >= thresh["danger_threshold"]:
                    raise EnvironmentalConfigError(
                        f"Parameter '{req}' upper threshold warning must be < danger threshold."
                    )
            else:
                raise EnvironmentalConfigError(
                    f"Parameter '{req}' has unsupported threshold type: '{ttype}'. Expected 'range' or 'upper'."
                )

    @property
    def is_running(self) -> bool:
        """Returns True if the automatic simulation loop is currently actively advancing."""
        return self._is_running

    @property
    def is_paused(self) -> bool:
        """Returns True if the automatic simulation has been started and is currently paused."""
        return self._is_paused

    @property
    def tick_count(self) -> int:
        """Returns the number of elapsed simulation steps since initialization or reset."""
        return self._tick_count

    @property
    def disclaimer(self) -> str:
        """Returns the legal/regulatory simulation notice."""
        return self._config.get("_metadata", {}).get(
            "regulatory_disclaimer",
            "DISCLAIMER: SIMULATED DATA — NOT LIVE SENSOR READINGS. Default demonstration thresholds. Calibrate per applicable safety regulations."
        )

    def get_parameter_keys(self) -> List[str]:
        """Returns the list of monitored parameter identifier keys."""
        return list(self.REQUIRED_PARAMETERS)

    def get_parameter_metadata(self, param_key: str) -> Dict[str, Any]:
        """Returns a copy of the configuration metadata dictionary for the specified parameter."""
        if param_key not in self.parameters_meta:
            raise KeyError(f"Unknown environmental parameter '{param_key}'. Valid: {self.REQUIRED_PARAMETERS}")
        return copy.deepcopy(self.parameters_meta[param_key])

    def get_default_readings(self) -> Dict[str, float]:
        """Returns a dictionary mapping each parameter key to its documented default baseline value."""
        defaults = {}
        for key in self.REQUIRED_PARAMETERS:
            defaults[key] = float(self.parameters_meta[key]["default_value"])
        return defaults

    def reset(self) -> Dict[str, EnvironmentalReading]:
        """
        Restores all parameters to their documented default values and resets simulation state.

        Returns:
            Dictionary of default EnvironmentalReading instances for all 5 parameters.
        """
        self._readings = self.get_default_readings()
        self._is_running = False
        self._is_paused = False
        self._tick_count = 0
        if hasattr(self, "history") and self.history is not None:
            self.history.clear()
            site_ass = self.assess_current_hazards()
            self.history.record_snapshot(self._readings, tick=0, site_assessment=site_ass, mode="manual", force=True)
        return self.get_all_readings()

    def set_reading(self, param_key: str, value: Union[int, float]) -> EnvironmentalReading:
        """
        Updates a single parameter reading manually with clamping to configured physical min/max.
        Does not mutate or reset any other parameter reading.

        Args:
            param_key: The identifier of the parameter (e.g. 'temperature', 'gas').
            value: Numerical reading value.

        Returns:
            The newly evaluated EnvironmentalReading instance.
        """
        if param_key not in self.parameters_meta:
            raise KeyError(
                f"Unknown environmental parameter: '{param_key}'. Available: {self.REQUIRED_PARAMETERS}"
            )

        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            raise ValueError(f"Reading value for '{param_key}' must be a finite number, received: {value}")

        meta = self.parameters_meta[param_key]
        min_val = float(meta["min_value"])
        max_val = float(meta["max_value"])

        clamped_value = max(min_val, min(max_val, float(value)))
        self._readings[param_key] = clamped_value

        if hasattr(self, "history") and self.history is not None:
            site_ass = self.assess_current_hazards()
            self.history.record_snapshot(self._readings, tick=self._tick_count, site_assessment=site_ass, mode="manual")
            self.history.record_events_from_assessment(site_ass, tick=self._tick_count)

        return self.get_reading(param_key)

    def set_all_readings(self, readings: Dict[str, Union[int, float]]) -> Dict[str, EnvironmentalReading]:
        """
        Updates multiple parameter readings in a single call.

        Args:
            readings: Dict mapping parameter keys to numerical values.

        Returns:
            Dictionary of evaluated EnvironmentalReading instances.
        """
        for k, v in readings.items():
            self.set_reading(k, v)
        return self.get_all_readings()

    def evaluate_parameter_status(self, param_key: str, value: float) -> Tuple[HazardStatus, str]:
        """
        Evaluates a single parameter numerical value against its configured thresholds.
        Threshold evaluation logic is completely decoupled from UI display layers.

        Args:
            param_key: Parameter identifier.
            value: Numerical reading value.

        Returns:
            Tuple of (HazardStatus, informative explanation string).
        """
        if param_key not in self.parameters_meta:
            raise KeyError(f"Unknown environmental parameter: '{param_key}'")

        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            raise ValueError(f"Invalid numerical reading for '{param_key}': {value}")

        meta = self.parameters_meta[param_key]
        thresh = meta["thresholds"]
        ttype = thresh["type"]
        unit = meta["unit"]
        dname = meta["display_name"]

        if ttype == "range":
            s_min = thresh["safe_min"]
            s_max = thresh["safe_max"]
            w_min = thresh["warning_min"]
            w_max = thresh["warning_max"]

            if s_min <= value <= s_max:
                return (
                    HazardStatus.SAFE,
                    f"{dname} is within optimal safe range ({s_min} - {s_max} {unit}).",
                )
            elif w_min <= value < s_min:
                return (
                    HazardStatus.WARNING,
                    f"{dname} is low ({value:.1f} {unit}). Cold/dry warning threshold ({w_min} - {s_min} {unit}).",
                )
            elif s_max < value <= w_max:
                return (
                    HazardStatus.WARNING,
                    f"{dname} is elevated ({value:.1f} {unit}). Heat/humidity warning threshold ({s_max} - {w_max} {unit}).",
                )
            elif value < w_min:
                return (
                    HazardStatus.DANGER,
                    f"CRITICAL DANGER: {dname} is critically low ({value:.1f} {unit} < {w_min} {unit}). Extreme cold/dry hazard.",
                )
            else:
                return (
                    HazardStatus.DANGER,
                    f"CRITICAL DANGER: {dname} is dangerously high ({value:.1f} {unit} > {w_max} {unit}). Extreme heat/stress hazard.",
                )

        elif ttype == "upper":
            w_thresh = thresh["warning_threshold"]
            d_thresh = thresh["danger_threshold"]

            if value <= w_thresh:
                return (
                    HazardStatus.SAFE,
                    f"{dname} is within normal safe limits (<= {w_thresh} {unit}).",
                )
            elif value <= d_thresh:
                return (
                    HazardStatus.WARNING,
                    f"{dname} is elevated ({value:.1f} {unit}). Exceeds advisory warning threshold ({w_thresh} {unit}).",
                )
            else:
                return (
                    HazardStatus.DANGER,
                    f"CRITICAL DANGER: {dname} is at hazardous level ({value:.1f} {unit} > {d_thresh} {unit}).",
                )

        raise EnvironmentalConfigError(f"Unsupported threshold evaluation type '{ttype}'")

    def get_reading(self, param_key: str) -> EnvironmentalReading:
        """
        Retrieves the evaluated snapshot of a single parameter.

        Args:
            param_key: Parameter identifier.

        Returns:
            EnvironmentalReading instance.
        """
        if param_key not in self._readings:
            self._readings[param_key] = float(self.parameters_meta[param_key]["default_value"])

        val = self._readings[param_key]
        meta = self.parameters_meta[param_key]
        status, msg = self.evaluate_parameter_status(param_key, val)

        return EnvironmentalReading(
            parameter_key=param_key,
            display_name=meta["display_name"],
            value=val,
            unit=meta["unit"],
            status=status,
            status_message=msg,
            min_value=float(meta["min_value"]),
            max_value=float(meta["max_value"]),
            step=float(meta["step"]),
            description=meta.get("description", ""),
        )

    def get_all_readings(self) -> Dict[str, EnvironmentalReading]:
        """
        Retrieves evaluated snapshots for all 5 monitored environmental parameters.

        Returns:
            Dictionary mapping parameter keys to EnvironmentalReading instances.
        """
        return {key: self.get_reading(key) for key in self.REQUIRED_PARAMETERS}

    def get_overall_site_status(
        self, readings: Optional[Dict[str, EnvironmentalReading]] = None
    ) -> Tuple[HazardStatus, str, List[str]]:
        """
        Determines the overall site status based on the single most severe active hazard.
        Classification hierarchy:
          - If any parameter is DANGER (RED) -> Overall site status is DANGER.
          - Else if any parameter is WARNING (ORANGE) -> Overall site status is WARNING.
          - Else -> Overall site status is SAFE (GREEN).

        Args:
            readings: Optional pre-evaluated dictionary of readings.

        Returns:
            Tuple of (overall_status: HazardStatus, summary_message: str, active_hazard_names: List[str]).
        """
        if readings is None:
            readings = self.get_all_readings()

        statuses = [r.status for r in readings.values()]
        max_status = max(statuses, key=lambda s: s.severity)

        danger_params = [r.display_name for r in readings.values() if r.status == HazardStatus.DANGER]
        warning_params = [r.display_name for r in readings.values() if r.status == HazardStatus.WARNING]

        if max_status == HazardStatus.DANGER:
            hazards_str = ", ".join(danger_params)
            msg = f"DANGER DETECTED: Critical hazards active for: {hazards_str}. Site safety action required."
            return (HazardStatus.DANGER, msg, danger_params)
        elif max_status == HazardStatus.WARNING:
            hazards_str = ", ".join(warning_params)
            msg = f"WARNING: Elevated levels detected for: {hazards_str}. Caution and monitoring advised."
            return (HazardStatus.WARNING, msg, warning_params)
        else:
            return (HazardStatus.SAFE, "All environmental parameters are within safe operational thresholds.", [])

    def start_auto_simulation(self) -> None:
        """Puts simulator into running auto mode."""
        self._is_running = True
        self._is_paused = False

    def pause_auto_simulation(self) -> None:
        """Freezes automatic simulation at current values."""
        self._is_running = False
        self._is_paused = True

    def resume_auto_simulation(self) -> None:
        """Resumes automatic simulation from paused state."""
        self._is_running = True
        self._is_paused = False

    def stop_auto_simulation(self) -> None:
        """Stops automatic simulation loop."""
        self._is_running = False
        self._is_paused = False

    def step_simulation(self, drift_scale: float = 1.0) -> Dict[str, EnvironmentalReading]:
        """
        Generates one step of bounded, realistic stochastic simulation for all five parameters.
        Uses mean-reverting random walk with occasional realistic spikes.
        All values are strictly clamped to [min_value, max_value].

        Args:
            drift_scale: Multiplier for perturbation magnitude (useful in testing).

        Returns:
            Dictionary of updated EnvironmentalReading snapshots.
        """
        self._tick_count += 1

        for key in self.REQUIRED_PARAMETERS:
            meta = self.parameters_meta[key]
            curr_val = self._readings[key]
            default_val = float(meta["default_value"])
            drift_max = float(meta.get("simulation_drift_max", 2.0)) * max(0.1, drift_scale)
            min_val = float(meta["min_value"])
            max_val = float(meta["max_value"])
            step = float(meta.get("step", 0.5))

            # Mean-reverting drift pull toward baseline default
            mean_reversion_pull = 0.08 * (default_val - curr_val)

            # Gaussian perturbation
            random_delta = random.gauss(0, drift_max * 0.5)

            # Occasional transient spike simulating heavy machinery operation or dust plume
            spike = 0.0
            if random.random() < 0.06:
                spike_direction = 1 if key in ["dust", "noise", "gas"] else random.choice([-1, 1])
                spike = spike_direction * drift_max * 1.8

            new_val = curr_val + mean_reversion_pull + random_delta + spike

            # Clamp strictly to configured physical ranges
            new_val = max(min_val, min(max_val, new_val))

            # Round according to parameter step precision
            precision = 1 if step < 1.0 else 0
            if precision > 0:
                new_val = round(new_val, precision)
            else:
                new_val = round(new_val)

            self._readings[key] = float(new_val)

        if hasattr(self, "history") and self.history is not None:
            site_ass = self.assess_current_hazards()
            self.history.record_snapshot(self._readings, tick=self._tick_count, site_assessment=site_ass, mode="manual")
            self.history.record_events_from_assessment(site_ass, tick=self._tick_count)

        return self.get_all_readings()

    def get_simulation_state(self) -> Dict[str, Any]:
        """
        Returns full diagnostic state snapshot for telemetry, persistence, or UI serialization.
        """
        readings = self.get_all_readings()
        overall_status, summary_msg, active_hazards = self.get_overall_site_status(readings)

        return {
            "is_running": self._is_running,
            "is_paused": self._is_paused,
            "tick_count": self._tick_count,
            "overall_status": overall_status.value,
            "overall_status_color": overall_status.color_label,
            "overall_message": summary_msg,
            "active_hazards": active_hazards,
            "disclaimer": SIMULATED_DATA_DISCLAIMER_SHORT,
            "readings": {k: r.to_dict() for k, r in readings.items()},
        }

    def assess_current_hazards(self) -> Any:
        """
        Evaluates full site safety recommendations and hazard assessments using HazardAssessmentEngine.
        Returns a SiteHazardAssessment with prioritized active hazards, root causes, and recommended actions.
        """
        from core.hazard_assessment import HazardAssessmentEngine
        readings = self.get_all_readings()
        return HazardAssessmentEngine.assess_site(readings, self.parameters_meta)

    def assess_parameter(self, param_key: str) -> Any:
        """
        Evaluates a specific parameter using the HazardAssessmentEngine.
        Returns a HazardAssessment instance.
        """
        from core.hazard_assessment import HazardAssessmentEngine
        val = self._readings.get(param_key)
        meta = self.get_parameter_metadata(param_key)
        return HazardAssessmentEngine.assess_parameter(param_key, val, meta)

    def get_history_tracker(self) -> Any:
        """Returns the session HazardHistoryTracker instance."""
        return getattr(self, "history", None)

    def get_history_snapshots_df(self) -> Any:
        """Returns session history snapshots as a pandas DataFrame."""
        if hasattr(self, "history") and self.history is not None:
            return self.history.get_snapshots_dataframe()
        import pandas as pd
        return pd.DataFrame()

    def get_hazard_events_df(self) -> Any:
        """Returns recorded session hazard events as a pandas DataFrame."""
        if hasattr(self, "history") and self.history is not None:
            return self.history.get_events_dataframe()
        import pandas as pd
        return pd.DataFrame()

    def export_readings_csv(self) -> str:
        """Exports session historical readings as a standardized CSV string."""
        if hasattr(self, "history") and self.history is not None:
            return self.history.export_readings_csv()
        return ""

    def export_events_csv(self) -> str:
        """Exports session hazard state-transition events as a standardized CSV string."""
        if hasattr(self, "history") and self.history is not None:
            return self.history.export_events_csv()
        return ""

    def export_json_report(self, additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Exports a complete structured audit report as a Python dictionary."""
        if hasattr(self, "history") and self.history is not None:
            return self.history.export_json_report(
                configured_thresholds=self.parameters_meta,
                additional_metadata=additional_metadata,
            )
        return {}

    def export_json_report_str(self, additional_metadata: Optional[Dict[str, Any]] = None, indent: int = 2) -> str:
        """Exports a complete structured audit report as a formatted JSON string."""
        if hasattr(self, "history") and self.history is not None:
            return self.history.export_json_report_str(
                configured_thresholds=self.parameters_meta,
                additional_metadata=additional_metadata,
                indent=indent,
            )
        return "{}"

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculates session summary statistics across all parameters and events."""
        if hasattr(self, "history") and self.history is not None:
            return self.history.get_summary_statistics()
        return {}
