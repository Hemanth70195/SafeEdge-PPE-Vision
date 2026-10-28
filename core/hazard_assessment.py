"""
SafeEdge PPE Vision - Environmental Hazard Assessment & Safety Recommendations Engine.
Phase 2: Decoupled analytical assessment engine for construction site environmental monitoring.

Provides:
- Detailed reasoning explaining why each reading is Safe, Warning, or Danger.
- Parameter-specific recommended protective actions:
  * High / low temperature (cooling, rest, hydration, heat-safety protocols, cold protection)
  * Dust concentration (source control, water misting, ventilation, respiratory protection PPE)
  * Noise level (exposure reduction, acoustic shielding, hearing protection PPE)
  * Gas concentration (emergency procedures, evacuation, calibrated gas detectors)
  * Humidity (evaporative cooling failure, heat index, surface condensation, electrostatic spark risks)
- Multi-hazard prioritization sorted strictly by severity (DANGER > WARNING > SAFE).
- Robust error handling for missing, non-numeric, NaN/infinite, and out-of-range readings.
- Complete decoupling from UI presentation layers.
"""

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from core.environmental_simulator import (
    HazardStatus,
    EnvironmentalReading,
    SIMULATED_DATA_DISCLAIMER_SHORT,
)


@dataclass(frozen=True)
class HazardAssessment:
    """Detailed safety assessment and recommended actions for a single environmental parameter."""
    parameter_key: str
    display_name: str
    value: float
    unit: str
    status: HazardStatus
    reason: str
    recommended_actions: List[str]
    severity: int = field(init=False)

    def __post_init__(self):
        # Assign numeric severity: 0 (SAFE), 1 (WARNING), 2 (DANGER)
        object.__setattr__(self, "severity", self.status.severity)

    @property
    def is_safe(self) -> bool:
        return self.status == HazardStatus.SAFE

    @property
    def is_warning(self) -> bool:
        return self.status == HazardStatus.WARNING

    @property
    def is_danger(self) -> bool:
        return self.status == HazardStatus.DANGER

    @property
    def status_color_label(self) -> str:
        return self.status.color_label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_key": self.parameter_key,
            "display_name": self.display_name,
            "value": round(self.value, 2),
            "unit": self.unit,
            "status": self.status.value,
            "status_color": self.status.color_label,
            "severity": self.severity,
            "reason": self.reason,
            "recommended_actions": list(self.recommended_actions),
            "disclaimer": SIMULATED_DATA_DISCLAIMER_SHORT,
        }


@dataclass
class SiteHazardAssessment:
    """Aggregated construction site safety assessment covering all monitored environmental parameters."""
    overall_status: HazardStatus
    overall_summary: str
    active_hazards: List[HazardAssessment]
    safe_parameters: List[HazardAssessment]
    all_assessments: Dict[str, HazardAssessment]
    disclaimer: str = SIMULATED_DATA_DISCLAIMER_SHORT

    @property
    def has_hazards(self) -> bool:
        return len(self.active_hazards) > 0

    @property
    def danger_count(self) -> int:
        return sum(1 for h in self.active_hazards if h.status == HazardStatus.DANGER)

    @property
    def warning_count(self) -> int:
        return sum(1 for h in self.active_hazards if h.status == HazardStatus.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "overall_status_color": self.overall_status.color_label,
            "overall_summary": self.overall_summary,
            "danger_count": self.danger_count,
            "warning_count": self.warning_count,
            "active_hazards": [h.to_dict() for h in self.active_hazards],
            "safe_parameters": [s.to_dict() for s in self.safe_parameters],
            "all_assessments": {k: v.to_dict() for k, v in self.all_assessments.items()},
            "disclaimer": self.disclaimer,
        }


class HazardAssessmentEngine:
    """
    Decoupled assessment engine evaluating environmental readings against configured thresholds,
    synthesizing root-cause explanations, and formulating regulatory protective recommendations.
    """

    @classmethod
    def evaluate_temperature(
        cls, value: float, meta: Dict[str, Any]
    ) -> Tuple[HazardStatus, str, List[str]]:
        """Formulates diagnostic reason and heat/cold stress actions for temperature."""
        thresh = meta.get("thresholds", {})
        s_min = float(thresh.get("safe_min", 15.0))
        s_max = float(thresh.get("safe_max", 30.0))
        w_min = float(thresh.get("warning_min", 5.0))
        w_max = float(thresh.get("warning_max", 38.0))
        unit = meta.get("unit", "°C")

        if s_min <= value <= s_max:
            reason = (
                f"Ambient temperature ({value:.1f} {unit}) is within optimal safe comfort limits "
                f"({s_min} â {s_max} {unit}). Thermal stress risk is minimal."
            )
            actions = [
                "Maintain standard occupational hydration: ensure drinking water is readily available across all work zones.",
                "Monitor ongoing ambient temperature trends, particularly during peak afternoon solar exposure.",
            ]
            return HazardStatus.SAFE, reason, actions

        elif w_min <= value < s_min:
            reason = (
                f"Ambient temperature ({value:.1f} {unit}) is below optimal comfort ({s_min} {unit}), "
                f"entering the cold warning range ({w_min} â {s_min} {unit}). "
                "Workers face early cold discomfort, reduced manual dexterity, and chilled extremities."
            )
            actions = [
                "Encourage warm layered clothing and insulated, flexible work gloves to maintain tool grip dexterity.",
                "Provide heated, wind-sheltered rest break areas with warm fluids.",
                "Monitor outdoor workers for early symptoms of shivering, numbness, and cold fatigue.",
            ]
            return HazardStatus.WARNING, reason, actions

        elif s_max < value <= w_max:
            reason = (
                f"Elevated ambient temperature ({value:.1f} {unit}) exceeds optimal comfort ({s_max} {unit}), "
                f"entering the heat warning threshold ({s_max} â {w_max} {unit}). "
                "Workers face heightened risk of heat fatigue, dehydration, reduced alertness, and heat exhaustion."
            )
            actions = [
                "Implement scheduled cooling rest cycles in shaded or air-conditioned break areas.",
                "Mandate proactive hydration: workers should drink at least 1 cup (250 ml) of cool water every 20 minutes.",
                "Provide active cooling aids such as shade canopies, misting fans, and neck cooling towels.",
                "Follow site heat-safety procedures: rotate heavy physical tasks, adjust work pace, and monitor unacclimatized workers.",
            ]
            return HazardStatus.WARNING, reason, actions

        elif value < w_min:
            reason = (
                f"CRITICAL HAZARD: Ambient temperature ({value:.1f} {unit}) has fallen into extreme cold "
                f"(< {w_min} {unit}). Severe risk of hypothermia, frostbite, and cold-induced accident susceptibility."
            )
            actions = [
                "Implement severe cold-weather warming protocols with mandatory 15-minute warming breaks every hour.",
                "Require full insulated thermal winter gear, face masks, thermal headwear under hardhats, and insulated boots.",
                "Protect outdoor workstations with temporary windbreaks or radiant heaters where safe and permitted.",
                "Enforce a strict buddy system to monitor personnel for confusion, slurred speech, or frostbite blanching.",
            ]
            return HazardStatus.DANGER, reason, actions

        else:  # value > w_max
            reason = (
                f"CRITICAL HAZARD: Ambient temperature ({value:.1f} {unit}) exceeds hazardous safety threshold "
                f"(> {w_max} {unit}). Severe risk of life-threatening heat stroke, heat collapse, and neurological impairment."
            )
            actions = [
                "Immediately suspend non-essential strenuous physical labor and reschedule high-exertion tasks to cooler hours.",
                "Enforce mandatory cooling breaks (minimum 30 minutes rest per hour) in air-conditioned or chilled rest shelters.",
                "Mandate intensive hydration with cold drinking water and electrolyte-replenishment solutions.",
                "Deploy active site cooling measures (continuous high-capacity misting stations, personal cooling vests).",
                "Execute site heat-safety emergency procedures: verify first-aid cooling supplies and emergency medical transport readiness.",
            ]
            return HazardStatus.DANGER, reason, actions

    @classmethod
    def evaluate_humidity(
        cls, value: float, meta: Dict[str, Any]
    ) -> Tuple[HazardStatus, str, List[str]]:
        """Formulates diagnostic reason and moisture/evaporative cooling actions for humidity."""
        thresh = meta.get("thresholds", {})
        s_min = float(thresh.get("safe_min", 30.0))
        s_max = float(thresh.get("safe_max", 60.0))
        w_min = float(thresh.get("warning_min", 20.0))
        w_max = float(thresh.get("warning_max", 80.0))
        unit = meta.get("unit", "%")

        if s_min <= value <= s_max:
            reason = (
                f"Relative atmospheric humidity ({value:.1f} {unit}) is within optimal operational range "
                f"({s_min} â {s_max} {unit}). Natural evaporative sweat cooling and airborne dust settling operate normally."
            )
            actions = [
                "Maintain standard occupational hydration protocols across all active work areas.",
                "Continue routine surveillance of ambient microclimate trends.",
            ]
            return HazardStatus.SAFE, reason, actions

        elif w_min <= value < s_min:
            reason = (
                f"Low relative humidity ({value:.1f} {unit}) creates arid conditions ({w_min} â {s_min} {unit}). "
                "Causes accelerated moisture loss, dry mucous membranes, eye irritation, static electricity accumulation, "
                "and prolonged airborne particulate suspension."
            )
            actions = [
                "Increase frequency of water-spray dust suppression as arid air causes rapid surface dryout.",
                "Ensure proper electrical grounding and anti-static bonding on flammable fluid transfers to prevent electrostatic sparks.",
                "Provide eye lubricating drops, skin protection, and encourage frequent water consumption.",
            ]
            return HazardStatus.WARNING, reason, actions

        elif s_max < value <= w_max:
            reason = (
                f"Elevated atmospheric humidity ({value:.1f} {unit}) exceeds optimal comfort ({s_max} â {w_max} {unit}). "
                "Significantly compromises natural evaporative sweat cooling, escalates effective heat index, "
                "and promotes condensation on scaffolding, stairs, and smooth surfaces."
            )
            actions = [
                "Increase frequency of rest cycles to compensate for severely impaired natural sweat evaporation.",
                "Inspect metal scaffolding, ladders, and walkways for slippery condensation moisture.",
                "Improve mechanical air movement using high-velocity ventilation blowers to assist convective cooling.",
                "Provide electrolyte-replenishing beverages rather than plain water to prevent hyponatremia from heavy sweating.",
            ]
            return HazardStatus.WARNING, reason, actions

        elif value < w_min:
            reason = (
                f"CRITICAL HAZARD: Extremely low relative humidity ({value:.1f} {unit}) creates high electrostatic discharge "
                f"(ESD) danger (< {w_min} {unit}). Severe spark-ignition risk around combustible vapors, flammable solvents, or fine dusts."
            )
            actions = [
                "Mandate strict electrical grounding and anti-static bonding protocols across all fuel, solvent, and painting operations.",
                "Halt dry abrasive operations and ensure anti-static footwear/clothing in hazardous classified zones.",
                "Deploy continuous water misting to maintain surface moisture and suppress static charge generation.",
                "Provide saline eye drops and ensure personnel maintain high fluid intake to counter extreme respiratory dryness.",
            ]
            return HazardStatus.DANGER, reason, actions

        else:  # value > w_max
            reason = (
                f"CRITICAL HAZARD: Saturated humidity ({value:.1f} {unit}) virtually eliminates the body's evaporative cooling "
                f"(> {w_max} {unit}). Extreme risk of rapid-onset heat stroke even at moderate dry-bulb temperatures, "
                "severe surface condensation, slip-and-fall hazards, and compromised tool grip."
            )
            actions = [
                "Drastically shorten work shifts with mandatory cooling breaks in dehumidified or air-conditioned rest facilities.",
                "Inspect all elevated work surfaces, steel decking, and scaffolding planks; apply anti-slip treatments or grit.",
                "Mandate high-traction footwear and non-slip textured safety gloves for tool and material handling.",
                "Closely monitor personnel for rapid-onset heat exhaustion symptoms (dizziness, nausea, mental confusion).",
            ]
            return HazardStatus.DANGER, reason, actions

    @classmethod
    def evaluate_dust(
        cls, value: float, meta: Dict[str, Any]
    ) -> Tuple[HazardStatus, str, List[str]]:
        """Formulates diagnostic reason and dust suppression/respiratory actions for dust."""
        thresh = meta.get("thresholds", {})
        w_thresh = float(thresh.get("warning_threshold", 50.0))
        d_thresh = float(thresh.get("danger_threshold", 150.0))
        unit = meta.get("unit", "µg/m³")

        if value <= w_thresh:
            reason = (
                f"Respirable particulate dust ({value:.1f} {unit}) is within acceptable baseline limits "
                f"(â¤ {w_thresh} {unit}). Ambient air quality is clear."
            )
            actions = [
                "Maintain standard site housekeeping and routine dust-control practices.",
                "Keep localized dust suppression equipment on standby for upcoming cutting or grinding activities.",
            ]
            return HazardStatus.SAFE, reason, actions

        elif value <= d_thresh:
            reason = (
                f"Elevated respirable particulate concentration ({value:.1f} {unit}) exceeds advisory threshold "
                f"({w_thresh} â {d_thresh} {unit}). Risk of respiratory tract irritation, coughing, "
                "eye discomfort, and reduced site visibility."
            )
            actions = [
                "Control the dust source at point-of-generation using water misting, spray bars, and wet-cutting techniques.",
                "Improve suitable ventilation by positioning portable extraction fans and dust collectors.",
                "Follow site respiratory-protection procedures: advise certified N95 or FFP2 particulate respirators for nearby workers.",
                "Cover loose stockpiles, dampen unpaved haul roads, and enforce site vehicle speed limits to minimize re-entrainment.",
            ]
            return HazardStatus.WARNING, reason, actions

        else:
            reason = (
                f"CRITICAL HAZARD: Respirable dust concentration ({value:.1f} {unit}) exceeds the critical exposure limit "
                f"(> {d_thresh} {unit}). Severe occupational hazard for respirable silica and fine particulate exposure, "
                "risking acute pulmonary inflammation and long-term respiratory disease."
            )
            actions = [
                "Follow site respiratory-protection procedures immediately: mandate certified N95, P100, or PAPR respirators with verified seal checks.",
                "Halt high-dust operations (dry abrasive blasting, uncontrolled masonry cutting) until active controls are functioning.",
                "Aggressively control dust sources using high-pressure fogging cannons, localized vacuum shrouds, and continuous watering.",
                "Isolate and barricade the dust generation zone; restrict access strictly to authorized personnel wearing full respiratory PPE.",
                "Improve suitable ventilation: operate industrial negative-pressure air scrubbers or continuous exhaust dilution.",
            ]
            return HazardStatus.DANGER, reason, actions

    @classmethod
    def evaluate_noise(
        cls, value: float, meta: Dict[str, Any]
    ) -> Tuple[HazardStatus, str, List[str]]:
        """Formulates diagnostic reason and hearing protection actions for noise."""
        thresh = meta.get("thresholds", {})
        w_thresh = float(thresh.get("warning_threshold", 70.0))
        d_thresh = float(thresh.get("danger_threshold", 85.0))
        unit = meta.get("unit", "dB")

        if value <= w_thresh:
            reason = (
                f"Acoustic noise level ({value:.1f} {unit}) is within comfortable, safe operational limits "
                f"(â¤ {w_thresh} {unit}). Verbal communication and auditory comfort are unimpaired."
            )
            actions = [
                "Continue standard work operations; hearing protection is optional in general site areas.",
                "Maintain routine acoustic monitoring around active mechanical equipment zones.",
            ]
            return HazardStatus.SAFE, reason, actions

        elif value <= d_thresh:
            reason = (
                f"Elevated acoustic sound level ({value:.1f} {unit}) is approaching the 85 dB occupational action limit "
                f"({w_thresh} â {d_thresh} {unit}). Prolonged exposure can induce auditory fatigue, elevated stress, "
                "and communication difficulties."
            )
            actions = [
                "Reduce worker exposure duration through task rotation and scheduling high-noise tasks during low-occupancy periods.",
                "Use appropriate hearing protection: provide and advise earplugs or earmuffs with sufficient Noise Reduction Rating (NRR â¥ 20 dB).",
                "Inspect noisy machinery for missing acoustic baffles, loose cowlings, or unlubricated components.",
                "Post cautionary noise warning notices in the vicinity of active mechanical plant.",
            ]
            return HazardStatus.WARNING, reason, actions

        else:
            reason = (
                f"CRITICAL HAZARD: Acoustic noise level ({value:.1f} {unit}) exceeds the hazardous threshold "
                f"(> {d_thresh} {unit}). High risk of permanent noise-induced hearing loss, tinnitus, and auditory masking of emergency warnings."
            )
            actions = [
                "Mandate appropriate hearing protection immediately: require certified high-attenuation earmuffs or dual protection (earmuffs + earplugs).",
                "Reduce exposure strictly: enforce maximum allowable exposure stay-times and mandatory worker rotation away from noise sources.",
                "Implement engineering noise controls: erect sound curtains, acoustic baffles, or temporary acoustic enclosure shields.",
                "Establish and demarcate an exclusive Hearing Protection Zone with high-visibility warning perimeter signage.",
                "Deploy synchronized visual flashing strobe alarms to complement acoustic sirens in high-noise work zones.",
            ]
            return HazardStatus.DANGER, reason, actions

    @classmethod
    def evaluate_gas(
        cls, value: float, meta: Dict[str, Any]
    ) -> Tuple[HazardStatus, str, List[str]]:
        """Formulates diagnostic reason and emergency/evacuation actions for gas."""
        thresh = meta.get("thresholds", {})
        w_thresh = float(thresh.get("warning_threshold", 25.0))
        d_thresh = float(thresh.get("danger_threshold", 50.0))
        unit = meta.get("unit", "ppm")

        if value <= w_thresh:
            reason = (
                f"Hazardous gas concentration index ({value:.1f} {unit}) is within normal atmospheric baseline limits "
                f"(â¤ {w_thresh} {unit}). No airborne toxic or combustible gas hazard detected."
            )
            actions = [
                "Continue continuous gas monitoring, particularly in trenches, basements, and confined spaces.",
                "Ensure fixed and portable atmospheric monitors are calibrated and daily bump-tested.",
            ]
            return HazardStatus.SAFE, reason, actions

        elif value <= d_thresh:
            reason = (
                f"Elevated airborne gas concentration index ({value:.1f} {unit}) exceeds normal baseline "
                f"({w_thresh} â {d_thresh} {unit}). Indicates potential accumulation of vehicle exhaust (CO), "
                "sewer/decomposing vapors (H2S), or volatile chemical fumes."
            )
            actions = [
                "Deploy and use appropriate calibrated gas detectors to identify specific chemical constituents and concentration trends.",
                "Inspect and improve suitable ventilation: activate explosion-proof positive-pressure mechanical ventilation blowers.",
                "Inspect potential emission sources: check fuel lines, generator exhaust routings, welding cylinders, and pipe connections.",
                "Prepare to execute site emergency procedures if gas concentrations continue to rise.",
                "Restrict non-essential personnel entry into the affected zone until verified clear.",
            ]
            return HazardStatus.WARNING, reason, actions

        else:
            reason = (
                f"CRITICAL HAZARD: Hazardous airborne gas concentration ({value:.1f} {unit}) exceeds emergency threshold "
                f"(> {d_thresh} {unit}). Immediate threat of toxic poisoning, asphyxiation, or combustible vapor ignition."
            )
            actions = [
                "Follow site emergency procedures immediately: trigger the hazardous gas alarm protocol across the facility.",
                "Immediately leave the affected area: order complete and rapid evacuation of all personnel upwind to the designated emergency assembly point.",
                "Do NOT re-enter the affected zone without appropriate calibrated gas detectors and positive-pressure Self-Contained Breathing Apparatus (SCBA).",
                "Isolate gas/fuel supply shut-off valves and de-energize potential ignition sources only if safe and certified to do so.",
                "Erect perimeter hazard barricades, prohibit site re-entry, and notify the Site Safety Officer and municipal emergency responders.",
            ]
            return HazardStatus.DANGER, reason, actions

    @classmethod
    def assess_parameter(
        cls,
        param_key: str,
        value: Optional[Union[int, float]],
        param_meta: Dict[str, Any],
    ) -> HazardAssessment:
        """
        Assesses a single parameter reading safely, returning a comprehensive HazardAssessment.
        Handles missing, non-numeric, NaN/infinite, and out-of-range values gracefully without crashing.
        """
        display_name = param_meta.get("display_name", param_key.capitalize())
        unit = param_meta.get("unit", "")

        # Safe handling for missing or invalid values
        if value is None or not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            return HazardAssessment(
                parameter_key=param_key,
                display_name=display_name,
                value=0.0,
                unit=unit,
                status=HazardStatus.WARNING,
                reason=(
                    f"INVALID OR MISSING READING for {display_name} (received: {value!r}). "
                    "Sensor communication fault or telemetry dropout. Treat as precautionary warning."
                ),
                recommended_actions=[
                    f"Check sensor telemetry connection, wiring, or simulation data feed for '{display_name}'.",
                    "Do not assume unmonitored atmospheric conditions are safe; conduct manual verification.",
                    "Inspect data acquisition hardware and recalibrate telemetry interface.",
                ],
            )

        # Clamping to physical range bounds
        min_val = float(param_meta.get("min_value", -1000.0))
        max_val = float(param_meta.get("max_value", 10000.0))
        clamped_val = max(min_val, min(max_val, float(value)))

        # Route evaluation by parameter key
        if param_key == "temperature":
            status, reason, actions = cls.evaluate_temperature(clamped_val, param_meta)
        elif param_key == "humidity":
            status, reason, actions = cls.evaluate_humidity(clamped_val, param_meta)
        elif param_key == "dust":
            status, reason, actions = cls.evaluate_dust(clamped_val, param_meta)
        elif param_key == "noise":
            status, reason, actions = cls.evaluate_noise(clamped_val, param_meta)
        elif param_key == "gas":
            status, reason, actions = cls.evaluate_gas(clamped_val, param_meta)
        else:
            # Fallback for generic or custom parameters
            status = HazardStatus.SAFE
            reason = f"{display_name} reading ({clamped_val:.1f} {unit}) evaluated."
            actions = ["Maintain routine monitoring per general site safety procedures."]

        return HazardAssessment(
            parameter_key=param_key,
            display_name=display_name,
            value=clamped_val,
            unit=unit,
            status=status,
            reason=reason,
            recommended_actions=actions,
        )

    @classmethod
    def assess_site(
        cls,
        readings: Dict[str, Any],
        parameters_meta: Dict[str, Any],
    ) -> SiteHazardAssessment:
        """
        Assesses the complete construction site environmental state across all monitored parameters.
        Sorts active hazards in strict severity order (DANGER first, then WARNING).
        Formulates overall site status based on the single worst active condition.
        """
        all_assessments: Dict[str, HazardAssessment] = {}

        for key, meta in parameters_meta.items():
            val = None
            if key in readings:
                raw_reading = readings[key]
                if isinstance(raw_reading, EnvironmentalReading):
                    val = raw_reading.value
                elif isinstance(raw_reading, (int, float)):
                    val = raw_reading
                elif isinstance(raw_reading, dict) and "value" in raw_reading:
                    val = raw_reading["value"]
            all_assessments[key] = cls.assess_parameter(key, val, meta)

        # Separate into active hazards (WARNING or DANGER) and safe parameters (SAFE)
        active_hazards = [a for a in all_assessments.values() if not a.is_safe]
        safe_parameters = [a for a in all_assessments.values() if a.is_safe]

        # Sort active hazards by severity descending (DANGER=2 first, WARNING=1 second)
        active_hazards.sort(key=lambda h: h.severity, reverse=True)

        # Determine overall site status
        if any(h.status == HazardStatus.DANGER for h in active_hazards):
            overall_status = HazardStatus.DANGER
            danger_names = [h.display_name for h in active_hazards if h.status == HazardStatus.DANGER]
            summary = (
                f"CRITICAL SITE DANGER DETECTED: Urgent safety action required for: {', '.join(danger_names)}. "
                "Follow mandatory protective protocols and execute emergency procedures."
            )
        elif any(h.status == HazardStatus.WARNING for h in active_hazards):
            overall_status = HazardStatus.WARNING
            warning_names = [h.display_name for h in active_hazards if h.status == HazardStatus.WARNING]
            summary = (
                f"ELEVATED SITE HAZARD WARNING: Precautionary measures advised for: {', '.join(warning_names)}. "
                "Heightened monitoring and protective action implementation required."
            )
        else:
            overall_status = HazardStatus.SAFE
            summary = (
                "ALL ENVIRONMENTAL PARAMETERS WITHIN ACCEPTABLE LIMITS. "
                "Site conditions are currently evaluated as normal and safe according to configured thresholds."
            )

        return SiteHazardAssessment(
            overall_status=overall_status,
            overall_summary=summary,
            active_hazards=active_hazards,
            safe_parameters=safe_parameters,
            all_assessments=all_assessments,
            disclaimer=SIMULATED_DATA_DISCLAIMER_SHORT,
        )
