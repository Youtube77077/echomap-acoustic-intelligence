"""Analysis modes and per-case capability status (what ran, what did not, and why)."""
from __future__ import annotations

MODES = {
    "quick": {"title": "Quick", "steps": {"integrity", "dsp", "sed", "scene"}},
    "standard": {"title": "Standard", "steps": {"integrity", "dsp", "sed", "scene", "birdnet", "langid", "specialists",
                                                "direction", "geo"}},
    "deep": {"title": "Deep", "steps": {"integrity", "dsp", "sed", "scene", "birdnet", "langid", "specialists", "direction",
                                        "geo", "bells", "references", "relations"}},
    "experimental": {"title": "Experimental", "steps": {"integrity", "dsp", "sed", "scene", "birdnet", "langid", "specialists",
                                                        "direction", "geo", "bells", "references", "relations", "discovery",
                                                        "avas"}},
}

# capability class per feature (shown in the UI and report)
CLASSES = {
    "measured": "Directly measurable from the recording",
    "model": "Model-based acoustic prediction",
    "context": "External contextual inference",
    "hardware": "Requires external hardware",
    "impossible": "Not identifiable from the available recording",
}

STATIC = [
    {"id": "rf", "name": "Radio-frequency / microwave emitter detection", "class": "hardware", "status": "requires_hardware",
     "reason": "A microphone measures acoustic pressure, not RF fields. RF analysis needs an SDR receiver (e.g. RTL-SDR, "
               "HackRF, Airspy) and antenna. Only electrical interference that leaks into the audio (mains hum, switching "
               "noise) is analysed."},
    {"id": "source_sep", "name": "Neural source separation (Demucs / AudioSep / SAM Audio)", "class": "model",
     "status": "unavailable", "reason": "Not bundled: these models are 100 MB to several GB and GPU-oriented. EchoMap offers "
                                        "band-limited listening of each event's frequency region and harmonic/percussive "
                                        "analysis instead. A cloud-side separation pass is planned."},
    {"id": "nl_search", "name": "Natural-language sound search (TinyCLAP)", "class": "model", "status": "unavailable",
     "reason": "Verified: TinyCLAP code and weights are Apache-2.0, student checkpoint 17.8 MB. Its text side needs the "
               "Microsoft CLAP 2022 teacher (2.3 GB, licence stated inconsistently as MIT / CC-BY-3.0-US / MS-PL), so the "
               "conversion to ONNX with precomputed text embeddings is not yet done. AudioSet-label text search is offered "
               "instead."},
    {"id": "vehicle_make", "name": "Vehicle manufacturer / model identification", "class": "impossible",
     "status": "unavailable", "reason": "No validated public dataset or model. Only vehicle class, engine harmonics and "
                                        "behaviour are reported."},
    {"id": "speaker_id", "name": "Speaker identification", "class": "impossible", "status": "unavailable",
     "reason": "Deliberately not implemented (privacy)."},
    {"id": "island_grid", "name": "Small / island grid identification from hum fluctuations", "class": "model",
     "status": "unavailable", "reason": "Small grids (islands, isolated systems) fluctuate more than large synchronous grids, "
                                        "but EchoMap does not yet bundle verified per-grid fluctuation statistics, so the "
                                        "measured hum variability is reported without a geographic interpretation."},
    {"id": "train_model", "name": "Train model / operator identification", "class": "impossible", "status": "unavailable",
     "reason": "No open labelled dataset of train-model sounds. Station names in transcribed announcements are used "
               "as place clues instead; door chimes match only if a similar reference recording exists."},
    {"id": "accent_id", "name": "Accent / dialect identification", "class": "model", "status": "unavailable",
     "reason": "Not bundled. Would separate e.g. British from other English speakers; needs a new model (vetting first)."},
    {"id": "altitude", "name": "Source altitude / 3-D position", "class": "impossible", "status": "unsupported",
     "reason": "Needs a calibrated microphone array. Mono or stereo recordings cannot establish elevation or distance."},
]


class CapabilityLog:
    def __init__(self, mode: str):
        self.mode = mode if mode in MODES else "deep"
        self.steps = MODES[self.mode]["steps"]
        self.items: list[dict] = []

    def wants(self, step: str) -> bool:
        return step in self.steps

    def add(self, cid: str, name: str, status: str, reason: str = "", cls: str = "model", **detail):
        self.items.append({"id": cid, "name": name, "status": status, "reason": reason, "class": cls,
                           "class_title": CLASSES.get(cls, cls), **detail})

    def skipped_by_mode(self, cid: str, name: str, cls: str = "model"):
        self.add(cid, name, "not_executed", f"Not part of the '{MODES[self.mode]['title']}' analysis mode.", cls)

    def to_json(self) -> dict:
        return {"mode": self.mode, "mode_title": MODES[self.mode]["title"], "items": self.items + STATIC,
                "classes": CLASSES}
