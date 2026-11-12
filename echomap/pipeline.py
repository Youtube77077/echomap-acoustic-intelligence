"""Case orchestration.

Stage A (analyse): ingest → DSP → models → detection inventory → scene
Stage B (context): questionnaire → geography → species profiles → time → weather → report
Stage B can be re-run with different context answers without re-analysing the audio.
"""
from __future__ import annotations

import json
import shutil
import threading
import traceback
from pathlib import Path

import numpy as np

from . import config
from .detect import scene as scene_mod
from .detect import specialists as sp
from .detect.inventory import Inventory
from .detect.wildlife import add_birdnet
from .dsp import events as ev
from .dsp import reverb, spectral, tonal
from .geo.engine import GeoEngine
from .ingest import media
from .models import registry, runtime
from .models import sed as sedm
from .provenance import History, utcnow
from .capabilities import CapabilityLog
from .dsp import bells as bells_dsp, integrity, spatial
from .detect import discovery, references, relations
from .geo import bells_dove
from .temporal import timeinfo, weather


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


class _SkipSpecialists(Exception):
    pass


class Case:
    def __init__(self, path: str, settings: config.Settings):
        self.src = Path(path)
        self.settings = settings
        self.history = History()
        self.state = "created"
        self.progress = {"stage": "", "fraction": 0.0, "message": ""}
        self.error = None
        self.meta = None
        self.sha256 = None
        self.result: dict = {}
        self.x = None          # mono at analysis rate (≤ 48 kHz)
        self.sr = None
        self.lock = threading.Lock()
        self.dir: Path | None = None

    # ------------------------------------------------------------------ helpers
    def _p(self, stage, frac, msg=""):
        self.progress = {"stage": stage, "fraction": round(float(frac), 3), "message": msg}

    def prepare(self) -> dict:
        """Hash + metadata (before the user confirms analysis)."""
        h = self.history
        with h.timed("ingest.hash", "SHA-256 of original file (read-only)", file=str(self.src)):
            self.sha256 = media.sha256_file(self.src)
        with h.timed("ingest.probe", "Container and stream metadata via PyAV/FFmpeg"):
            self.meta = media.probe(self.src)
        d = self.meta.get("duration_s")
        if d and d > config.MAX_DURATION_S + 0.5:
            raise media.IngestError(f"Recording is {d:.1f} s long. EchoMap supports up to 5 minutes.")
        self.state = "prepared"
        return {"sha256": self.sha256, "meta": self.meta}

    def model_status(self):
        return registry.status(self.settings.models_dir)

    # ------------------------------------------------------------------ stage A
    def analyse(self):
        try:
            self._analyse()
            self.state = "analysed"
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.history.log("error", self.error, traceback=traceback.format_exc()[-2000:])
            self.state = "error"
        finally:
            runtime.release_all()

    def _analyse(self):
        s, h, md = self.settings, self.history, self.settings.models_dir
        status = registry.status(md)
        cap = CapabilityLog(getattr(s, "analysis_mode", "deep"))
        self.cap = cap
        h.log("mode", f"Analysis mode: {cap.mode}", steps=sorted(cap.steps))
        self.state = "analysing"
        case_dir = Path(s.cases_dir) / media.safe_case_name(self.src, self.sha256)
        case_dir.mkdir(parents=True, exist_ok=True)
        self.dir = case_dir
        h.log("case.create", "Case folder created", path=str(case_dir))
        if self.meta["is_video"]:
            self._p("extract", 0.02, "Extracting audio stream without re-encoding")
            with h.timed("ingest.extract", "Stream-copy primary audio track to Matroska (no re-encode)"):
                ex = media.extract_audio_stream(self.src, case_dir)
            self.result["extraction"] = ex
            h.log("ingest.extract.settings", "Extraction settings", **{k: v for k, v in ex.items() if k != "path"})
        self._p("decode", 0.05, "Decoding audio")
        sr_native = self.meta["primary_audio"]["sample_rate"] or 48000
        sr = min(sr_native, 48000)
        with h.timed("decode", "Decode primary audio (up to 2 channels kept for direction cues; mono mix for analysis)",
                     target_sr=sr, resampler="libswresample (PyAV)"):
            xs, sr = media.decode_channels(self.src, sr, max_channels=2)
            nch_total = self.meta["primary_audio"].get("channels") or xs.shape[0]
            x = xs.mean(axis=0).astype(np.float32)
            ch_rms = [float(20 * np.log10(np.sqrt(np.mean(c ** 2)) + 1e-12)) for c in xs]
            if len(x) > int((config.MAX_DURATION_S + 1) * sr):
                raise media.IngestError("Recording longer than 5 minutes.")
        media.check_duration(self.meta, len(x) / sr)
        peak = float(np.max(np.abs(x))) if len(x) else 0
        self.x, self.sr = x, sr
        dur = len(x) / sr
        self.result["analysis_copy"] = {"sample_rate": sr, "channels_downmixed": len(ch_rms), "channel_rms_dbfs": ch_rms,
                                        "duration_s": round(dur, 3), "peak_dbfs": round(20 * np.log10(peak + 1e-12), 2),
                                        "clipping_samples": int(np.sum(np.abs(x) >= 0.999)),
                                        "lossless_source": media.is_lossless(self.meta)}
        if s.keep_analysis_wav or self.meta["is_video"]:
            p = case_dir / "analysis_mono.wav"
            media.write_wav16(p, x, sr)
            self.result["analysis_copy"]["wav"] = {"path": str(p), "sha256": media.sha256_file(p)}
            h.log("decode.save", "Saved 16-bit mono analysis copy for playback", path=str(p))
        with h.timed("integrity", "Forensic integrity checks (bandwidth, clipping, dropouts, DC, metadata)"):
            self.result["integrity"] = integrity.run(x, sr, self.meta)
        cap.add("integrity", "Forensic integrity checks", "executed", cls="measured",
                flags=len(self.result["integrity"]["summary_flags"]))
        rep = spatial.channel_report(xs, sr)
        rep["channels_in_file"] = nch_total
        self.result["spatial"] = rep
        x32 = media.resample(x, sr, 32000)
        x16 = media.resample(x, sr, 16000)
        x4 = media.resample(x, sr, 4000)
        x2 = media.resample(x, sr, 2000)
        h.log("resample", "Model-rate copies via libswresample", rates=[32000, 16000, 4000, 2000])

        # ---------------- DSP
        self._p("dsp", 0.12, "Signal analysis")
        with h.timed("dsp.features", "Frame features, MFCC, LTAS, waveform envelope"):
            feats = spectral.frame_features(x, sr)
            self.result["signal"] = {"summary": spectral.summarize(feats), "ltas": spectral.ltas(x, sr),
                                     "waveform": spectral.waveform_envelope(x, 4000),
                                     "series": {"hop_s": feats["hop_s"],
                                                "rms_db": [round(float(v), 2) for v in 20 * np.log10(feats["rms"][::2] + 1e-12)],
                                                "centroid": [round(float(v), 1) for v in feats["centroid"][::2]]}}
        with h.timed("dsp.onsets", "Spectral-flux transients and periodicity"):
            on = ev.onsets(x, sr)
            per = ev.periodicity(on.pop("_env"), on["frame_period_s"])
            self.result["signal"]["onsets"] = {"count": len(on["times_s"]), "times_s": on["times_s"][:2000]}
            self.result["signal"]["periodicity"] = per
        with h.timed("dsp.snr", "SNR estimate"):
            self.result["signal"]["snr"] = ev.snr_estimate(x, sr)
        with h.timed("dsp.segments", "Energy segmentation"):
            segs, floor = ev.energy_segments(x, sr)
        with h.timed("dsp.reverb", "Blind RT60 from free decays"):
            rt = reverb.rt60_blind(x, sr, on["times_s"])
        with h.timed("dsp.wind", "Low-frequency wind-noise analysis"):
            wind = reverb.wind_analysis(x, sr)
        with h.timed("dsp.mains", "Mains hum / ENF analysis (2 kHz copy)"):
            mains = tonal.mains_hum(x2, 2000)
        self._p("dsp", 0.22, "Tonal tracking")
        with h.timed("dsp.tones", "Narrowband tone tracking and periodic patterns"):
            tracks = tonal.tone_tracks(x32, 32000)
            patterns = tonal.periodic_tone_patterns(tracks) + tonal.click_trains(on["times_s"])
        self.result["signal"].update({"reverb": rt, "wind": wind, "mains": mains, "tone_tracks": tracks[:300],
                                      "periodic_patterns": patterns, "noise_floor_dbfs_20ms": round(floor, 1)})

        def profile(t0, t1, lo=50.0, hi=None):
            return ev.band_profile(x, sr, t0, t1, lo, hi)

        # ---------------- models
        inv = Inventory(dur, s.list_threshold)
        if not status["sed"]["installed"]:
            raise RuntimeError("The AudioSet model is missing: " + ", ".join(status["sed"]["missing"]))
        self._p("sed", 0.28, "AudioSet sound event detection")
        with h.timed("model.sed", "EfficientAT mn10_as, 5 s windows, 0.5 s hop", onnxruntime=runtime.version()):
            sed = sedm.run(x32, md, s.sed_hop_s,
                           progress=lambda f: self._p("sed", 0.28 + 0.2 * f, "AudioSet sound event detection"))
        labels = sedm.load_labels(md)
        self._sed = sed
        cap.add("sed", "AudioSet sound event detection (527 classes)", "executed", model="EfficientAT mn10_as")
        raw = inv.sed_events(sed)
        groups = inv.associate(raw)
        inv.add_sed_groups(groups, profile)
        h.log("inventory.sed", "AudioSet events associated", raw_events=len(raw), grouped_records=len(groups))
        bn = None
        if not cap.wants("birdnet"):
            cap.skipped_by_mode("birdnet", "Wildlife species identification")
        elif not status["birdnet"]["installed"]:
            cap.add("birdnet", "Wildlife species identification", "unavailable", "BirdNET+ model files not found")
        if status["birdnet"]["installed"] and cap.wants("birdnet"):
            self._p("birdnet", 0.5, "Species identification (BirdNET+)")
            from .models import birdnet as bnm
            with h.timed("model.birdnet", "BirdNET+ V3.0-preview3.1, 3 s windows, 1.5 s hop"):
                bn = bnm.run(x32, md, s.birdnet_hop_s, progress=lambda f: self._p("birdnet", 0.5 + 0.2 * f, "Species identification"))

            def calls(t0, t1):
                sub = [t for t in on["times_s"] if t0 <= t <= t1]
                return {"count": len(sub), "times_s": sub[:100],
                        "note": "Broadband onsets inside the interval; may include non-bird transients."}
            add_birdnet(inv, bn, bnm.labels(md), profile, onset_fn=calls)
            if bnm.RESCALED:
                h.log("model.birdnet.rescaled", "Chunks re-run at reduced gain after non-finite FP16 outputs",
                      chunks=sum(bnm.RESCALED))
                bnm.RESCALED.clear()
            self.result["birdnet_raw"] = {"starts": bn["starts"], "top": self._top_raw(bn["scores"], bnm.labels(md), "common")}
            cap.add("birdnet", "Wildlife species identification", "executed", model="BirdNET+ V3.0-preview3.1")
        else:
            h.log("model.birdnet", "skipped")
        if not cap.wants("langid"):
            cap.skipped_by_mode("langid", "Spoken language identification")
        elif not status["langid"]["installed"]:
            cap.add("langid", "Spoken language identification", "unavailable", "Language pack not installed")
        if status["langid"]["installed"] and cap.wants("langid"):
            self._p("langid", 0.72, "Language identification")
            from .models import langid
            regions = sp.speech_regions(sed)
            with h.timed("model.langid", "Whisper-base LID-only graph on speech regions", regions=len(regions)):
                sp.add_languages(inv, regions, x16, lambda c: langid.identify(c, md))
            cap.add("langid", "Spoken language identification", "executed" if regions else "partial",
                    "" if regions else "No speech region of at least 1 s with a speech score of at least 0.3, so there is "
                                       "insufficient speech for language identification.", model="Whisper base LID")
        else:
            h.log("model.langid", "skipped: language pack not installed")
        if not s.transcribe_speech:
            cap.add("asr", "Speech transcription (all speech, incl. faint background)", "not_executed", "Disabled in settings.")
        elif not status.get("asr", {}).get("installed"):
            cap.add("asr", "Speech transcription (all speech, incl. faint background)", "unavailable",
                    "Speech-to-text pack (models-asr-v1) not installed.")
        else:
            from .models import asr
            from .detect.taxonomy import SPEECH, class_table
            sidx = [c["idx"] for c in class_table() if c["name"] in SPEECH]
            self._p("asr", 0.78, "Transcribing speech (including faint background speech)")
            langs = [(r["start"], r["end"], r["language_code"]) for r in inv.records if r.get("language_code")]

            def hint(t0, t1):
                best = max(langs, key=lambda L: min(t1, L[1]) - max(t0, L[0]), default=None)
                return best[2] if best and min(t1, best[1]) - max(t0, best[0]) > 0 else None
            with h.timed("asr", "Whisper-base transcription of all speech candidates (low threshold; faint speech enhanced)"):
                trs = asr.run_all(x16, sed, sidx, md, hint)
            self.result["transcripts"] = {"segments": trs, "method": asr.__doc__.strip(),
                                          "counts": {q: sum(1 for t in trs if t["quality"] == q)
                                                     for q in ("clear", "uncertain", "likely spurious")}}
            for tr in trs:
                for r in inv.records:
                    if r["category"] == "language" and r["start"] < tr["end"] and r["end"] > tr["start"]:
                        r.setdefault("transcripts", []).append({k: tr[k] for k in ("start", "end", "text", "quality", "language")})
            cap.add("asr", "Speech transcription (all speech, incl. faint background)", "executed" if trs else "partial",
                    "" if trs else "No speech candidates found, even at a low threshold.", segments=len(trs))
        self._p("specialists", 0.82, "Engine, siren, bell, signal and wind analysis")
        try:
          with h.timed("detect.specialists", "Engine harmonics, sirens, bells, periodic signals, wind, mains, reverb"):
            if not cap.wants("specialists"):
                cap.skipped_by_mode("specialists", "Engine, siren, signal and bell analysis", "measured")
                sp.add_mains(inv, mains, dur)
                sp.add_wind(inv, wind)
                sp.add_reverb(inv, rt, dur)
                sp.add_unclassified(inv, segs, profile, sed, [l["name"] for l in labels])
                raise _SkipSpecialists()
            sp.add_engine_analysis(inv, x4, tonal.engine_f0_track)
            sp.add_sirens(inv, x32, tonal.siren_pattern)
            sp.add_siren_scan(inv, tonal.siren_scan(x32, 32000), profile)
            sp.add_bell_counts(inv, on["times_s"])
            sp.add_periodic(inv, patterns, profile)
            if cap.wants("avas"):
                sp.add_avas(inv, tonal.avas_like(tracks), profile)
                cap.add("avas", "AVAS (electric-vehicle warning sound) pattern detector", "executed", cls="measured",
                        experimental=True)
            else:
                cap.skipped_by_mode("avas", "AVAS pattern detector (experimental)", "measured")
            sp.add_mains(inv, mains, dur)
            sp.add_wind(inv, wind)
            sp.add_reverb(inv, rt, dur)
            sp.add_unclassified(inv, segs, profile, sed, [l["name"] for l in labels])
            cap.add("specialists", "Engine harmonics, sirens, bells, periodic signals, wind, mains hum, reverberation",
                    "executed", cls="measured")
        except _SkipSpecialists:
            pass
        # bells: strike analysis + Dove's Guide match
        self._bell = None
        bell_recs = [r for r in inv.records if set(r.get("_audioset_names", [])) &
                     {"Church bell", "Change ringing (campanology)", "Bell"} and r["confidence"] >= 0.2]
        if cap.wants("bells") and bell_recs:
            with h.timed("bells", "Bell partial analysis and Dove's Guide matching"):
                ons = [t for t in on["times_s"] if any(r["start"] - 0.2 <= t <= r["end"] + 0.2 for r in bell_recs)]
                ana = bells_dsp.analyse_strikes(x, sr, ons)
                m = bells_dove.match(ana) if ana.get("bells") else {"available": True, "candidates": []}
                self.result["bells"] = {"analysis": {k: v for k, v in ana.items() if k != "strikes"}, "dove_match":
                                        {k: v for k, v in m.items() if k != "all_compatible"}}
                rec = max(bell_recs, key=lambda r: r["confidence"])
                rec["evidence"]["bell_analysis"] = self.result["bells"]["analysis"]
                if m.get("candidates"):
                    rec["evidence"]["dove_candidates"] = m["candidates"][:10]
                    rec["specific"] = (f"{ana['ring_size_observed']} bell(s) reconstructed, tenor nominal "
                                       f"{ana['tenor_nominal_hz']} Hz; {m['n_compatible']} compatible Dove rings")
                conf = min(0.6, 0.15 + 0.05 * ana.get("ring_size_observed", 0) + (0.1 if ana.get("diatonic_like") else 0))
                self._bell = {"bell_match": m, "bell_record_obj": rec, "bell_confidence": conf}
            cap.add("bells", "Bell partial analysis and Dove's Guide tower matching",
                    "executed" if ana.get("bells") else "partial", ana.get("reason", ""), cls="measured",
                    compatible_rings=m.get("n_compatible", 0))
        elif cap.wants("bells"):
            cap.add("bells", "Bell partial analysis and Dove's Guide tower matching", "not_applicable",
                    "No bell detected in the recording.", cls="measured")
        else:
            cap.skipped_by_mode("bells", "Bell partial analysis and Dove's Guide matching", "measured")
        # stereo direction per event
        if cap.wants("direction"):
            if rep.get("supported"):
                for r in inv.records:
                    if r["end"] - r["start"] < 0.9 * dur:
                        r["direction"] = spatial.event_direction(xs, sr, r["start"], r["end"])
                cap.add("direction", "Stereo direction cues (lateral angle range)", "executed", cls="measured",
                        experimental=True)
            else:
                cap.add("direction", "Stereo direction cues", "unsupported_by_recording", rep.get("reason", ""), cls="measured")
        with h.timed("detect.scene", "Scene inference rules"):
            self.result["scene"] = scene_mod.infer(sed, rt, wind)
        self.result["sed_raw"] = {"frame_t": sed["frame_t"], "hop_s": sed["hop_s"], "win_s": sed["win_s"],
                                  "padded_to_5s": sed["padded_to_5s"],
                                  "top": self._top_raw(sed["frame_scores"], labels, "name")}
        self._sed_frame_scores = sed["frame_scores"]
        self._labels = [l["name"] for l in labels]
        records = inv.finalize()
        self.result["inventory"] = records
        if self._bell:
            self._bell["bell_record"] = self._bell.pop("bell_record_obj")["id"]
        x8 = media.resample(x, sr, 8000)
        if cap.wants("references"):
            with h.timed("references", "Reverse sound search (fingerprints + embeddings) against the reference library"):
                self.result["references"] = references.search_events(records, sed, x8)
            from .models import vehicle_head
            nveh = vehicle_head.annotate(records, sed)
            cap.add("vehicle_type", "Vehicle type (car / bus / truck / tram / motorcycle / bicycle)",
                    "executed" if vehicle_head.head() else "unavailable",
                    f"{nveh} vehicle event(s) classified" if vehicle_head.head() else "Vehicle-type head not bundled.")
            cap.add("references", "Acoustic signature database / reverse sound search",
                    "executed" if self.result["references"].get("available") else "unavailable",
                    self.result["references"].get("reason", ""))
        else:
            cap.skipped_by_mode("references", "Acoustic signature database / reverse sound search")
        if cap.wants("relations"):
            with h.timed("relations", "Event relationship graph and sequence hypotheses"):
                sers = self.result["signal"]["series"]
                self.result["relations"] = relations.build(records, dur, sers["rms_db"], sers["hop_s"] * 2)
            cap.add("relations", "Event relationships and sequence hypotheses", "executed", cls="measured")
        else:
            cap.skipped_by_mode("relations", "Event relationships", "measured")
        if cap.wants("discovery"):
            with h.timed("discovery", "Unknown-sound discovery (novelty, disagreement, embedding families)"):
                self.result["discovery"] = discovery.discover(
                    records, sed, bn, x, sr, patterns, profile, [l["name"] for l in labels],
                    references.search_embedding, case_id=self.dir.name)
            cap.add("discovery", "Unknown-sound discovery", "executed", experimental=True)
        else:
            cap.skipped_by_mode("discovery", "Unknown-sound discovery (experimental)")
        self.result["capabilities"] = cap.to_json()
        self.result["inventory_summary"] = inv.summary()
        self.result["acoustic_weather"] = weather.acoustic_observations(records, wind)
        h.log("inventory.final", "Detection inventory finalised", records=len(records))
        self._p("done", 1.0, "Analysis complete. Provide context for geographic analysis.")
        self.save()

    @staticmethod
    def _top_raw(scores, labels, key, k=40):
        mx = scores.max(axis=0)
        idx = np.argsort(-mx)[:k]
        return [{"label": labels[i][key], "max": round(float(mx[i]), 4)} for i in idx]

    # ------------------------------------------------------------------ stage B
    def contextualise(self, ctx: dict):
        try:
            self._contextualise(ctx)
            self.state = "complete"
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.history.log("error", self.error, traceback=traceback.format_exc()[-2000:])
            self.state = "error"
        finally:
            runtime.release_all()

    def _contextualise(self, ctx: dict):
        s, h = self.settings, self.history
        self.state = "geolocating"
        self.result["context"] = ctx
        h.log("context", "User context recorded", **{k: v for k, v in (ctx or {}).items()})
        records = self.result["inventory"]
        self._p("geo", 0.1, "Geographic inference")
        eng = GeoEngine(s.models_dir, h)
        with h.timed("geo.fusion", "Evidence fusion on 0.5° land grid"):
            extra = dict(getattr(self, "_bell", None) or {})
            refgeo = []
            for e in (self.result.get("references") or {}).get("events", []):
                for m in e["matches"]:
                    transit = ("crossing signal", "level crossing", "announcement", "bells", "siren", "tram", "metro", "train", "bus")
                    if m.get("lat") is not None and m["tier"] == "strong acoustic similarity" and m["category"] in transit:
                        refgeo.append({**m, "record": e["record"]})
            # near-exact fingerprint matches (the same jingle / announcement recording) are much stronger
            for m in (self.result.get("references") or {}).get("fingerprint_matches", []):
                if m.get("lat") is not None:
                    refgeo.append({**m, "similarity": 1.0, "record": None, "exact": True})
            refgeo.sort(key=lambda m: (not m.get("exact"), -m.get("similarity", 0)))
            extra["reference_geo"] = refgeo[:6]
            # v1.0.3 point clues: tram hypothesis and spoken place names
            from .geo import places
            extra["tram"] = places.tram_hypothesis(records, self.result.get("references"))
            segs = (self.result.get("transcripts") or {}).get("segments") or []
            mentions = places.place_mentions(segs)
            for m in mentions:
                t0 = m["segments"][0]["start"] or 0
                rec = next((r for r in records if r.get("transcripts") and r["start"] - 0.5 <= t0 <= r["end"] + 0.5), None)
                m["record"] = rec["id"] if rec else None
            extra["place_mentions"] = mentions
            self.result["place_mentions"] = [{k: v for k, v in m.items() if k != "rows"} for m in mentions]
            if getattr(self, "cap", None):
                tn = places.tram_networks()["networks"]
                self.cap.items = [i for i in self.cap.items if i["id"] not in ("tram", "place_names")]
                self.cap.add("tram", "Tram / streetcar clue (OSM tram networks)", "executed" if tn else "unavailable",
                             ((extra["tram"]["basis"] if extra["tram"] else "No tram-like sound found.") if tn
                              else "Tram network dataset not bundled."), cls="context", networks=len(tn))
                self.cap.add("place_names", "Place names in transcribed speech", "executed" if segs else "not_executed",
                             f"{len(mentions)} place name(s) found." if segs else "No transcript available.", cls="context")
            geo = eng.run(records, self.result["signal"]["mains"], ctx, extra)
        best = (geo.get("local_refinement", {}).get("candidates") or geo.get("candidates") or [None])[0]
        with h.timed("geo.species", "Species distribution and seasonality profiles"):
            geo["species_profiles"] = eng.species_profiles(records, best)
        # optional online lookups
        online = {}
        if s.allow_online_gbif:
            from .geo import online as onl
            for p in geo["species_profiles"][:10]:
                try:
                    online.setdefault("gbif", {})[p["scientific_name"]] = onl.gbif_country_counts(p["scientific_name"], h)
                except Exception as e:  # noqa: BLE001
                    online.setdefault("gbif", {})[p["scientific_name"]] = {"error": str(e)}
        if s.allow_online_osm and best and best["uncertainty_radius_km"] <= 100:
            from .geo import online as onl
            wanted = set()
            for r in records:
                if r["category"] == "railway":
                    wanted |= {"railway", "level_crossing"}
                if "Church bell" in r["label"] or "bell" in r["label"].lower():
                    wanted.add("place_of_worship_christian")
                if r["label"] in ("Waves, surf", "Ocean"):
                    wanted.add("coastline_beach")
                if r["label"] == "Waterfall":
                    wanted.add("waterfall")
                if "pedestrian" in json.dumps(r.get("candidates", [])).lower():
                    wanted.add("audible_traffic_signals")
            if wanted:
                try:
                    online["osm"] = onl.overpass_feature_counts(best["centre"]["lat"], best["centre"]["lon"],
                                                                best["uncertainty_radius_km"], sorted(wanted), h)
                except Exception as e:  # noqa: BLE001
                    online["osm"] = {"error": str(e)}
        self._p("time", 0.6, "Temporal analysis")
        with h.timed("time.analyse", "Separate time sources and seasonal compatibility"):
            self.result["time"] = timeinfo.analyse(self.meta, ctx, records, geo["species_profiles"])
        dctx = (ctx or {}).get("date") or {}
        if s.allow_online_weather and best and dctx.get("mode") in ("exact", "approx_date") and dctx.get("value"):
            from .geo import online as onl
            try:
                ext = onl.open_meteo_day(best["centre"]["lat"], best["centre"]["lon"], dctx["value"][:10], h)
                hour = int(dctx["value"][11:13]) if dctx.get("mode") == "exact" and len(dctx["value"]) >= 13 else None
                self.result["weather_external"] = weather.compare_external(self.result["acoustic_weather"], ext, hour)
            except Exception as e:  # noqa: BLE001
                self.result["weather_external"] = {"error": str(e)}
        geo["online"] = online
        if getattr(self, "cap", None):
            self.cap.items = [i for i in self.cap.items if i["id"] != "geo"]
            self.cap.add("geo", "Probabilistic geographic inference", "executed" if geo.get("status") == "estimate" else "partial",
                         "" if geo.get("status") == "estimate" else geo.get("summary", ""), cls="context",
                         footprints=len(geo.get("footprints", [])))
            self.result["capabilities"] = self.cap.to_json()
        self.result["geo"] = geo
        self._p("report", 0.85, "Writing report")
        from .report import html as rhtml
        with h.timed("report.html", "Self-contained HTML report"):
            p = self.dir / "report.html"
            p.write_text(rhtml.render(self), encoding="utf-8")
            self.result["report_html"] = str(p)
        self._p("done", 1.0, "Complete")
        self.save()

    # ------------------------------------------------------------------ persistence
    def to_json(self) -> dict:
        return _jsonable({
            "app": {"name": config.APP_NAME, "version": config.APP_VERSION},
            "created_utc": self.history.steps[0]["time_utc"] if self.history.steps else utcnow(),
            "file": {"path": str(self.src), "sha256": self.sha256}, "meta": self.meta,
            "models": {k: {kk: v[kk] for kk in ("name", "source", "licence", "published_accuracy", "limitations", "installed")}
                       for k, v in registry.status(self.settings.models_dir).items()},
            "settings": {k: v for k, v in vars(self.settings).items() if k != "extra"},
            "result": self.result, "history": self.history.to_json(), "state": self.state, "error": self.error,
        })

    def save(self):
        if not self.dir:
            return
        d = self.to_json()
        if getattr(self, "_sed_frame_scores", None) is not None:
            d["raw_scores"] = {"audioset_frame_scores": {"hop_s": self.settings.sed_hop_s, "labels": self._labels,
                                                         "scores": np.round(self._sed_frame_scores, 4).tolist()}}
        (self.dir / "case.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    def spectrogram(self, **kw):
        return spectral.spectrogram_view(self.x, self.sr, **kw)

    def delete_files(self):
        if self.dir and self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
