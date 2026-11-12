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
        with h.timed("decode", "Decode primary audio to float32 mono", target_sr=sr, resampler="libswresample (PyAV)"):
            x, sr, ch_rms = media.decode_mono(self.src, sr)
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
        raw = inv.sed_events(sed)
        groups = inv.associate(raw)
        inv.add_sed_groups(groups, profile)
        h.log("inventory.sed", "AudioSet events associated", raw_events=len(raw), grouped_records=len(groups))
        if status["birdnet"]["installed"]:
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
        else:
            h.log("model.birdnet", "skipped: model not installed")
        if status["langid"]["installed"]:
            self._p("langid", 0.72, "Language identification")
            from .models import langid
            regions = sp.speech_regions(sed)
            with h.timed("model.langid", "Whisper-base LID-only graph on speech regions", regions=len(regions)):
                sp.add_languages(inv, regions, x16, lambda c: langid.identify(c, md))
        else:
            h.log("model.langid", "skipped: language pack not installed")
        self._p("specialists", 0.82, "Engine, siren, bell, signal and wind analysis")
        with h.timed("detect.specialists", "Engine harmonics, sirens, bells, periodic signals, wind, mains, reverb"):
            sp.add_engine_analysis(inv, x4, tonal.engine_f0_track)
            sp.add_sirens(inv, x32, tonal.siren_pattern)
            sp.add_bell_counts(inv, on["times_s"])
            sp.add_periodic(inv, patterns, profile)
            sp.add_mains(inv, mains, dur)
            sp.add_wind(inv, wind)
            sp.add_reverb(inv, rt, dur)
            sp.add_unclassified(inv, segs, profile, sed, [l["name"] for l in labels])
        with h.timed("detect.scene", "Scene inference rules"):
            self.result["scene"] = scene_mod.infer(sed, rt, wind)
        self.result["sed_raw"] = {"frame_t": sed["frame_t"], "hop_s": sed["hop_s"], "win_s": sed["win_s"],
                                  "padded_to_5s": sed["padded_to_5s"],
                                  "top": self._top_raw(sed["frame_scores"], labels, "name")}
        self._sed_frame_scores = sed["frame_scores"]
        self._labels = [l["name"] for l in labels]
        records = inv.finalize()
        self.result["inventory"] = records
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
            geo = eng.run(records, self.result["signal"]["mains"], ctx)
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
