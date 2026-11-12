"""Acoustic scene / environment inference from AudioSet scores and measurements (transparent rules)."""
from __future__ import annotations

import numpy as np

from .taxonomy import class_table

RULES = {
    "Beach / seashore": {"Waves, surf": 1.0, "Ocean": 1.0, "Gull, seagull": 0.6, "Wind": 0.3, "Wind noise (microphone)": 0.2},
    "Forest / woodland": {"Bird": 0.6, "Bird vocalization, bird call, bird song": 0.7, "Chirp, tweet": 0.5,
                          "Rustling leaves": 0.8, "Insect": 0.4, "Outside, rural or natural": 0.6, "Traffic noise, roadway noise": -0.8},
    "Urban street": {"Traffic noise, roadway noise": 1.0, "Car": 0.6, "Bus": 0.6, "Vehicle horn, car horn, honking": 0.6,
                     "Car passing by": 0.7, "Siren": 0.3, "Outside, urban or manmade": 0.8, "Walk, footsteps": 0.2,
                     "Hubbub, speech noise, speech babble": 0.3},
    "Rural / countryside": {"Outside, rural or natural": 1.0, "Bird": 0.4, "Insect": 0.4, "Cricket": 0.4,
                            "Livestock, farm animals, working animals": 0.7, "Chicken, rooster": 0.5, "Cattle, bovinae": 0.5,
                            "Traffic noise, roadway noise": -0.5},
    "Industrial area / worksite": {"Jackhammer": 0.8, "Power tool": 0.6, "Hammer": 0.4, "Heavy engine (low frequency)": 0.5,
                                   "Mechanisms": 0.4, "Drill": 0.5, "Sawing": 0.4, "Air compressor": 0.6, "Vibration": 0.4},
    "Railway environment": {"Train": 1.0, "Rail transport": 1.0, "Railroad car, train wagon": 0.8, "Train horn": 0.8,
                            "Train wheels squealing": 0.8, "Subway, metro, underground": 0.8, "Train whistle": 0.6},
    "Underground space / tunnel": {"Subway, metro, underground": 1.0, "Echo": 0.5, "Reverberation": 0.6},
    "Large indoor hall / public space": {"Inside, large room or hall": 1.0, "Inside, public space": 0.9, "Reverberation": 0.5,
                                         "Hubbub, speech noise, speech babble": 0.4, "Crowd": 0.3},
    "Small room / indoors": {"Inside, small room": 1.0, "Speech": 0.2, "Television": 0.4, "Typing": 0.4,
                             "Mechanical fan": 0.3, "Air conditioning": 0.3},
    "Water body (river, stream, waterfall)": {"Stream": 1.0, "Waterfall": 1.0, "Water": 0.4, "Trickle, dribble": 0.5, "Gurgling": 0.5},
}


def infer(sed: dict, rt60: dict | None, wind: dict | None) -> dict:
    t = {c["name"]: c["idx"] for c in class_table()}
    fs = sed["frame_scores"]
    p90 = np.percentile(fs, 90, axis=0)
    out = []
    for scene, w in RULES.items():
        contrib = []
        s = 0.0
        for name, wt in w.items():
            if name in t:
                v = float(p90[t[name]])
                if v >= 0.05:
                    contrib.append({"label": name, "p90_score": round(v, 3), "weight": wt})
                s += wt * v
        if scene.startswith("Large indoor") and rt60 and rt60.get("rt60_s"):
            if rt60["rt60_s"] >= 1.0:
                s += 0.3
                contrib.append({"label": f"RT60 {rt60['rt60_s']} s", "weight": 0.3})
        if scene.startswith("Small room") and rt60 and rt60.get("rt60_s") and rt60["rt60_s"] < 0.5:
            s += 0.15
            contrib.append({"label": f"RT60 {rt60['rt60_s']} s", "weight": 0.15})
        if scene.startswith("Underground") and rt60 and rt60.get("rt60_s") and rt60["rt60_s"] >= 1.2:
            s += 0.3
            contrib.append({"label": f"RT60 {rt60['rt60_s']} s", "weight": 0.3})
        out.append({"scene": scene, "score": round(s, 3), "contributors": contrib})
    out.sort(key=lambda d: -d["score"])
    inside = float(max(p90[t[n]] for n in ("Inside, small room", "Inside, large room or hall", "Inside, public space")))
    outside = float(max(p90[t[n]] for n in ("Outside, urban or manmade", "Outside, rural or natural")))
    io = ("indoors" if inside > outside + 0.05 else "outdoors" if outside > inside + 0.05 else "undetermined")
    top = [o for o in out if o["score"] >= 0.15][:3]
    return {"ranked": out, "top": top, "indoor_outdoor": {"estimate": io, "inside_p90": round(inside, 3), "outside_p90": round(outside, 3)},
            "not_inferable": ["Desert or sand environments (no reliable acoustic signature in the models used)",
                              "Temperature, humidity, sand or soil composition",
                              "Exact wind speed"],
            "method": "Weighted sum of 90th-percentile AudioSet frame scores plus RT60 cues. Weights are hand-set "
                      "heuristics, shown in full, not trained."}
