"""Birds and wildlife: BirdNET+ species events, association with AudioSet animal events."""
from __future__ import annotations

import numpy as np

from .inventory import Inventory, overlap, runs, tier_for

CLASS_TO_AUDIOSET = {
    "Aves": {"Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet", "Squawk", "Pigeon, dove", "Coo", "Crow",
             "Caw", "Owl", "Hoot", "Gull, seagull", "Duck", "Quack", "Goose", "Honk", "Chicken, rooster",
             "Crowing, cock-a-doodle-doo", "Cluck", "Wild animals", "Animal", "Bird flight, flapping wings"},
    "Insecta": {"Insect", "Cricket", "Mosquito", "Fly, housefly", "Bee, wasp, etc.", "Buzz", "Animal", "Wild animals"},
    "Amphibia": {"Frog", "Croak", "Animal", "Wild animals"},
    "Mammalia": {"Animal", "Wild animals", "Dog", "Bark", "Howl", "Cat", "Meow", "Roaring cats (lions, tigers)", "Roar",
                 "Livestock, farm animals, working animals", "Cattle, bovinae", "Moo", "Sheep", "Bleat", "Goat",
                 "Pig", "Oink", "Horse", "Neigh, whinny", "Rodents, rats, mice", "Bat", "Whale vocalization"},
    "Reptilia": {"Animal", "Wild animals", "Snake", "Hiss"},
}
CLASS_TITLE = {"Aves": "Bird", "Insecta": "Insect", "Amphibia": "Amphibian", "Mammalia": "Mammal", "Reptilia": "Reptile"}


def add_birdnet(inv: Inventory, bn: dict, labels: list[dict], profile_fn, x32k=None, onset_fn=None) -> list[dict]:
    """One record per species per contiguous run of windows (an individual calling bout)."""
    sc = bn["scores"]
    st, en = bn["starts"], bn["ends"]
    hop = float(st[1] - st[0]) if len(st) > 1 else 3.0
    recs = []
    cand_species = np.where(sc.max(axis=0) >= inv.thr)[0]
    for k in cand_species:
        col = sc[:, k]
        for a, b in runs(col >= inv.thr, gap=1):
            t0, t1 = float(st[a]), float(en[b - 1])
            lab = labels[k]
            conf = float(col[a:b].max())
            others = sc[a:b].max(axis=0)
            alt_idx = [i for i in np.argsort(-others)[:4] if i != k and others[i] >= 0.05][:3]
            taxon = lab["class"] or "Unknown"
            sed_match = inv.find_overlapping(
                t0, t1, lambda r, t=taxon: any(n in CLASS_TO_AUDIOSET.get(t, set()) for n in r.get("_audioset_names", [])),
                min_frac=0.2)
            corroborated = bool(sed_match)
            calls = onset_fn(t0, t1) if onset_fn else None
            rec = inv.add(
                category="birds_wildlife",
                label=f"{CLASS_TITLE.get(taxon, 'Animal')} vocalization",
                specific=f"Possible species: {lab['common']} ({lab['sci']})",
                common_name=lab["common"], scientific_name=lab["sci"], taxon_class=taxon, taxon_order=lab["order"],
                candidates=[{"label": f"{labels[i]['common']} ({labels[i]['sci']})", "score": round(float(others[i]), 4)}
                            for i in alt_idx],
                start=t0, end=t1, confidence=conf, tier=tier_for(conf, corroborated),
                frequency=profile_fn(t0, t1, 500.0, 12000.0), model="BirdNET+ V3.0-preview3.1",
                corroborated_by=[f"{r['id']} {r['label']} (AudioSet)" for r in sed_match],
                evidence={"model_outputs": [{"model": "BirdNET+ V3.0-preview3.1", "window_start": round(float(st[i]), 2),
                                             "window_end": round(float(en[i]), 2), "score": round(float(col[i]), 4)}
                                            for i in range(a, b)],
                          "estimated_call_onsets": calls},
                call_or_song="Not supported by model (BirdNET+ V3 does not distinguish call from song).",
                limitations=("Species identity is a model prediction, never confirmed by EchoMap. BirdNET+ V3 is a "
                             "developer preview without published accuracy. Similar species, mimics (e.g. starlings, "
                             "mockingbirds), recordings of birds, captive birds, and bird-imitating crossing signals "
                             "(Japan, Canada) can all produce detections."),
                signature=f"{CLASS_TITLE.get(taxon, 'Animal')} vocal activity, {b - a} window(s)",
            )
            recs.append(rec)
    # absorb AudioSet animal events fully covered by species records into those records' evidence
    absorbed = []
    for r in list(inv.records):
        names = r.get("_audioset_names")
        if not names or r["category"] != "birds_wildlife":
            continue
        cover = [s for s in recs if overlap(r["start"], r["end"], s["start"], s["end"]) >= 0.6 * max(r["duration"], 1e-3)]
        if cover:
            for s in cover:
                s["evidence"].setdefault("associated_audioset_events", []).append(
                    {"label": r["label"], "score": r["confidence"], "start": r["start"], "end": r["end"]})
            absorbed.append(r)
    inv.remove(absorbed)
    for r in inv.records:
        if r.get("_audioset_names") and r["category"] == "birds_wildlife" and r["label"] in CLASS_TO_AUDIOSET["Aves"]:
            r["specific"] = "Species not identified (no BirdNET+ species score above threshold in this interval)"
    return recs
