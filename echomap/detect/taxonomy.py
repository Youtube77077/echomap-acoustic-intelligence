"""EchoMap detection categories and the AudioSet → category mapping.

Categories are a registry: add a new entry to CATEGORIES and a rule in `categorize` (or register
from a new detector) to extend the inventory.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

CATEGORIES = [
    ("birds_wildlife", "Birds and wildlife"),
    ("vehicles", "Vehicles"),
    ("engines_machinery", "Engines and machinery"),
    ("public_transport", "Public transport"),
    ("railway", "Railway systems"),
    ("road_infrastructure", "Road infrastructure"),
    ("human_nonspeech", "Human-produced non-speech sounds"),
    ("language", "Language detection"),
    ("environmental", "Environmental sounds"),
    ("weather", "Weather-related sounds"),
    ("water_geological", "Water and geological sounds"),
    ("buildings_acoustics", "Buildings and architectural acoustics"),
    ("industrial", "Industrial sounds"),
    ("bells_alarms", "Bells and alarms"),
    ("electronic_signals", "Electronic signals"),
    ("music", "Music and musical sounds"),
    ("ambience", "Indoor and outdoor ambience"),
    ("unknown", "Unknown or unclassified sounds"),
]
CATEGORY_NAMES = dict(CATEGORIES)


def register_category(key: str, title: str) -> None:
    if key not in CATEGORY_NAMES:
        CATEGORIES.insert(-1, (key, title))
        CATEGORY_NAMES[key] = title


RAIL = {"Rail transport", "Train", "Train whistle", "Train horn", "Railroad car, train wagon",
        "Train wheels squealing", "Subway, metro, underground"}
PUBLIC_TRANSPORT = {"Bus"}
ROAD = {"Traffic noise, roadway noise"}
APPLIANCES = {"Vacuum cleaner", "Hair dryer", "Blender", "Microwave oven", "Electric toothbrush",
              "Electric shaver, electric razor", "Mechanical fan", "Air conditioning", "Sewing machine"}
ELECTRONIC = {"Beep, bleep", "Sine wave", "Static", "Mains hum", "White noise", "Pink noise", "Telephone",
              "Telephone bell ringing", "Ringtone", "Telephone dialing, DTMF", "Dial tone", "Busy signal",
              "Sidetone", "Hum", "Distortion", "Chirp tone", "Electronic tuner", "Sound effect", "Pulse",
              "Buzzer", "Radio", "Television", "Effects unit"}
BELLS = {"Church bell", "Jingle bell", "Bicycle bell", "Tuning fork", "Chime", "Wind chime", "Change ringing (campanology)",
         "Bell", "Doorbell", "Ding-dong", "Clock", "Tick-tock", "Tick", "Cowbell", "Alarm clock"}
ALARMS = {"Alarm", "Siren", "Civil defense siren", "Smoke detector, smoke alarm", "Fire alarm", "Foghorn",
          "Whistle", "Steam whistle", "Car alarm", "Reversing beeps", "Air horn, truck horn", "Police car (siren)",
          "Ambulance (siren)", "Fire engine, fire truck (siren)", "Kettle whistle", "Ice cream truck, ice cream van"}
WEATHER = {"Wind", "Wind noise (microphone)", "Thunderstorm", "Thunder", "Rain", "Raindrop", "Rain on surface"}
WATER = {"Water", "Stream", "Waterfall", "Ocean", "Waves, surf", "Steam", "Gurgling", "Drip", "Splash, splatter",
         "Pour", "Trickle, dribble", "Gush", "Fill (with liquid)", "Spray", "Liquid", "Boiling", "Sink (filling or washing)",
         "Bathtub (filling or washing)", "Water tap, faucet", "Toilet flush", "Slosh", "Squish", "Drip"}
BUILDINGS = {"Inside, small room", "Inside, large room or hall", "Inside, public space", "Reverberation", "Echo"}
AMBIENCE = {"Outside, urban or manmade", "Outside, rural or natural", "Environmental noise", "Silence",
            "Field recording", "Cacophony"}
SPEECH = {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
          "Conversation", "Narration, monologue", "Babbling", "Speech synthesizer", "Whispering"}
INDUSTRIAL_TOOLS = {"Tools", "Hammer", "Jackhammer", "Sawing", "Filing (rasp)", "Sanding", "Power tool", "Drill",
                    "Dental drill, dentist's drill", "Chainsaw", "Lawn mower"}

# groups whose members are *alternative* classifications of one physical event
ALTERNATIVE_GROUPS = [
    {"Car", "Bus", "Truck", "Motorcycle", "Motor vehicle (road)", "Vehicle", "Race car, auto racing", "Car passing by",
     "Van", "Pickup truck", "Ice cream truck, ice cream van", "Fire engine, fire truck (siren)", "Police car (siren)",
     "Ambulance (siren)", "Emergency vehicle", "Traffic noise, roadway noise"},
    {"Light engine (high frequency)", "Medium engine (mid frequency)", "Heavy engine (low frequency)", "Engine",
     "Idling", "Accelerating, revving, vroom", "Engine knocking", "Engine starting"},
    {"Train", "Rail transport", "Railroad car, train wagon", "Subway, metro, underground", "Train wheels squealing",
     "Train horn", "Train whistle"},
    {"Aircraft", "Aircraft engine", "Jet engine", "Propeller, airscrew", "Helicopter", "Fixed-wing aircraft, airplane"},
    {"Boat, Water vehicle", "Ship", "Motorboat, speedboat", "Sailboat, sailing ship", "Rowboat, canoe, kayak"},
    {"Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet", "Squawk", "Pigeon, dove", "Coo", "Crow", "Caw",
     "Owl", "Hoot", "Bird flight, flapping wings", "Gull, seagull", "Duck", "Quack", "Goose", "Honk", "Chicken, rooster",
     "Crowing, cock-a-doodle-doo", "Cluck", "Wild animals", "Animal"},
    SPEECH,
    {"Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)",
     "Emergency vehicle"},
    {"Bell", "Church bell", "Change ringing (campanology)", "Chime", "Jingle bell", "Bicycle bell", "Wind chime", "Tuning fork"},
    {"Rain", "Raindrop", "Rain on surface"},
    {"Wind", "Wind noise (microphone)", "Rustling leaves"},
    {"Thunderstorm", "Thunder"},
    {"Water", "Stream", "Waterfall", "Ocean", "Waves, surf", "Trickle, dribble", "Gush"},
    {"Music", "Musical instrument"},
    {"Crowd", "Hubbub, speech noise, speech babble", "Chatter", "Cheering", "Applause", "Children playing"},
    {"Footsteps", "Walk, footsteps", "Run", "Shuffle"},
    {"Door", "Slam", "Knock", "Sliding door", "Squeak", "Creak"},
    {"Insect", "Cricket", "Mosquito", "Fly, housefly", "Bee, wasp, etc.", "Buzz"},
    {"Frog", "Croak"},
    {"Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)"},
]


@lru_cache(maxsize=1)
def audioset_meta() -> list[dict]:
    p = Path(__file__).resolve().parents[1] / "data" / "audioset.json"
    return json.loads(p.read_text(encoding="utf-8"))


def categorize(name: str, ancestors: list[str]) -> str:
    anc = set(ancestors)
    if name in SPEECH:
        return "language"
    if name in RAIL:
        return "railway"
    if name in PUBLIC_TRANSPORT:
        return "public_transport"
    if name in ROAD:
        return "road_infrastructure"
    if name in BUILDINGS:
        return "buildings_acoustics"
    if name in AMBIENCE:
        return "ambience"
    if name in BELLS or (anc & {"Bell"}):
        return "bells_alarms"
    if name in ALARMS or "Alarm" in anc and name not in ELECTRONIC:
        return "bells_alarms"
    if name in ELECTRONIC:
        return "electronic_signals"
    if name in WEATHER:
        return "weather"
    if name in WATER or "Water" in anc:
        return "water_geological"
    if "Music" in anc or name == "Music":
        return "music"
    if "Animal" in anc or name == "Animal":
        return "birds_wildlife"
    if name in APPLIANCES:
        return "engines_machinery"
    if name in INDUSTRIAL_TOOLS or "Tools" in anc:
        return "industrial"
    if "Engine" in anc or name == "Engine" or "Mechanisms" in anc:
        return "engines_machinery"
    if "Vehicle" in anc or name == "Vehicle":
        return "vehicles"
    if "Human sounds" in anc or "Domestic sounds, home sounds" in anc or name in {"Door", "Footsteps"}:
        return "human_nonspeech"
    if name in {"Vibration", "Throbbing", "Mechanical fan"}:
        return "engines_machinery"
    if "Natural sounds" in anc or name in {"Fire", "Crackle"}:
        return "environmental"
    return "environmental"


@lru_cache(maxsize=1)
def class_table() -> list[dict]:
    """527 AudioSet classes with EchoMap category, depth and alternative-group id."""
    out = []
    for c in audioset_meta():
        grp = next((i for i, g in enumerate(ALTERNATIVE_GROUPS) if c["name"] in g), None)
        out.append({**c, "category": categorize(c["name"], c["ancestors"]), "depth": len(c["ancestors"]),
                    "alt_group": grp})
    return out


def is_ancestor(a: str, b_ancestors: list[str]) -> bool:
    return a in b_ancestors


# classes too generic to stand alone when a more specific overlapping event exists
GENERIC = {"Animal", "Vehicle", "Motor vehicle (road)", "Engine", "Speech", "Music", "Bird", "Inside, small room",
           "Environmental noise", "Sound effect", "Domestic animals, pets", "Wild animals", "Livestock, farm animals, working animals",
           "Mechanisms", "Tools", "Alarm", "Bell", "Water", "Liquid", "Human sounds", "Natural sounds", "Noise",
           "Outside, urban or manmade", "Outside, rural or natural", "Silence"}
