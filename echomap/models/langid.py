"""Spoken-language identification with the Whisper LID-only graph (no transcription is possible)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .frontends import whisper_log_mel
from .runtime import session

LANG_NAMES = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian", "ko": "Korean", "fr": "French",
    "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "ca": "Catalan", "nl": "Dutch",
    "ar": "Arabic", "sv": "Swedish", "it": "Italian", "id": "Indonesian", "hi": "Hindi", "fi": "Finnish",
    "vi": "Vietnamese", "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay", "cs": "Czech",
    "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian", "th": "Thai",
    "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin", "mi": "Maori",
    "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak", "te": "Telugu", "fa": "Persian", "lv": "Latvian",
    "bn": "Bengali", "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada", "et": "Estonian",
    "mk": "Macedonian", "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian", "ne": "Nepali",
    "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili", "gl": "Galician",
    "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer", "sn": "Shona", "yo": "Yoruba",
    "so": "Somali", "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian", "be": "Belarusian", "tg": "Tajik",
    "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic", "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek",
    "fo": "Faroese", "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk", "mt": "Maltese",
    "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Myanmar", "bo": "Tibetan", "tl": "Tagalog", "mg": "Malagasy",
    "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa", "ba": "Bashkir",
    "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}


@lru_cache(maxsize=2)
def languages(models_dir: str) -> list[str]:
    return json.loads((Path(models_dir) / "whisper_base_languages.json").read_text(encoding="utf-8"))


def identify(x16k: np.ndarray, models_dir: str | Path, top: int = 3) -> list[dict]:
    md = Path(models_dir)
    mel_f = np.load(md / "whisper_base_mel_filters.npy")
    sess = session(md / "whisper_base_lid_int8.onnx")
    feat = whisper_log_mel(x16k, mel_f)[None]
    p = sess.run(None, {"input_features": feat})[0][0]
    codes = languages(str(md))
    order = np.argsort(-p)[:top]
    return [{"code": codes[i], "name": LANG_NAMES.get(codes[i], codes[i]), "prob": float(p[i])} for i in order]
