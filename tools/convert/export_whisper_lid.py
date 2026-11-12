"""Export a *language-identification-only* Whisper graph to ONNX.  Runs in GitHub Actions only.

The graph is: encoder + one decoder step from <|startoftranscript|> + projection onto the
language-token rows only. It outputs a softmax over Whisper's languages, and it CANNOT produce
text tokens (no vocabulary projection is included). That is deliberate, for privacy.

Usage: python export_whisper_lid.py <model id, e.g. openai/whisper-base> <out dir>
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

model_id, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, os.environ["ECHOMAP_SRC"])
from echomap.models.frontends import whisper_log_mel  # noqa: E402

tag = model_id.split("/")[-1].replace("whisper-", "")
fe = WhisperFeatureExtractor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id, attn_implementation="eager").eval()
gen = model.generation_config
lang_to_id = dict(gen.lang_to_id)                     # {"<|en|>": 50259, ...}
langs = sorted(lang_to_id.items(), key=lambda kv: kv[1])
lang_codes = [k.strip("<|>") for k, _ in langs]
lang_ids = torch.tensor([v for _, v in langs])
sot = gen.decoder_start_token_id                      # 50258


class LID(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.encoder = m.model.encoder
        dec = m.model.decoder
        self.dec_layers = dec.layers
        self.dec_ln = dec.layer_norm
        self.sot_emb = torch.nn.Parameter(dec.embed_tokens.weight[sot].detach().clone()[None, None])
        self.pos0 = torch.nn.Parameter(dec.embed_positions.weight[0].detach().clone()[None, None])
        self.lang_proj = torch.nn.Parameter(m.proj_out.weight[lang_ids].detach().clone())  # (L, d)

    def forward(self, input_features):
        enc = self.encoder(input_features).last_hidden_state
        h = self.sot_emb + self.pos0
        for layer in self.dec_layers:
            o = layer(h, encoder_hidden_states=enc)
            h = o[0] if isinstance(o, tuple) else o
        h = self.dec_ln(h)[:, -1]
        return torch.softmax(h @ self.lang_proj.T, dim=-1)


lid = LID(model).eval()

# reference: full model, first-step logits restricted to language tokens
rng = np.random.default_rng(0)


def tts(text, voice, path):
    subprocess.run(["espeak-ng", "-v", voice, "-s", "150", "-w", path, text], check=True)
    import soundfile as sf
    import librosa
    y, sr = sf.read(path, dtype="float32")
    return librosa.resample(y, orig_sr=sr, target_sr=16000).astype(np.float32)


samples = {
    "en": tts("The weather today is cold and the train to the city centre is late again.", "en", "/tmp/en.wav"),
    "de": tts("Das Wetter ist heute kalt und der Zug in die Innenstadt hat wieder Verspätung.", "de", "/tmp/de.wav"),
    "es": tts("Hoy hace frío y el tren al centro de la ciudad llega tarde otra vez.", "es", "/tmp/es.wav"),
    "nl": tts("Het weer is vandaag koud en de trein naar het centrum is weer te laat.", "nl", "/tmp/nl.wav"),
    "fr": tts("Il fait froid aujourd'hui et le train pour le centre-ville est encore en retard.", "fr", "/tmp/fr.wav"),
}
mel_filters = fe.mel_filters.T.astype(np.float32)   # (80, 201)
np.save(os.path.join(out, f"whisper_{tag}_mel_filters.npy"), mel_filters)

fp32_path = os.path.join(out, f"whisper_{tag}_lid_fp32.onnx")
dummy = torch.from_numpy(whisper_log_mel(samples["en"], mel_filters))[None]
torch.onnx.export(lid, dummy, fp32_path, input_names=["input_features"], output_names=["lang_probs"],
                  dynamic_axes={"input_features": {0: "batch"}, "lang_probs": {0: "batch"}},
                  opset_version=17)

from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402
import onnxruntime as ort  # noqa: E402

int8_path = os.path.join(out, f"whisper_{tag}_lid_int8.onnx")
quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

report = {"model": model_id, "languages": len(lang_codes), "cases": {}}
ok = True
for code, wav in samples.items():
    ref_feat = fe(wav, sampling_rate=16000, return_tensors="np").input_features[0]
    np_feat = whisper_log_mel(wav, mel_filters)
    feat_err = float(np.abs(ref_feat - np_feat).max())
    with torch.no_grad():
        full = model(input_features=torch.from_numpy(ref_feat)[None],
                     decoder_input_ids=torch.tensor([[sot]])).logits[0, -1]
        ref_p = torch.softmax(full[lang_ids], -1).numpy()
        lid_p = lid(torch.from_numpy(np_feat)[None])[0].numpy()
    q_p = sess.run(None, {"input_features": np_feat[None]})[0][0]
    top = lang_codes[int(q_p.argmax())]
    report["cases"][code] = {"feature_max_abs_err": feat_err,
                             "lid_vs_full_max_abs_err": float(np.abs(ref_p - lid_p).max()),
                             "int8_vs_full_max_abs_err": float(np.abs(ref_p - q_p).max()),
                             "ref_top1": lang_codes[int(ref_p.argmax())], "int8_top1": top,
                             "int8_top1_prob": float(q_p.max())}
    ok &= feat_err < 1e-3 and float(np.abs(ref_p - lid_p).max()) < 1e-3 and top == lang_codes[int(ref_p.argmax())]
report["passed"] = bool(ok)
json.dump(lang_codes, open(os.path.join(out, f"whisper_{tag}_languages.json"), "w"))
json.dump(report, open(os.path.join(out, f"whisper_{tag}_parity.json"), "w"), indent=2)
os.remove(fp32_path)
print(json.dumps(report, indent=2))
if not ok:
    sys.exit("PARITY FAILED")
