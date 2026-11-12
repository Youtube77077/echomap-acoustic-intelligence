"""Export EfficientAT mn10_as (MIT, Florian Schmid) to ONNX.  Runs in GitHub Actions only.

Usage: python export_efficientat.py <EfficientAT checkout> <out dir>

Produces in <out dir>:
  efficientat_mn10_as.onnx       input  mel  [B, 1, 128, T] (normalised log-mel from frontends.efficientat_mel)
                                 output probs [B, 527] (sigmoid), embedding [B, 960]
  efficientat_mn10_as_melbank.npy  (128, 513) kaldi mel filterbank used by AugmentMelSTFT (eval)
  audioset_labels.csv            index,mid,display_name
  efficientat_parity.json        numpy-frontend + ONNX vs PyTorch reference
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

repo, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, repo)
os.chdir(repo)  # EfficientAT uses relative paths (resources/, metadata/)

from models.mn.model import get_model  # noqa: E402
from models.preprocess import AugmentMelSTFT  # noqa: E402

sys.path.insert(0, os.environ["ECHOMAP_SRC"])
from echomap.models.frontends import efficientat_mel  # noqa: E402

torch.manual_seed(0)
model = get_model(width_mult=1.0, pretrained_name="mn10_as", strides=(2, 2, 2, 2), head_type="mlp")
model.eval()
mel_ref = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320)
mel_ref.eval()


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mel):
        logits, emb = self.m(mel)
        return torch.sigmoid(logits), emb


w = Wrapper(model).eval()

# mel filterbank exactly as the reference builds it in eval mode
import torchaudio  # noqa: E402

mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(128, 1024, 32000, 0.0, mel_ref.fmax,
                                                         vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
mel_basis = torch.nn.functional.pad(mel_basis, (0, 1), mode="constant", value=0).numpy().astype(np.float32)
np.save(os.path.join(out, "efficientat_mn10_as_melbank.npy"), mel_basis)

dummy = torch.randn(1, 1, 128, 300)
onnx_path = os.path.join(out, "efficientat_mn10_as.onnx")
torch.onnx.export(w, dummy, onnx_path, input_names=["mel"], output_names=["probs", "embedding"],
                  dynamic_axes={"mel": {0: "batch", 3: "frames"}, "probs": {0: "batch"}, "embedding": {0: "batch"}},
                  opset_version=17)

# ---------------------------------------------------------------- parity
import onnxruntime as ort  # noqa: E402

rng = np.random.default_rng(1)
sr = 32000
t = np.arange(sr * 3) / sr
signals = {
    "noise": (rng.standard_normal(sr * 3) * 0.05).astype(np.float32),
    "chirp_tone": (0.3 * np.sin(2 * np.pi * (500 + 800 * t) * t)).astype(np.float32),
    "clicks": np.zeros(sr * 3, np.float32),
}
signals["clicks"][::sr // 2] = 0.9
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
report = {"model": "mn10_as_mAP_471.pt", "commit": os.environ.get("EAT_COMMIT"), "cases": {}}
ok = True
for name, x in signals.items():
    with torch.no_grad():
        ref_mel = mel_ref(torch.from_numpy(x)[None])            # (1, 128, T)
        ref_p, _ = w(ref_mel[:, None])
    np_mel = efficientat_mel(x, mel_basis)
    mel_err = float(np.abs(np_mel - ref_mel[0].numpy()).max())
    p = sess.run(None, {"mel": np_mel[None, None]})[0]
    p_err = float(np.abs(p - ref_p.numpy()).max())
    report["cases"][name] = {"mel_max_abs_err": mel_err, "prob_max_abs_err": p_err,
                             "frames_np": int(np_mel.shape[1]), "frames_ref": int(ref_mel.shape[2])}
    ok &= (np_mel.shape[1] == ref_mel.shape[2]) and mel_err < 1e-3 and p_err < 1e-3
report["passed"] = bool(ok)
json.dump(report, open(os.path.join(out, "efficientat_parity.json"), "w"), indent=2)
shutil.copy(os.path.join(repo, "metadata", "class_labels_indices.csv"), os.path.join(out, "audioset_labels.csv"))
print(json.dumps(report, indent=2))
if not ok:
    sys.exit("PARITY FAILED")
