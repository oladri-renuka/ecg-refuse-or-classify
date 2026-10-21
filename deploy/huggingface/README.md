---
title: ECG Quality Gate
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: 12-lead ECG classifier that refuses low-quality signals
---

# ECG classification with a signal-quality gate (API)

`POST /classify` with `{"signal": [[...1000 floats...] x 12]}` (mV, 100 Hz, leads I, II, III,
aVR, aVL, aVF, V1-V6). Returns the PTB-XL superclass labels with probabilities, or
`{"status": "low_quality"}` when the autoencoder quality gate rejects the signal.
Interactive docs at `/docs`.

Research prototype trained on PTB-XL (CC BY 4.0); **not a medical device**.
Source, training code and evaluation: https://github.com/oladri-renuka/ecg-quality-gate
