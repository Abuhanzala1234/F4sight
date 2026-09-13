# MODELS.md — every weight file, where it came from, what it costs

**It costs nothing.** That is the point of this file. Every model below is free to
download and run locally, forever, with no account, no API key, and no metered call.
`make models` fetches and exports all of them into `models/` and writes
`models/MANIFEST.json` with a SHA-256 per file. The worker verifies that manifest at
startup and refuses to run on a mismatch.

If you are adding a model: it goes in this table first, with its licence, or it does not
go in.

---

## 1. Active models

| Key | File | Task | Source | Licence | Size | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| `detector.yolo11n` | `models/detect/yolo11n.onnx` | person/vehicle/animal/bag detection | Ultralytics YOLO11n, exported to ONNX opset 12 | **AGPL-3.0** | 10.2 MB | ₹0 |
| `detector.yolo11s` | `models/detect/yolo11s.onnx` | same, `bop` profile | Ultralytics YOLO11s | **AGPL-3.0** | 36.4 MB | ₹0 |
| `detector.rtdetr_r18` | `models/detect/rtdetr-r18.onnx` | permissive-licence fallback | RT-DETR R18 (PaddleDetection → ONNX) | **Apache-2.0** | 77 MB | ₹0 |
| `plate.detect` | `models/anpr/plate-yolo11n.onnx` | licence-plate localisation | YOLO11n fine-tuned on CCPD + open Indian-plate sets | **AGPL-3.0** (weights: dataset terms) | 10.2 MB | ₹0 |
| `plate.rec` | `models/anpr/en_PP-OCRv4_rec.onnx` | plate text recognition | PaddleOCR v4 English recogniser | **Apache-2.0** | 10.8 MB | ₹0 |
| `plate.det` | `models/anpr/ch_PP-OCRv4_det.onnx` | text-region detection in the crop | PaddleOCR v4 detector | **Apache-2.0** | 4.7 MB | ₹0 |
| `face.detect` | `models/face/scrfd_500m.onnx` | face detection *(opt-in)* | InsightFace `buffalo_s` | **MIT** (code), free weights | 3.3 MB | ₹0 |
| `face.embed` | `models/face/w600k_mbf.onnx` | 512-d face embedding *(opt-in)* | InsightFace `buffalo_s` ArcFace | **MIT** (code), free weights | 13.0 MB | ₹0 |
| `enhance.zerodce` | `models/enhance/zero_dce_pp.onnx` | low-light curve enhancement | Zero-DCE++ | **MIT** | 0.03 MB | ₹0 |

**Total download: ~166 MB.** One time. Then the box can be offline forever (P9).

No weights are needed for: ByteTrack (algorithm, reimplemented in `track.py`), dark
channel prior dehaze, CLAHE, or the Merkle/JCS machinery.

---

## 2. Licence position — read this before the demo Q&A

A judge may ask "is this actually free?" The honest answer, in three parts:

1. **Everything runs at zero marginal cost.** No inference is billed. No cloud account
   exists. Unplug the network and `make demo` still works.
2. **Ultralytics YOLO is AGPL-3.0.** Free to use and modify; the obligation is that if you
   distribute a modified version or offer it as a network service, you publish the source.
   For a government deployment of an in-house system that is an acceptable posture, and
   this repository is source-available anyway.
3. **If AGPL is unacceptable to the deploying organisation**, set
   `detector.backend: rtdetr` in `config/detector.yaml`. RT-DETR-R18 is Apache-2.0 and the
   code path is otherwise identical — §7.4 was designed for exactly this swap. Accuracy on
   our eval set drops ~2 points mAP; nothing else changes.

InsightFace model weights are distributed for free use; the face module ships **disabled**
(P6), so nothing loads them unless an admin turns the feature on deliberately.

---

## 3. Why not the obvious alternatives

| Alternative | Why not |
| --- | --- |
| Hosted vision APIs (any vendor) | Metered, key-gated, and useless at a BOP with no uplink. Violates P9 and P10. |
| NVIDIA DeepStream / TAO models | Free to download but NVIDIA-hardware-locked; breaks "runs on the evaluator's laptop". Kept as an optional `bop`-profile accelerator only. |
| Fine-tuning a large VLM for scene description | Interesting, not demoable in a hackathon, and unexplainable — a VLM cannot produce the additive risk breakdown P2 requires. |
| Commercial ANPR SDK | Better on Indian plates, costs per camera per year. We close the gap with multi-frame voting (§7.9) instead of money. |

---

## 4. `make models`

```bash
make models          # download + export everything, verify hashes
make models FORCE=1  # re-download even if present
```

The script (`scripts/fetch_models.py`):

1. downloads each artefact from its upstream release URL;
2. exports the Ultralytics `.pt` files to ONNX (opset 12, dynamic batch, simplified);
3. computes SHA-256 for every file;
4. writes `models/MANIFEST.json`;
5. prints a licence summary — so nobody can claim they were not told.

Air-gapped install: run `make models` on a connected machine, copy the whole `models/`
directory across, done. The manifest makes the copy verifiable.

---

## 5. Upgrade paths (documented, not taken)

| If you later need | Swap to | Still free? |
| --- | --- | --- |
| Better small-object recall at range | YOLO11m + tiled inference (§7.4 supports `crop_x/crop_y`) | Yes |
| Better Indian plate accuracy | Fine-tune `plate.rec` on a local plate corpus | Yes |
| Thermal/IR cameras | Retrain detector on FLIR ADAS (free for research) | Yes |
| Person re-ID across cameras | OSNet / FastReID (both free), + pgvector | Yes |

Every upgrade path stays inside the free stack. There is no point at which this system
requires a purchase to keep working.
