# Real-World Test Set Plan

Screen recording is useful for a demo video, but it is not enough for evaluation. A real-world test set should contain saved source frames plus metadata so the model can be evaluated and debugged later.

## Target Size

Start with 100-200 webcam frames:

| Bucket | Target frames |
| --- | ---: |
| correct_mask normal | 30 |
| no_mask normal | 30 |
| incorrect_mask under nose | 30 |
| incorrect_mask under chin / neck | 30 |
| low light / backlight | 20 |
| hand or arm occlusion | 20 |
| motion blur / head movement | 20 |

The buckets can overlap through metadata. For example, one image can be `incorrect_mask`, subtype `under_chin`, lighting `low_light`, occlusion `none`.

## Label Policy

```text
correct_mask:
  mask covers nose and mouth

incorrect_mask:
  mask is visible but worn incorrectly, including under nose, under chin, or under neck

no_mask:
  no visible mask on the face/person
```

Compliance grouping:

```text
compliant = correct_mask
violation = incorrect_mask + no_mask
```

## Capture Workflow

Run the local demo:

```bash
source .venv/bin/activate
MODEL_PATH=models/best.pt bash scripts/run_server.sh
```

Open:

```text
http://localhost:8000
```

Use the `Test Capture` panel:

```text
1. Set ground-truth label.
2. Set subtype and edge-case metadata.
3. Click "Capture test frame".
4. Repeat until each bucket has enough examples.
```

Saved output:

```text
outputs/real_world_test/images/
outputs/real_world_test/metadata.csv
```

## What To Report

After collecting frames, report:

```text
class distribution
edge-case distribution
examples where prediction differs from label
under-chin/under-neck confusion: incorrect_mask vs no_mask
low-light and occlusion failure cases
```

Do not train on this set initially. Use it as an external evaluation set first.
