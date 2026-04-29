from collections import Counter
from typing import Any


DEFAULT_CLASS_NAMES = {
    0: "correct_mask",
    1: "incorrect_mask",
    2: "no_mask",
}


def parse_ultralytics_result(result: Any) -> list[dict]:
    names = getattr(result, "names", DEFAULT_CLASS_NAMES)
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    detections: list[dict] = []
    for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        xyxy = [round(float(value), 2) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, DEFAULT_CLASS_NAMES.get(class_id, str(class_id))),
                "confidence": round(confidence, 4),
                "bbox": xyxy,
            }
        )
    return detections


def count_detections(detections: list[dict]) -> dict[str, int]:
    counts = Counter(detection["class_name"] for detection in detections)
    return {
        "correct_mask": counts.get("correct_mask", 0),
        "incorrect_mask": counts.get("incorrect_mask", 0),
        "no_mask": counts.get("no_mask", 0),
    }

