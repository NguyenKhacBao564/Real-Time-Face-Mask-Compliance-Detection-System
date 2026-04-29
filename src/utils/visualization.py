from pathlib import Path

import cv2


CLASS_COLORS = {
    "correct_mask": (40, 180, 90),
    "incorrect_mask": (20, 160, 230),
    "no_mask": (40, 40, 230),
}


def draw_detections(image_path: str, detections: list[dict], output_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{class_name} {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)

