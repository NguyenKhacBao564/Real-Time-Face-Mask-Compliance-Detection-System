import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--data", default="data/mask_detection/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.val(data=args.data, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()

