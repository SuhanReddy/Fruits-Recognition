"""Live fruit recognition from your webcam.

Hold a fruit inside the green box. Press 'q' to quit.

Example:
    python webcam.py
"""
import argparse

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from data import get_transforms
from model import get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Live fruit recognition from a webcam")
    parser.add_argument("--checkpoint", default="models/fruit_classifier.pth")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0 = default webcam)")
    parser.add_argument("--box", type=int, default=300, help="Size of the scan box in pixels")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Below this confidence the program says it is unsure")
    args = parser.parse_args()

    device = get_device()
    model, class_names, img_size = load_checkpoint(args.checkpoint, device)
    transform = get_transforms(img_size, train=False)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Try --camera 1, and close other apps using it.")

    label, conf, frame_count = "", 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view so it feels natural

        h, w = frame.shape[:2]
        size = min(args.box, h, w)
        x1, y1 = (w - size) // 2, (h - size) // 2
        x2, y2 = x1 + size, y1 + size

        # Classify the box every 5th frame to keep things smooth on a CPU
        if frame_count % 5 == 0:
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(crop)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(model(tensor), dim=1)[0]
            top_prob, top_idx = probs.max(dim=0)
            label, conf = class_names[int(top_idx)], float(top_prob)
        frame_count += 1

        text = f"{label} ({conf * 100:.0f}%)" if conf >= args.threshold else "Unsure - centre the fruit in the box"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Fruit recognition - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
