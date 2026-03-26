import cv2
import numpy as np

from detasks.objects import (
    Annotation,
    Sample,
    distribution_count,
    distribution_size,
    make_objects,
)


def plot(frame: np.ndarray, sample: Sample[Annotation]) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    for ann in sample.annotations:
        x1, y1, x2, y2 = ann.bbox
        x1, y1 = int(x1 * w), int(y1 * h)
        x2, y2 = int(x2 * w), int(y2 * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            img,
            ann.label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )

    return img


def test_objects():
    objects = make_objects(
        n_samples=100,
        distribution_count=distribution_count,
        distribution_size=distribution_size,
    )
    print(objects)
