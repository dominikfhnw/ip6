import cv2
import numpy as np

from typing import Optional, Tuple
from pyk4a import ImageFormat
import pyk4a
from pyk4a import Config, PyK4A

def colorize(
        image: np.ndarray,
        clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
        colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    #img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #img = cv2.applyColorMap(img, colormap)
    return img
def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.OFF,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=False,
        )
    )
    k4a.start()

    # getters and setters directly get and set on device
    #k4a.whitebalance = 4500
    #assert k4a.whitebalance == 4500
    #k4a.whitebalance = 4510
    #assert k4a.whitebalance == 4510

    while True:
        capture = k4a.get_capture()
        if np.any(capture.depth):
            cv2.imshow("k4a", colorize(capture.depth, (250, 500), cv2.COLORMAP_JET))
            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break
    k4a.stop()


if __name__ == "__main__":
    main()