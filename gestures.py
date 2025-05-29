import log
import cv2 as cv
import numpy as np
import meta

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp

log, dbg, logger = log.auto(__name__)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

ClassifierOptions = mp.tasks.components.processors.ClassifierOptions

MARGIN = 20  # pixels
FONT_SIZE = 0.7
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

globalresult = None


def get_point(detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Get the top left corner of the detected hand's bounding box.
        # height, width, _ = rgb_image.shape
        width = meta.get("height")
        height = meta.get("width")
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height)

        return text_x, text_y
    return False


def crossP(a,b,p):
    d = (p[0] - a[0]) * ( b[1]-a[1])  - (p[1]-a[1]) * (b[0]-a[0])
    return d < 0

def topoint(landmark, height, width, num):
    l = landmark[num]
    return int(l.x * width), int(l.y * height)


def draw_landmarks_on_image(rgb_image, detection_result):
    has_gesture = meta.true("gestures")
    hand_landmarks_list = detection_result.hand_landmarks
    hand_world_landmarks_list = detection_result.hand_world_landmarks
    handedness_list = detection_result.handedness
    if has_gesture:
        gestures_list = detection_result.gestures
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        hand_world_landmarks = hand_world_landmarks_list[idx]
        handedness = handedness_list[idx]
        if has_gesture:
            gestures = gestures_list[idx]

        if meta.true("skeleton"):
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        is_left = handedness[0].category_name == "Left"
        index = topoint(hand_landmarks, width, height, 8)
        mid = topoint(hand_landmarks, width, height, 7)
        base = topoint(hand_landmarks, width, height, 5)
        pinkybase = topoint(hand_landmarks, width, height, 17)
        wrist = topoint(hand_landmarks, width, height, 0)

        flip = crossP(base, pinkybase, wrist)
        lightsaber = (index[0] + 4 * (index[0] - mid[0]), index[1] + 4 * (index[1] - mid[1]))
        # lightsaber = (index + vector)
        dbg(f"{index=} {lightsaber=} {base=}")
        if meta.true("lightsaber"):
            if is_left:
                color = (255, 165, 120)
            else:
                color = (120, 165, 255)
            cv.line(annotated_image, pt1=index, pt2=lightsaber, thickness=8, color=color)

        z = hand_landmarks[8].z
        z2 = hand_world_landmarks[8].z
        dbg(f"{detection_result.handedness}")
        # z2 = detection_result.hand_world_landmarks[0].8].z
        dbg(f"{z=} {z2=}")

        if flip:
            index_offset = -120
        else:
            index_offset = 20

        if meta.true("et"):
            cv.circle(annotated_image, center=index, radius=10, color=(0, 165, 255), thickness=-1)
            cv.line(annotated_image, pt1=index, pt2=base, thickness=2, color=(0, 165, 255))
            cv.putText(annotated_image, f"{z * 100: .2f} {z2 * 100: .2f}",
                       (index[0] + index_offset, index[1]), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 255), FONT_THICKNESS, cv.LINE_AA)

        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        if has_gesture:
            text = f"{gestures[0].category_name} {handedness[0].category_name}"
        else:
            text = f"{handedness[0].category_name}"

        if flip ^ is_left:
            text = text + " back"
        else:
            text = text + " palm"

        cv.putText(annotated_image, text,
                   (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                   FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

    return annotated_image


def parse(detection_result):
    has_gesture = meta.true("gestures")
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    if has_gesture:
        gestures_list = detection_result.gestures

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        handedness = handedness_list[idx]
        if has_gesture:
            gestures = gestures_list[idx]
            text = f"{gestures[0].score: .2f} {gestures[0].category_name}"
        else:
            text = f"{handedness[0].category_name}"

    return text


def process_result(result):
    dbg("process_result")
    res2 = get_point(result)
    if res2:
        text = parse(result)
        dbg("print_result res: " + text)
        # meta.set("result", text)
        meta.set("res_point", res2)
    else:
        meta.unset("result")
        meta.unset("res_point")
        pass


def print_result(result, output_image: mp.Image, timestamp_ms: int):
    dbg("print_result")
    global globalresult
    globalresult = result
    process_result(result)
    dbg("end print_result")


def process(img: np.ndarray, t1):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    timestamp_ms = int(t1 * 1000)
    dbg(f"gestures proc {img.shape} {timestamp_ms}")

    if meta.true("async_gestures"):
        if meta.true("gestures"):
            rec.recognize_async(mp_image, timestamp_ms)
        else:
            rec.detect_async(mp_image, timestamp_ms)
        dbg("async started")
        if globalresult:
            # dbg("got result " + str(globalresult.hand_world_landmarks))
            img = draw_landmarks_on_image(img, globalresult)
    else:
        if meta.true("gestures"):
            result = rec.recognize_for_video(mp_image, timestamp_ms)
        else:
            result = rec.detect_for_video(mp_image, timestamp_ms)
        img = draw_landmarks_on_image(img, result)
        process_result(result)

    return img

if meta.true("mediapipe"):
    if meta.true("async_gestures"):
        log("running asynchronously")
        mode = VisionRunningMode.LIVE_STREAM
        callback = print_result
    else:
        log("running synchronously")
        mode = VisionRunningMode.VIDEO
        callback = None

    delegate=BaseOptions.Delegate.CPU
    if meta.true("mediapipe_gpu"):
        delegate=BaseOptions.Delegate.GPU
    if not meta.true("gestures"):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=mode,
            result_callback=callback,
            num_hands=meta.get("hands"),
            min_hand_detection_confidence=meta.get("hand_detection"),  # finding palm of hand
            min_hand_presence_confidence=meta.get("hand_presence"),  # lower than value to get predictions more often
        )
        rec = HandLandmarker.create_from_options(options)
    else:
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path='gesture_recognizer.task', delegate=delegate),
            running_mode=mode,
            result_callback=callback,
            num_hands=meta.get("hands"),
            min_hand_detection_confidence=meta.get("hand_detection"),  # finding palm of hand
            min_hand_presence_confidence=meta.get("hand_presence"),  # lower than value to get predictions more often
            # NB: spelling error in documentation!
            # canned_gestures_classifier_options = ...
            canned_gesture_classifier_options=ClassifierOptions(
                score_threshold=0,
                max_results=-1
            )
        )
        rec = GestureRecognizer.create_from_options(options)
