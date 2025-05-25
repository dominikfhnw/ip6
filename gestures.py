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

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

globalresult = None


def get_point(detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Get the top left corner of the detected hand's bounding box.
        # height, width, _ = rgb_image.shape
        height = meta.get("height")
        width = meta.get("width")
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        return text_x, text_y
    return False


def draw_landmarks_on_image(rgb_image, detection_result):
    has_gesture = meta.true("gestures")
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    if has_gesture:
        gestures_list = detection_result.gestures
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        if has_gesture:
            gestures = gestures_list[idx]

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
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        if has_gesture:
            text = f"{handedness[0].category_name} {gestures[0].category_name}"
        else:
            text = f"{handedness[0].category_name}"

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
        meta.set("result", text)
        meta.set("res_point", get_point(result))
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


if meta.true("async_gestures"):
    log("running asynchronously")
    mode = VisionRunningMode.LIVE_STREAM
    callback = print_result
else:
    log("running synchronously")
    mode = VisionRunningMode.VIDEO
    callback = None

if not meta.true("gestures"):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=mode,
        result_callback=callback,
        num_hands=meta.get("hands"),
        min_hand_detection_confidence=0.3,  # lower than value to get predictions more often
        min_hand_presence_confidence=0.3,  # lower than value to get predictions more often
    )
    rec = HandLandmarker.create_from_options(options)
else:
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
        running_mode=mode,
        result_callback=callback,
        num_hands=meta.get("hands"),
        min_hand_detection_confidence=0.3,  # lower than value to get predictions more often
        min_hand_presence_confidence=0.3,  # lower than value to get predictions more often
        # NB: spelling error in documentation!
        # canned_gestures_classifier_options = ...
        canned_gesture_classifier_options=ClassifierOptions(
            score_threshold=0,
            max_results=-1
        )
    )
    rec = GestureRecognizer.create_from_options(options)
