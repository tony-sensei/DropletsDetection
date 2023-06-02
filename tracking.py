from ultralytics import YOLO
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


def create_video_writer(video_cap, output_path):
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    return cv2.VideoWriter(output_path, codec, fps, (width, height))


def main(input_video: str, model_path: str, output_video: str, confidence_threshold: float, max_age: int):
    # initialize the video capture object
    video_cap = cv2.VideoCapture(input_video)

    # Check if the video capture object is opened correctly
    if not video_cap.isOpened():
        print("Error: Video file could not be opened.")
        return

    print("Video file opened successfully.")
    print("Total frames in the video:", int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # initialize the video writer object
    writer = create_video_writer(video_cap, output_video)

    # Set device to 'cuda' or 'cpu' or 'mps' explicitly.
    device = torch.device('cuda' if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else 'cpu')

    # load the pre-trained YOLOv8 model
    model = YOLO(model_path)
    tracker = DeepSort(max_age=max_age)

    while video_cap.isOpened():
        ret, frame = video_cap.read()

        if not ret:
            print("Video is ended")
            break

        # run the YOLO model on the frame
        detections = model(frame, device=device)[0]

        # initialize the list of bounding boxes and confidences
        results = []

        ######################################
        # DETECTION
        ######################################

        # loop over the detections
        for data in detections.boxes.data.tolist():
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = data[4]

            # filter out weak detections by ensuring the
            # confidence is greater than the minimum confidence
            if float(confidence) < confidence_threshold:
                continue

            # if the confidence is greater than the minimum confidence,
            # get the bounding box and the class id
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            # add the bounding box (x, y, w, h), confidence and class id to the results list
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        ######################################
        # TRACKING
        ######################################

        # update the tracker with the new detections
        tracks = tracker.update_tracks(results, frame=frame)
        # loop over the tracks
        for track in tracks:
            # if the track is not confirmed, ignore it
            if not track.is_confirmed():
                continue

            # get the track id and the bounding box
            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                ltrb[1]), int(ltrb[2]), int(ltrb[3])
            # draw the bounding box and the track id
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        writer.write(frame)

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video = "path/to/input/video.mp4"
    model_path = "path/to/best.pt"
    output_video = "path/to/output.mp4"
    confidence_threshold = 0.6
    max_age = 15
    main(input_video, model_path, output_video, confidence_threshold, max_age)
