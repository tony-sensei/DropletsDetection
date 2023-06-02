# import packages
import os
import cv2


def extract_frames(video_paths, output_folders, frame_rate):
    """
    Function to extract frames from given videos and save them as .png in respective directories.

    Parameters:
    video_paths (list): List of paths to the videos.
    output_folders (list): List of paths to the folders where frames will be stored.
    frame_rate (int): The desired frame rate for frame extraction.

    Returns:
    num_frames (list): List of number of frames created for each video.
    """
    num_frames = []
    for video_path, output_folder in zip(video_paths, output_folders):
        cap = cv2.VideoCapture(video_path)
        count = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_rate == 0:
                cv2.imwrite(os.path.join(output_folder, f"frame{frame_count}.png"), frame)
                frame_count += 1

            count += 1

        cap.release()
        num_frames.append(frame_count)

    return num_frames


if __name__ == "__main__":
    # import raw data
    video1_path = "video_data/4x4 dispense SD.m4v"
    video2_path = "video_data/Barcode ligation.mp4"
    video3_path = "video_data/Rainbow 11-11-22.m4v"

    video_paths = [video1_path, video2_path, video3_path]
    output_folders = ["image_data/video1_frames", "image_data/video2_frames", "image_data/video3_frames"]

    frame_rate = 10  # Change this to the desired frame rate

    # Extract frames
    for output_folder in output_folders:
        os.makedirs(output_folder, exist_ok=True)

    num_frames = extract_frames(video_paths, output_folders, frame_rate)

    print(f"Number of frames extracted from each video: {num_frames}")
