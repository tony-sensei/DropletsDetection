# Droplet Detection and Tracking Pipeline

This project aims to extract frames from videos, label them, train a droplet detection model and perform tracking on droplets. If you are looking to only perform droplet tracking with existing models and videos, you can skip steps 1-3.

## Installation and Setup

1. First, you'll need to create a Python virtual environment to keep your project dependencies isolated. If you're using Python 3, you can create a new virtual environment using the following command:

    ```bash
    python3 -m venv env
    ```

2. After creating the virtual environment, activate it:

    For Windows:
    ```bash
    .\env\Scripts\activate
    ```
    For Unix or MacOS:
    ```bash
    source env/bin/activate
    ```

3. Install the required packages using the requirements.txt file:

    ```bash
    pip install -r requirements.txt
    ```

## Pipeline

### 1. Frame Extraction

Use the `frameExtraction.py` script to extract frames from videos. You can modify the source video paths, output folder paths, and frame_rate within the script.

Command to run the script:
```bash
python frameExtraction.py
```

### 2. Labeling

Use [Roboflow](https://app.roboflow.com/) for image labeling, splitting the datasets into train-val-test and generating code snippets for the datasets, which will be used in step 3.

A detailed tutorial on how to do this can be found on the Roboflow YouTube channel [here](https://www.youtube.com/watch?v=wuZtUMEiKWY&t=768s&ab_channel=Roboflow).

### 3. Droplet Detection

Go through each code block in the `dropletsDetection.ipynb` notebook and modify some of the codes based on comments and the Roboflow tutorial. The output of this step will be `best.pt`, the trained model for droplet detection.

### 4. Droplet Tracking

Use the `tracking.py` script to track the droplets. You may need to modify the code based on different file paths.

Command to run the script:
```bash
python tracking.py
```

## Credits

This project has been developed with the help of various resources:

- Roboflow's video tutorial: [link](https://www.youtube.com/watch?v=wuZtUMEiKWY&t=768s&ab_channel=Roboflow)
- The PythonCode's tutorial on real-time object tracking with YOLOv8 and OpenCV: [link](https://www.thepythoncode.com/article/real-time-object-tracking-with-yolov8-opencv)

