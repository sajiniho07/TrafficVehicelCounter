# Traffic Vehicle Counter

## About

This code is a Python implementation of a traffic vehicle counter that uses the YOLOv8 object detection model. It can be used to count the number of cars and motorcycles passing through a specific zone of a road, as captured by a video camera.

## Installation

Before running the code, you need to install the following libraries:

- cv2
- ultralytics

You can install these packages using pip:

```
pip install opencv ultralytics
```

## Usage

To use the traffic vehicle counter, first create an instance of the TrafficVehicelCounter class. Then, call the vehicle_detector method to start the vehicle detection process. The vehicle_detector method reads frames from a video file, detects vehicles using YOLOv8, and counts the number of cars and motorcycles passing through a specific zone of the road. 

The script will display the video stream with bounding boxes around each detected vehicle and the count of cars and motorcycles in real-time.

Note: The script uses a pre-trained YOLO model provided in the "models/yolov8n" directory. You can replace the model with your own if you wish.

## Example

![sample](https://github.com/sajiniho07/TrafficVehicelCounter/blob/master/res/TrafficVehicleCounter.gif)

## License 

Made with :heart: by <a href="https://github.com/sajiniho07" target="_blank">Sajad Kamali</a>

&#xa0;

<a href="#top">Back to top</a>
