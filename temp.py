from IPython import display
from ultralytics import YOLO
import supervision as sv
import cv2
import os
from IPython.display import Image

display.clear_output()


def vehicle_count(source_path, destination_path, line_start, line_end):
    model = YOLO("models/yolov8n")

    line_start = sv.Point(line_start[0], line_start[1])
    line_end = sv.Point(line_end[0], line_end[1])
    line_counter = sv.LineZone(line_start, line_end)

    line_annotator = sv.LineZoneAnnotator(thickness=2,
                                          text_thickness=1,
                                          text_scale=0.5)

    box_annotator = sv.BoxAnnotator(thickness=1,
                                    text_thickness=1,
                                    text_scale=0.5)

    video_info = sv.VideoInfo.from_video_path(source_path)

    video_name = os.path.splitext(os.path.basename(source_path))[0] + ".mp4"
    video_out_path = os.path.join(destination_path, video_name)

    video_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(
        *'mp4v'), video_info.fps, (video_info.width, video_info.height))

    for result in model.track(source=source_path, tracker='bytetrack.yaml', show=True, stream=True, agnostic_nms=True):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        detections = detections[(detections.class_id == 2)
                                | (detections.class_id == 7)]
        labels = [f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                  for _, confidence, class_id, tracker_id
                  in detections]
        line_counter.trigger(detections)
        line_annotator.annotate(frame, line_counter)
        frame = box_annotator.annotate(scene=frame,
                                       detections=detections,
                                       labels=labels)
        video_out.write(frame)
    video_out.release()
    display.clear_output()

    vehicle_count(source_path="res/mini_roud.mp4",
                  destination_path="res",
                  line_start=(337, 391),
                  line_end=(917, 387))


gifPath = 'res/test.gif'
Image(filename=gifPath)
