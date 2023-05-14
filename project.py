import cv2
from ultralytics import YOLO

class TrafficVehicelCounter():

    def __init__(self):
        self.detector_zone = [[20, 150], [500, 350]]
        self.sensitive_zone = [240, 260]
        self.model = YOLO("models/yolov8n")
     
    def vehicle_detector(self):
        motor_cnt = 0
        car_cnt = 0
        cap = cv2.VideoCapture("res/roud.mp4")
        while True:
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                new_size = (int(width/2), int(height/2))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
                ltl_frame = frame[self.detector_zone[0][1]:self.detector_zone[1][1], 
                                  self.detector_zone[0][0]:self.detector_zone[1][0]]
                
                car_cnt_title = f"Car count: {car_cnt}"
                BLACK = (0, 0, 0)
                cv2.putText(frame, car_cnt_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)
                
                motor_cnt_title = f"Motor count: {motor_cnt}"
                BLACK = (0, 0, 0)
                cv2.putText(frame, motor_cnt_title, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)

                results = self.model(source=ltl_frame)
                result = results[0]
                for (obj_xyxy, obj_cls) in zip(result.boxes.xyxy, result.boxes.cls):
                    obj_cls = int(obj_cls)
                    x1 = int(obj_xyxy[0].item()) + self.detector_zone[0][0]
                    y1 = int(obj_xyxy[1].item()) + self.detector_zone[0][1]
                    x2 = int(obj_xyxy[2].item()) + self.detector_zone[0][0]
                    y2 = int(obj_xyxy[3].item()) + self.detector_zone[0][1]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    # x_cent = (x2 + x1) / 2
                    y_cent = int((y2 + y1) / 2)
                    cv2.rectangle(frame, (self.detector_zone[0][0], self.sensitive_zone[0]), 
                                (self.detector_zone[1][0], self.sensitive_zone[1]), (0, 0, 255), 1)
                    if self.sensitive_zone[0] < y_cent < self.sensitive_zone[1]:
                        if obj_cls == 2:
                            car_cnt = car_cnt + 1
                        elif obj_cls == 3:
                            motor_cnt = motor_cnt + 1
                cv2.imshow("Video", frame)
                q = cv2.waitKey(1)
                if q == ord('q'):
                    break

trafficVehicelCounter = TrafficVehicelCounter()
trafficVehicelCounter.vehicle_detector()
