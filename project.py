import cv2
from ultralytics import YOLO

class TrafficVehicelCounter():

    def __init__(self):
        self.detector_zone = [[20, 150], [500, 350]]
        self.sensitive_zone = [245, 260]
        self.x_pos_limit = 7
        self.car_id = 2
        self.motor_id = 3
        self.model = YOLO("models/yolov8n")
     
    def vehicle_detector(self):
        motor_cnt = 0
        car_cnt = 0
        vehicle_lst = [[1, 1]]
        cap = cv2.VideoCapture("res/roud.mp4")
        while True:
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                new_size = (int(width/2), int(height/2))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
                ltl_frame = frame[self.detector_zone[0][1]:self.detector_zone[1][1], 
                                  self.detector_zone[0][0]:self.detector_zone[1][0]]
                
                self.set_label(motor_cnt, car_cnt, frame)

                results = self.model(source=ltl_frame)
                result = results[0]
                for (obj_xyxy, obj_cls) in zip(result.boxes.xyxy, result.boxes.cls):
                    obj_cls = int(obj_cls)
                    x1, y1, x2, y2 = self.get_coordinates(obj_xyxy)
                    x_cent, y_cent = self.get_center(x1, y1, x2, y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.rectangle(frame, (self.detector_zone[0][0], self.sensitive_zone[0]), 
                                (self.detector_zone[1][0], self.sensitive_zone[1]), (0, 0, 255), 1)
                    
                    if self.sensitive_zone[0] < y_cent < self.sensitive_zone[1]:
                        if abs(vehicle_lst[-1][0] - x_cent) > self.x_pos_limit or vehicle_lst[-1][1] != obj_cls:
                            if obj_cls == self.car_id:
                                car_cnt = car_cnt + 1
                                vehicle_lst.append([x_cent, obj_cls])
                            elif obj_cls == self.motor_id:
                                motor_cnt = motor_cnt + 1
                                vehicle_lst.append([x_cent, obj_cls])
                cv2.imshow("Video", frame)
                q = cv2.waitKey(1)
                if q == ord('q'):
                    break

    def set_label(self, motor_cnt, car_cnt, frame):
        car_cnt_title = f"Cars count: {car_cnt}"
        COLOR = (0, 0, 255)
        cv2.putText(frame, car_cnt_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)
                
        motor_cnt_title = f"Motors count: {motor_cnt}"
        COLOR = (0, 0, 255)
        cv2.putText(frame, motor_cnt_title, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)

    def get_coordinates(self, obj_xyxy):
        x1 = int(obj_xyxy[0].item()) + self.detector_zone[0][0]
        y1 = int(obj_xyxy[1].item()) + self.detector_zone[0][1]
        x2 = int(obj_xyxy[2].item()) + self.detector_zone[0][0]
        y2 = int(obj_xyxy[3].item()) + self.detector_zone[0][1]
        return x1, y1, x2, y2
    
    def get_center(self, x1, y1, x2, y2):
        x_cent = int((x2 + x1) / 2)
        y_cent = int((y2 + y1) / 2)
        return x_cent, y_cent
    
trafficVehicelCounter = TrafficVehicelCounter()
trafficVehicelCounter.vehicle_detector()
