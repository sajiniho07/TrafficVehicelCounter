import cv2
from ultralytics import YOLO

class TrafficVehicelCounter():

    def __init__(self):
        self.sensitive_area_x1 = 40
        self.sensitive_area_y1 = 300
        self.sensitive_area_x2 = 1000
        self.sensitive_area_y2 = 700
        self.model = YOLO("models/yolov8n")
     
    def vehicle_detector(self):
        motor_cnt = 0
        car_cnt = 0
        cap = cv2.VideoCapture("res/mini_roud.mp4")
        while True:
            ret, frame = cap.read()
            if ret:
                width, height = frame.shape[1], frame.shape[0]
                ltl_frame = frame[self.sensitive_area_y1:self.sensitive_area_y2, self.sensitive_area_x1:self.sensitive_area_x2]
                results = self.model.track(source=ltl_frame)
                for result in results:
                    ooo = result.keypoints
                    for (obj_xyxy, obj_cls) in zip(result.boxes.xywh, result.boxes.cls):
                        obj_cls = int(obj_cls)
                        x1 = int(obj_xyxy[0].item()) + self.sensitive_area_x1
                        y1 = int(obj_xyxy[1].item()) + self.sensitive_area_y1
                        x2 = int(obj_xyxy[2].item()) + self.sensitive_area_x1
                        y2 = int(obj_xyxy[3].item()) + self.sensitive_area_y1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        x_cent = (x2 + x1) / 2
                        y_cent = (y2 + y1) / 2
                        if obj_cls == 3:
                            # print(keypoints)
                            car_cnt = car_cnt + 1
                        elif obj_cls == 4:
                            motor_cnt = motor_cnt + 1
            cv2.imshow("Video", frame)
            q = cv2.waitKey(1)
            if q == ord('q'):
                break
        print(f"car_cnt={car_cnt}")
        print(f"motor_cnt={motor_cnt}")

trafficVehicelCounter = TrafficVehicelCounter()
trafficVehicelCounter.vehicle_detector()
