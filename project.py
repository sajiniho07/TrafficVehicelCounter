from ultralytics import YOLO

class TrafficVehicelCounter():

    def __init__(self):
        self.model = YOLO("models/yolov8n")
     
    def vehicle_detector(self):
        motor_cnt = 0
        car_cnt = 0
        results = self.model("res/mini_roud.mp4", show=True)
        for result in results:
            for obj_cls in result.boxes.cls:
                obj_cls = int(obj_cls)
                if obj_cls == 3:
                    car_cnt = car_cnt + 1
                elif obj_cls == 4:
                    motor_cnt = motor_cnt + 1
        print(f"car_cnt={car_cnt}")
        print(f"motor_cnt={motor_cnt}")

trafficVehicelCounter = TrafficVehicelCounter()
trafficVehicelCounter.vehicle_detector()
