from ultralytics import YOLO

model1 = YOLO("RoadDetectionModel/RoadModel_yolov8m.pt_rounds120_b9/weights/best.pt")
print("Model 1 classes:", model1.names)

model2 = YOLO("YOLOv8_Small_2nd_Model.pt")
print("Model 2 classes:", model2.names)