from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(data='D:/AI-Project/kaggle/input/yyaaml/dataset.yaml', epochs=10)