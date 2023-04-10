from ultralytics import YOLO
resolution_x = 480
resolution_y = 480
if __name__ == "__main__":
# Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model 
    results = model.train(data="./datasets/data.yaml", imgsz=resolution_x, batch=5, epochs=10, device=0)