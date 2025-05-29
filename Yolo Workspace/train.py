from ultralytics import YOLO
import datetime

DATASET_PATH = "dataset"

if __name__ == '__main__':
    for i in range(4):
        if i == 0:
            model = YOLO('yolo11n-seg.pt')
        elif i == 1:
            model = YOLO('runs/segment/train/weights/last.pt')
        else:
            model = YOLO(f'runs/segment/train{str(i)}/weights/last.pt')

        print(f"Training Epoch {i+1} started. Current Time: ", datetime.datetime.now())
        model.train(data=f"{DATASET_PATH}/data.yaml", epochs=50, imgsz=640, device='cuda', workers=12)
        print(f"Training Epoch {i+1} completed. Current Time: ", datetime.datetime.now())
    #model.train(data=f"{DATASET_PATH}/data.yaml", epochs=50, imgsz=640, device='cuda', workers=12)
    #print("50 Epoch Training completed. Current Time: ", datetime.datetime.now())

    """
    trained_model_path = model.ckpt
    print(trained_model_path)
    """