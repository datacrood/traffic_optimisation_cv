import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('VehTr.MOV')

my_file = open("coco.txt", "r")
data = my_file.read()

class_list = data.split("\n") 

count = 0

tracker = Tracker()

cy1 = 322
cy2 = 368
offset = 6

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 8 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    bbox_count = 0
    list = []
             
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
            bbox_count += 1
    
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    
    cv2.putText(frame, f'No. of Cars: {bbox_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2)
    cv2.imshow("Vehicle Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()