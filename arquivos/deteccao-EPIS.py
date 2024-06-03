import winsound
import cv2
from ultralytics import YOLO

modelo = YOLO('arquivos/best_1000.pt')

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#video = cv2.VideoCapture('arquivos/videos/epi-2.mp4')
#img = cv2.imread('arquivos/imagens/epi-03.webp')

while True:
    
    check, img = video.read()
    resultado = modelo.predict(img,verbose = False)

    for obj in resultado:
        nomes = obj.names
        for item in obj.boxes:
            conf = round(float(item.conf[0]),2)
            if conf > 0.5: 
                x1,y1,x2,y2 = item.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cls = int(item.cls[0])
                nomeClasse = nomes[cls]
                texto = f'{nomeClasse} - {conf}'
                cv2.putText(img,texto,(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,0),2)
                if nomeClasse == 'Capacete' or nomeClasse == 'Luva' or nomeClasse == 'Colete':
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0), 4)
                elif nomeClasse == 'Sem capacete' or nomeClasse == 'Sem luva' or nomeClasse== 'Sem colete':
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    winsound.Beep(700, 50)
                elif nomeClasse == 'Pessoa':
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 220, 0), 2)

    cv2.imshow('IMG',img)
    if cv2.waitKey(1) == 27:
        break
    