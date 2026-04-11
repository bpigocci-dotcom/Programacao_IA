#import das bibliotecas
import cv2 
from ultralytics import YOLO 

#carregamento do modelo
print("carregando modelo...")
model = YOLO('yolov8n.pt') #versao leve

# 2 abrir uma conexão com webcam
cap = cv2.VideoCapture(0)
# numero 0 representa uma versao integrada com o computador
# numero 1 representa uma versao de web cam via usb
# caso seja remota o ip deve ser informado

#verificar se a camera abriu corretamente
if not cap.isOpened():
    print("Erro ao acessar câmera")
    exit()

print("Iniciando detectação. Pressione 'q' para sair")

#3- Iniciar a leitura das detectações
while True:
    sucesso, frame = cap.read()

    if sucesso:
        results = model(frame, conf=0.5)
        annotated_frame = results[0].plot()
        cv2.imshow("Visão Computacional - YOLOv8", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

#limpeza
cap.release()
cv2.destroyAllWindows()