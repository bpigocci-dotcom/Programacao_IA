#import das lib
import cv2
from deepface import DeepFace
import time

#passo 1
imagem_referencia= "face_id.jpg"
print("carregando identidade do morador")

#pré analise
try:
    DeepFace.represent(img_path= imagem_referencia, model_name="VGG-Face")
    print("identidade carregada com sucesso")
except:
    print("Erro! não encontrei o arquivo ou não há rosto nele :(")
    exit()

#Iniciar camera
cap = cv2.VideoCapture(0)
print("Sistema de portaria ativo")

while True:
    ret, frame = cap.read() #ret= retorno
    if not ret: break

    frame_small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

    #desenhar retangulo
    height, width, _ = frame.shape
    cv2.rectangle(frame, (100,100), (width-100, height-100), (255,0,0),2)
    #tamanho, cor 

    #verificação da imagem
    cv2.putText(frame, "Pressione V para verificar o acesso" , (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
    key = cv2.waitKey(1)

    if key & 0xFF == ord("V"):
        print("Verificando Identidade")
        try: 
            resultado = DeepFace.verify(
                img1_path = frame,
                img2_path = imagem_referencia,
                model_name="VGG-Face",
                enforce_detection = False
            )

            if resultado['verified']:
                print(">>>>ACESSO LIBERADO!>>>>")
            else:
                print(">>>>ACESSO NEGADO!>>>>")
                cv2.rectangle(frame, (0,0),(width, height), (0,0,255),2)
                cv2.imshow("Portaria", frame)
                cv2.waitKey(2000)

        except Exception as e:
            print(f"Erro na Leitura: {e}")

    cv2.imshow("Portaria", frame)

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()