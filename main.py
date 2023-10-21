import cv2
import mediapipe as np
import math
import time
import threading
from playsound import playsound
import os
import serial

# Inicializar la cámara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Variables
parpadeo = False
conteo = 0
tiempo = 0
inicio = 0
final = 0
conteo_sue = 0
muestra = 0
counting = False
connected = True

# Inicializar el puerto serial
use_serial = False
if use_serial:
    ser = serial.Serial('COM6', 9600)

# Inicializar mediapipe para la detección facial
mpDibujo = np.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)
mpMallaFacial = np.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)


def throw_alert():
    global conteo_sue

    message = ""
    if conteo_sue == 1:
        playsound(os.path.join(os.path.dirname(__file__), 'audio.mp3'), True)
        message = "alert-1"
    elif conteo_sue == 2:
        playsound(os.path.join(os.path.dirname(__file__), 'audio2.mp3'), True)
        message = "alert-2"
    elif conteo_sue == 3:
        playsound(os.path.join(os.path.dirname(__file__), 'audio3.mp3'), True)
        message = "stop"

    if use_serial:
        ser.write(message.encode())


def detector_thread():
    global conteo_sue
    global counting
    global connected

    while True:
        if use_serial:
            data = ser.readline()
            print(data.decode('utf-8').strip())

        actual_time = 0
        while counting:
            time.sleep(1)
            actual_time += 1
            print("Contando: ", actual_time)

            if actual_time == 3:
                conteo_sue += 1

                throw_alert()
                actual_time = 0

            """
            if actual_time > 5:
                throw_alert()
            elif conteo_sue >= 3:
                throw_alert()
                conteo_sue = 0
            """

        time.sleep(0.05)

        if not connected:
            break


threading.Thread(target=detector_thread).start()

while True:
    ret, frame = cap.read()

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = MallaFacial.process(frameRGB)

    px = []
    py = []
    lista = []
    r = 5
    t = 3

    if resultados.multi_face_landmarks:
        for rostro in resultados.multi_face_landmarks:
            mpDibujo.draw_landmarks(frame, rostro, mpMallaFacial.FACEMESH_FACE_OVAL, ConfDibu, ConfDibu)

            for id, puntos in enumerate(rostro.landmark):
                al, an, c = frame.shape
                cx, cy = int(puntos.x * an), int(puntos.y * al)
                px.append(cx)
                py.append(cy)
                lista.append([id, cx, cy])

                if len(lista) == 468:
                    # Ojo Derecho
                    x1, y1 = lista[145][1:]
                    x2, y2 = lista[159][1:]
                    hx, hy = (x1 + x2) // 2, (y1 + y2) // 2
                    longitud1 = math.hypot(x2 - x1, y2 - y1)

                    # Ojo Izquierdo
                    x3, y3 = lista[374][1:]
                    x4, y4 = lista[386][1:]
                    hx2, hy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    longitud2 = math.hypot(x4 - x3, y4 - y3)

                    # Conteo de parpadeo
                    cv2.putText(frame, f'Parpadeos: {int(conteo)}', (300, 60), cv2.FONT_HERSHEY_PLAIN, 3,
                                (255, 255, 255), 3)
                    cv2.putText(frame, f'MS: {int(conteo_sue)}', (720, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255),
                                3)

                    if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                        conteo += 1
                        # parpadeo = True
                        counting = True
                        # inicio = time.time()

                    elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:
                        # parpadeo = False
                        counting = False
                        # final = time.time()

                    # Temporizador
                    # tiempo = round(final - inicio, 0)
                    """
                    if tiempo >= 3:
                        # TODO: EN ESTE LUGAR SE DETECTA EL MICROSUEÑO
                        conteo_sue += 1
                        muestra = tiempo
                        inicio = 0
                        final = 0
                    """
    else:
        parpadeo = False
        counting = False

    # Mostramos el frame
    cv2.imshow('Detector de Microsueños', frame)

    # Agregamos una manera de salir del bucle - presionando la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos los recursos y cerramos las ventanas
cap.release()
cv2.destroyAllWindows()
connected = False
