from collections import Counter

import cv2
import mediapipe as np
import math
import time

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

parpadeo = False
conteo = 0
tiempo = 0
inicio = 0
final = 0
conteo_sue = 0
muestra = 0

mpDibujo = np.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

mpMallaFacial = np.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

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
                    cv2.putText(frame, f'Parpadeos: {int(conteo)}', (300, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    cv2.putText(frame, f'Micro Sueños: {int(conteo_sue)}', (720, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

                    if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                        conteo += 1
                        parpadeo = True
                        inicio = time.time()

                    elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:
                        parpadeo = False
                        final = time.time()

                    # Temporizador
                    tiempo = round(final - inicio, 0)

                    if tiempo >= 3:
                        conteo_sue += 1
                        muestra = tiempo
                        inicio = 0
                        final = 0
    # Mostramos el frame
    cv2.imshow('Frame', frame)

    # Agregamos una manera de salir del bucle - presionando la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos los recursos y cerramos las ventanas
cap.release()
cv2.destroyAllWindows()