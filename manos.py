import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

def palma_centro(lista_coordenadas):
    coordenadas = np.array(lista_coordenadas)
    centro = np.mean(coordenadas, axis=0)  # Promedio de todas las coordenadas en x
    centro = int(centro[0]), int(centro[1])
    return centro

def cargar_imagenes():
    # Cargar las imágenes para los dedos levantados desde el directorio actual
    imagenes = []
    for i in range(1, 6):
        imagen = cv2.imread(f'imagen_{i}.png', cv2.IMREAD_UNCHANGED)
        if imagen is not None:
            imagenes.append(imagen)
        else:
            print(f"Advertencia: imagen_{i}.png no encontrada.")
    return imagenes

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Cargar imágenes
imagenes_dedos = cargar_imagenes()

# Puntos de los dedos
pulgar = [1, 2, 4]  # Puntos del pulgar
puntos_palma = [0, 1, 2, 5, 9, 13, 17]
punta_dedos = [8, 12, 16, 20]
base_dedos = [6, 10, 14, 18]

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,  # Número máximo de manos a detectar
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands, mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:

    while True:
        ret, frame = cap.read()  # Leer los fotogramas detectados por la cámara
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # De BGR a RGB
        
        # Detección de la cara
        results_face = face_detection.process(frame_rgb)
        
        if results_face.detections:
            for detection in results_face.detections:
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * width)
                y_min = int(bboxC.ymin * height)
                x_max = int((bboxC.xmin + bboxC.width) * width)
                y_max = int((bboxC.ymin + bboxC.height) * height)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # Calcular el centro del rostro
                rostro_centro = (x_min + x_max) // 2, (y_min + y_max) // 2
                break  # Solo usa la primera detección
        
        results = hands.process(frame_rgb)
        conteo_dedos = "_"  # Inicializar para que no haya error en la visualización del conteo

        if results.multi_hand_landmarks:
            coordenadas_pulgar = []
            coordenadas_palma = []
            coordenadas_punta = []
            coordenadas_base = []
            for hand_landmarks in results.multi_hand_landmarks:
                for index in pulgar:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordenadas_pulgar.append([x, y])

                for index in puntos_palma:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordenadas_palma.append([x, y])

                for index in punta_dedos:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordenadas_punta.append([x, y])

                for index in base_dedos:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordenadas_base.append([x, y])

                # Puntos del triángulo para el pulgar
                p1 = np.array(coordenadas_pulgar[0])
                p2 = np.array(coordenadas_pulgar[1])
                p3 = np.array(coordenadas_pulgar[2])

                # Distancia entre los puntos
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)

                # Cálculo del ángulo
                angulo = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                pulgar_levantado = angulo > 150

                # Índice, medio, anular y meñique
                nx, ny = palma_centro(coordenadas_palma)
                cv2.circle(frame, (nx, ny), 3, (0, 255, 3), 2)
                coordenadas_centro = np.array([nx, ny])
                coordenadas_punta = np.array(coordenadas_punta)
                coordenadas_base = np.array(coordenadas_base)

                # Distancias de la punta al centro y de la base al centro
                dist_centro_punta = np.linalg.norm(coordenadas_centro - coordenadas_punta, axis=1)  # El axis sirve para obtener las distancias de los 4 dedos
                dist_centro_base = np.linalg.norm(coordenadas_centro - coordenadas_base, axis=1)
                diferencia = dist_centro_punta - dist_centro_base
                dedos_levantados = diferencia > 0  # Si el dedo está levantado devuelve True
                dedos_levantados = np.append(pulgar_levantado, dedos_levantados)  # Unir la información del pulgar

                # Conteo de dedos extendidos
                conteo_dedos = str(np.count_nonzero(dedos_levantados))

                # Superponer imagen correspondiente
                if conteo_dedos.isdigit():
                    conteo_dedos = int(conteo_dedos)
                    if 1 <= conteo_dedos <= 5:
                        imagen_actual = imagenes_dedos[conteo_dedos - 1]
                        imagen_alta = cv2.resize(imagen_actual, (150, 150))  # Aumento del tamaño de la imagen
                        # Colocar la imagen cerca del rostro
                        x_offset = rostro_centro[0] - imagen_alta.shape[1] // 2
                        y_offset = rostro_centro[1] - imagen_alta.shape[0] // 2
                        y1, y2 = max(y_offset, 0), min(y_offset + imagen_alta.shape[0], height)
                        x1, x2 = max(x_offset, 0), min(x_offset + imagen_alta.shape[1], width)

                        # Overlay image on frame
                        alpha_s = imagen_alta[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s

                        for c in range(0, 3):
                            frame[y1:y2, x1:x2, c] = (alpha_s * imagen_alta[:, :, c] +
                                                      alpha_l * frame[y1:y2, x1:x2, c])

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Visualizar en pantalla
        cv2.rectangle(frame, (0, 0), (100, 100), (0, 0, 0), -1)
        cv2.putText(frame, str(conteo_dedos), (20, 75), 1, 6, (255, 255, 255), 2)  # Asegúrate de convertir a string
        print(f"Dedos extendidos: {conteo_dedos}")
        cv2.putText(frame, "DEDOS", (10, 95), 1, 1.5, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Salir con la tecla ESC
            break

cap.release()
cv2.destroyAllWindows()
