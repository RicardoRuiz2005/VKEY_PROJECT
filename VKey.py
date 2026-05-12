import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

fps_limit = 30
frame_time = 1 / fps_limit

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320) # Calidad sobre cantidad, para reducir lag
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


puntos_calibracion = []
nombres = ["SUP-IZQ", "SUP-DER", "INF-DER", "INF-IZQ"]
bbox = None  # (x_min, y_min, x_max, y_max)

# Estado gestos
last_win_tab_time = 0
pointing_start_time = None
prev_cursor = None

dragging = False
click_hold_time = 0.3

# smoothing
smoothening = 2
plocX, plocY = 0, 0
clocX, clocY = 0, 0


print("CALIBRACION: pon tu indice en cada esquina y presiona ESPACIO")

def gesto_mano_abierta(lm):
    return (
        lm[8].y < lm[6].y - 0.04 and
        lm[12].y < lm[10].y - 0.04 and
        lm[16].y < lm[14].y - 0.04 and
        lm[20].y < lm[18].y - 0.04
    )

def gesto_indice_solito(lm):
    return (
        lm[8].y < lm[6].y and
        lm[12].y > lm[10].y and
        lm[16].y > lm[14].y and
        lm[20].y > lm[18].y
    )

def pinch(lm):

    thumb_tip = lm[4]
    index_tip = lm[8]

    distance = math.hypot(
        thumb_tip.x - index_tip.x,
        thumb_tip.y - index_tip.y
    )

    # tamaño de mano
    hand_size = math.hypot(
        lm[5].x - lm[17].x,
        lm[5].y - lm[17].y
    )

    return distance < hand_size * 0.35


while True:
    ok, frame = cap.read()
    if not ok:
        break


    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    current_time = time.time()


    tip_xy = None
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        tip = hand.landmark[8]
        tip_xy = (int(tip.x * w), int(tip.y * h))


    # Instrucciones
    if len(puntos_calibracion) < 4:
        texto = f"Toca esquina {nombres[len(puntos_calibracion)]} y presiona ESPACIO"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "CALIBRADO", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # Puntos ya guardados
    for p in puntos_calibracion:
        cv2.circle(frame, p, 8, (255, 0, 0), -1)


    # Rectangulo bounding box
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


    # Mover cursor SOLO si el dedo esta dentro del bbox
    if bbox is not None and tip_xy is not None:
        x1, y1, x2, y2 = bbox
        dentro = x1 <= tip_xy[0] <= x2 and y1 <= tip_xy[1] <= y2


        if dentro:
            x_s = (tip_xy[0] - x1) / (x2 - x1) * screen_w
            y_s = (tip_xy[1] - y1) / (y2 - y1) * screen_h

            # smoothing
            clocX = plocX + (x_s - plocX) / smoothening
            clocY = plocY + (y_s - plocY) / smoothening

            pyautogui.moveTo(int(clocX), int(clocY))
            
            # WIN + TAB
            if gesto_mano_abierta(lm):
                if current_time - last_win_tab_time > 2:
                    pyautogui.hotkey('win', 'tab')
                    last_win_tab_time = current_time
                    
            # DRAG
            if gesto_indice_solito(lm) and pinch(lm):
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True

            # CLICK
            elif pinch(lm):
                pyautogui.click()
                time.sleep(0.1)
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            plocX, plocY = clocX, clocY
            cv2.circle(frame, tip_xy, 15, (0, 255, 0), 3)
        else:
            cv2.circle(frame, tip_xy, 15, (0, 0, 255), 3)
    elif tip_xy is not None:
        # Sin calibrar todavia: solo marca el dedo
        cv2.circle(frame, tip_xy, 10, (0, 255, 0), -1)


    cv2.imshow("Paso 2 - BBox", frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' ') and tip_xy is not None and len(puntos_calibracion) < 4:
        puntos_calibracion.append(tip_xy)
        print(f"Esquina {len(puntos_calibracion)}: {tip_xy}")


        if len(puntos_calibracion) == 4:
            xs = [p[0] for p in puntos_calibracion]
            ys = [p[1] for p in puntos_calibracion]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            print(f"BBox: {bbox}")
    if key == ord('r'):
        puntos_calibracion = []
        bbox = None
        print("Reset")
    time.sleep(frame_time)

cap.release()
cv2.destroyAllWindows()
