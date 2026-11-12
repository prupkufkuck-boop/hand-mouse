# -*- coding: utf-8 -*-
"""
Hand Mouse — управление курсором через камеру.

Отслеживает руку через веб-камеру:
  * курсор привязывается к кончику указательного пальца, когда показан
    жест «палец вверх» (указательный вытянут, остальные сжаты);
  * когда рука пропадает из виду — привязка сбрасывается и курсор
    замирает, пока жест не покажут снова;
  * щипок (большой + указательный) — нажатие кнопки мыши;
  * короткий щипок = клик, удержание щипка с движением = перетаскивание;
  * ESC — выход.

Запуск: python hand_mouse.py
"""

import sys
import time

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ----------------------------- Настройки -----------------------------------

MODEL_PATH = "hand_landmarker.task"

CAM_WIDTH, CAM_HEIGHT = 640, 480   # разрешение захвата с камеры

# Сколько процентов кадра по краям игнорировать: работаем только в центре,
# а он растягивается на весь экран — так легче дотянуться до краёв экрана.
MARGIN_X = 0.15
MARGIN_Y = 0.15

SMOOTHING = 0.45       # 0 — курсор дёргается за пальцем, 1 — полностью заморожен
PINCH_ON = 0.38        # расстояние < этого порога — щипок начался
PINCH_OFF = 0.55       # расстояние > этого порога — щипок закончился (гистерезис)
CLICK_MIN_INTERVAL = 0.25   # защита от двойных кликов, сек

DRAW_LANDMARKS = True  # показывать точки руки в окне предпросмотра

# Индексы ключевых точек руки в MediaPipe HandLandmarker
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
WRIST = 0

# Суставы пальцев для проверки жеста «палец вверх»:
# (кончик, второй сустав) для среднего, безымянного и мизинца
OTHER_FINGERS = [(12, 10), (16, 14), (20, 18)]

# Сколько кадров подряд нужно держать открытую ладонь, чтобы отменить
# привязку (защита от случайных срабатываний при движении руки)
OPEN_PALM_FRAMES = 6

# --------------------------- Глобальное состояние --------------------------

pyautogui.FAILSAFE = False   # курсор управляется программно, «уголки-аварии» мешают
pyautogui.PAUSE = 0          # без задержки между вызовами

smooth_x = None              # сглаженная позиция курсора (экранные координаты)
smooth_y = None
pinching = False             # текущее состояние щипка
bound = False                # курсор привязан к пальцу (жест «палец вверх»)
palm_frames = 0              # сколько кадров подряд видна открытая ладонь
last_click_time = 0.0

screen_w, screen_h = pyautogui.size()

# --------------------------- Инициализация модели --------------------------

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Ладонь и палец-линия для отрисовки
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # большой палец
    (0, 5), (5, 6), (6, 7), (7, 8),          # указательный
    (5, 9), (9, 10), (10, 11), (11, 12),     # средний
    (9, 13), (13, 14), (14, 15), (15, 16),   # безымянный
    (13, 17), (17, 18), (18, 19), (19, 20),  # мизинец
    (0, 17),
]


def pinch_distance(landmarks) -> float:
    """Нормированное расстояние между кончиками большого и указательного пальцев.

    Делим на размер ладони (запястье — основание указательного), чтобы
    расстояние не зависело от того, насколько близко рука к камере.
    """
    thumb = landmarks[THUMB_TIP]
    index = landmarks[INDEX_TIP]
    wrist = landmarks[WRIST]
    mcp = landmarks[INDEX_MCP]
    hand_size = np.hypot(wrist.x - mcp.x, wrist.y - mcp.y)
    if hand_size < 1e-6:
        return 1.0
    d = np.hypot(thumb.x - index.x, thumb.y - index.y)
    return d / hand_size


def is_pointing(landmarks) -> bool:
    """Жест «палец вверх»: указательный вытянут, остальные пальцы сжаты.

    Сравниваем расстояние от запястья до кончика с расстоянием от запястья
    до второго сустава — работает при любом повороте руки.
    """
    wrist = landmarks[WRIST]

    def dist(i, j):
        return np.hypot(landmarks[i].x - wrist.x, landmarks[i].y - wrist.y)

    # Указательный вытянут (кончик заметно дальше от запястья, чем сустав)
    index_extended = dist(INDEX_TIP, 0) > dist(6, 0) * 1.15
    # Остальные три пальца сжаты
    others_folded = all(
        dist(tip, 0) < dist(pip, 0) for tip, pip in OTHER_FINGERS
    )
    return index_extended and others_folded


def is_open_palm(landmarks) -> bool:
    """Жест «открытая ладонь»: все 5 пальцев прямые — отмена привязки."""
    wrist = landmarks[WRIST]

    def dist(i):
        return np.hypot(landmarks[i].x - wrist.x, landmarks[i].y - wrist.y)

    # Указательный, средний, безымянный, мизинец — прямые
    fingers_extended = dist(INDEX_TIP) > dist(6) and all(
        dist(tip) > dist(pip) for tip, pip in OTHER_FINGERS
    )
    # Большой палец прямой: кончик дальше от мизинца, чем сустав IP
    thumb_extended = dist(THUMB_TIP) > dist(3)
    return fingers_extended and thumb_extended


def update_cursor(landmarks):
    """Переводит точку управления в экранные координаты и двигает курсор.

    Управляем основанием указательного пальца (пястный сустав), а не кончиком:
    при щипке кончик смещается к большому пальцу и курсор «прыгал бы»,
    а сустав стоит на месте. Перетаскивание при зажатом щипке сохраняется.
    """
    global smooth_x, smooth_y

    tip = landmarks[INDEX_MCP]

    # Зеркалим по X (самообзор — камера смотрит «как в зеркало»),
    # затем растягиваем центральную область кадра на весь экран.
    nx = 1.0 - tip.x
    ny = tip.y

    lo_x, hi_x = MARGIN_X, 1.0 - MARGIN_X
    lo_y, hi_y = MARGIN_Y, 1.0 - MARGIN_Y
    fx = (nx - lo_x) / (hi_x - lo_x)
    fy = (ny - lo_y) / (hi_y - lo_y)
    fx = min(1.0, max(0.0, fx))
    fy = min(1.0, max(0.0, fy))

    target_x = fx * (screen_w - 1)
    target_y = fy * (screen_h - 1)

    if smooth_x is None:
        smooth_x, smooth_y = target_x, target_y
    else:
        smooth_x += (target_x - smooth_x) * SMOOTHING
        smooth_y += (target_y - smooth_y) * SMOOTHING

    pyautogui.moveTo(int(smooth_x), int(smooth_y))


def update_pinch(distance: float):
    """Отслеживает щипок: нажатие при сжатии, отпускание при разжатии."""
    global pinching, last_click_time

    if not pinching and distance < PINCH_ON:
        pinching = True
        pyautogui.mouseDown()
    elif pinching and distance > PINCH_OFF:
        pinching = False
        pyautogui.mouseUp()


def unbind():
    """Сброс привязки: рука пропала или жест не показан — курсор замирает."""
    global bound, pinching, smooth_x, smooth_y
    if bound:
        bound = False
        smooth_x, smooth_y = None, None
    if pinching:
        pyautogui.mouseUp()
        pinching = False


def draw_overlay(frame, landmarks, distance, fps):
    """Рисует скелет руки, щипок и статус в окне предпросмотра."""
    h, w = frame.shape[:2]

    if landmarks is not None:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        if DRAW_LANDMARKS:
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (200, 220, 200), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
        # Линия щипка: зелёная — зажат, жёлтая — рядом
        color = (0, 255, 0) if pinching else ((0, 220, 255) if distance < PINCH_OFF else (160, 160, 160))
        cv2.line(frame, pts[THUMB_TIP], pts[INDEX_TIP], color, 3, cv2.LINE_AA)
        cv2.circle(frame, pts[INDEX_TIP], 8 if pinching else 5, color, 2, cv2.LINE_AA)

    state = "PINCH" if pinching else ("BOUND" if bound else "FREE")
    color = (0, 255, 0) if pinching else ((255, 255, 255) if bound else (0, 165, 255))
    cv2.putText(frame, f"{state}  d={distance:.2f}  {fps:.0f} fps",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0) if pinching else color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Point up = bind   Open palm = unbind   Pinch = click   ESC = exit",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    global smooth_x, smooth_y, pinching, bound, palm_frames

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Не удалось открыть камеру. Проверьте, не занята ли она другой программой.")
        sys.exit(1)

    print("Hand Mouse запущена. Покажите указательный палец вверх — курсор привяжется.")
    print("Щипок = клик, ESC — выход.")
    print("Чтобы быстро остановить: закройте окно предпросмотра или нажмите Ctrl+C в консоли.")

    prev_time = time.perf_counter()
    fps = 0.0
    last_timestamp = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Монотонный таймстамп — требование VIDEO-режима MediaPipe
            ts = int(time.perf_counter() * 1000)
            if ts <= last_timestamp:
                ts = last_timestamp + 1
            last_timestamp = ts

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, ts)

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                pointing = is_pointing(landmarks)

                # Открытая ладонь (5 прямых пальцев), продержавшаяся
                # OPEN_PALM_FRAMES кадров подряд, отменяет привязку.
                if is_open_palm(landmarks):
                    palm_frames += 1
                else:
                    palm_frames = 0

                if bound and palm_frames >= OPEN_PALM_FRAMES:
                    unbind()
                    distance = 1.0
                elif bound or (pointing and palm_frames == 0):
                    # Привязка включается жестом «палец вверх» и держится, пока
                    # рука в кадре; сбрасывается ладонью или пропажей руки.
                    if not bound:
                        bound = True
                        smooth_x, smooth_y = None, None  # привязка заново, без рывка
                    distance = pinch_distance(landmarks)
                    update_cursor(landmarks)
                    update_pinch(distance)
                else:
                    distance = 1.0  # рука видна, но жест ещё не показан
            else:
                unbind()
                distance = 1.0

            draw_overlay(frame, result.hand_landmarks[0] if result.hand_landmarks else None,
                         distance, fps)
            cv2.imshow("Hand Mouse", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        if pinching:
            pyautogui.mouseUp()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("Hand Mouse остановлена.")


if __name__ == "__main__":
    main()
