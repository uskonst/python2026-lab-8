"""
Лабораторная работа №8, Задания 2 и 3.
Задание 2: Отслеживание ArUco-метки через камеру.
Задание 3: Проверка попадания метки в центральный квадрат 200x200 пикселей.

Алгоритм работы:
1. Захватываем кадр с камеры
2. Конвертируем в оттенки серого (для детектора)
3. Ищем ArUco-метки на кадре
4. Рисуем метки, их ID и центр
5. Проверяем, попал ли центр метки в зону 200x200 по центру экрана
6. Окрашиваем зону в зелёный (попал) или красный (не попал)
"""

import cv2
import cv2.aruco as aruco
import numpy as np


# --- Константы ---
CAMERA_INDEX = 0        # 0 — встроенная камера, 1 — внешняя USB-камера
TARGET_MARKER_ID = 0    # Отслеживаем только метку с этим ID
ZONE_SIZE = 200         # Размер зоны попадания в пикселях (200x200)

# Цвета в формате BGR (Blue, Green, Red) — так работает OpenCV
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

# Словарь ArUco — должен совпадать с тем, что использовался при генерации
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMS = aruco.DetectorParameters()


def get_frame_center(frame: np.ndarray) -> tuple[int, int]:
    """
    Вычисляет координаты центра кадра.

    :param frame: Текущий кадр с камеры.
    :return: Кортеж (cx, cy) — центр кадра.
    """
    h, w = frame.shape[:2]
    return w // 2, h // 2


def draw_target_zone(
    frame: np.ndarray,
    center: tuple[int, int],
    zone_size: int,
    is_inside: bool
) -> tuple[int, int, int, int]:
    """
    Рисует квадратную зону попадания по центру кадра.
    Зелёная — метка внутри, красная — снаружи.

    :param frame: Кадр для рисования.
    :param center: Центр кадра (cx, cy).
    :param zone_size: Размер стороны квадрата в пикселях.
    :param is_inside: True если метка внутри зоны.
    :return: Координаты зоны (x1, y1, x2, y2).
    """
    cx, cy = center
    half = zone_size // 2

    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    color = COLOR_GREEN if is_inside else COLOR_RED

    # Полупрозрачный фон зоны
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # -1 = заливка
    # addWeighted смешивает два изображения: frame и overlay с прозрачностью
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Граница зоны (жирная рамка)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Подпись зоны
    label = "МЕТКА В ЗОНЕ!" if is_inside else "ЗОНА ПОПАДАНИЯ"
    cv2.putText(
        frame, label, (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )

    return x1, y1, x2, y2


def is_marker_in_zone(
    marker_center: tuple[int, int],
    zone: tuple[int, int, int, int]
) -> bool:
    """
    Проверяет, находится ли центр метки внутри зоны.

    :param marker_center: Координаты центра метки (mx, my).
    :param zone: Зона (x1, y1, x2, y2).
    :return: True если центр метки внутри зоны.
    """
    mx, my = marker_center
    x1, y1, x2, y2 = zone

    # Простая проверка: пиксель внутри прямоугольника
    return x1 <= mx <= x2 and y1 <= my <= y2


def detect_and_draw_markers(
    frame: np.ndarray,
    zone: tuple[int, int, int, int]
) -> bool:
    """
    Обнаруживает ArUco-метки на кадре, рисует их и проверяет попадание в зону.

    :param frame: Текущий кадр с камеры.
    :param zone: Координаты зоны попадания (x1, y1, x2, y2).
    :return: True если нужная метка найдена и попала в зону.
    """
    # Детектор работает лучше на сером изображении
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectMarkers возвращает:
    # corners — список углов каждой найденной метки (4 угла = 4 точки)
    # ids     — список ID найденных меток
    # _       — отклонённые кандидаты (нам не нужны)
    detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    corners, ids, _ = detector.detectMarkers(gray)

    # Если метки не найдены — ids будет None
    if ids is None:
        return False

    # Обходим все найденные метки
    for i, marker_id in enumerate(ids.flatten()):
        # Нас интересует только метка с нужным ID
        if marker_id != TARGET_MARKER_ID:
            continue

        # corners[i][0] — массив 4 точек углов одной метки
        # shape: (4, 2), то есть 4 пары (x, y)
        marker_corners = corners[i][0]

        # Центр метки = среднее арифметическое координат 4 углов
        mx = int(np.mean(marker_corners[:, 0]))
        my = int(np.mean(marker_corners[:, 1]))
        marker_center = (mx, my)

        # Рисуем контур метки (замкнутый полигон по 4 углам)
        pts = marker_corners.astype(int).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=COLOR_YELLOW, thickness=3)

        # Рисуем центр метки — залитый круг
        cv2.circle(frame, marker_center, 6, COLOR_YELLOW, -1)

        # Подпись с ID метки рядом с первым углом
        corner_pt = tuple(marker_corners[0].astype(int))
        cv2.putText(
            frame, f"ID: {marker_id}", corner_pt,
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2
        )

        # Проверяем попадание центра в зону
        return is_marker_in_zone(marker_center, zone)

    return False


def run_tracking() -> None:
    """
    Основной цикл: захват кадров с камеры и отслеживание метки.
    Нажмите Q для выхода.
    """
    # Открываем камеру
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"  ❌ Не удалось открыть камеру (индекс {CAMERA_INDEX}).")
        print("  Попробуйте изменить CAMERA_INDEX на 1 или 2.")
        return

    print("  ✅ Камера открыта. Поднесите метку к камере.")
    print("  Нажмите Q для выхода.")

    while True:
        # cap.read() возвращает: (успех: bool, кадр: numpy array)
        ret, frame = cap.read()

        if not ret:
            print("  ⚠️ Не удалось получить кадр.")
            break

        frame_center = get_frame_center(frame)

        # Сначала проверяем метку (без зоны — зону нарисуем позже)
        # Временно запускаем детектор чтобы узнать, внутри ли метка
        # (is_inside нужен ДО рисования зоны, чтобы выбрать цвет)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
        corners, ids, _ = detector.detectMarkers(gray)

        # Предварительно считаем центр метки для проверки попадания
        is_inside = False
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == TARGET_MARKER_ID:
                    mc = corners[i][0]
                    mx = int(np.mean(mc[:, 0]))
                    my = int(np.mean(mc[:, 1]))
                    half = ZONE_SIZE // 2
                    cx, cy = frame_center
                    zone_check = (cx - half, cy - half, cx + half, cy + half)
                    is_inside = is_marker_in_zone((mx, my), zone_check)
                    break

        # Рисуем зону с нужным цветом (зелёный/красный)
        zone = draw_target_zone(frame, frame_center, ZONE_SIZE, is_inside)

        # Рисуем метки поверх зоны
        detect_and_draw_markers(frame, zone)

        # Подсказка в углу
        cv2.putText(
            frame, "Q - выход", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2
        )

        cv2.imshow("ArUco Tracker — Лаб. 8", frame)

        # waitKey(1) — ждём 1 мс; если нажата Q (код 113 или 81) — выход
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    print("  Камера закрыта. Программа завершена.")


def main() -> None:
    """Точка входа для заданий 2 и 3."""
    print("=" * 48)
    print("  Задания 2 и 3 — ArUco-трекер с зоной")
    print("=" * 48)
    run_tracking()


if __name__ == "__main__":
    main()