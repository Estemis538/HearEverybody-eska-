import base64
import os
import time
import threading
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request, send_from_directory


app = Flask(__name__, static_folder="static", static_url_path="/static")

# MediaPipe Hands: извлекаем 21 ориентир (landmarks) из изображения руки.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # noqa: F401 (оставлено для удобства отладки)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# MediaPipe и наша история предсказаний не потокобезопасны.
process_lock = threading.Lock()


# 20 “базовых” слов повседневной речи (демо): мы сопоставляем их 20 статическим позам
# (наборы “какие пальцы раскрыты”). Точная точность для реального ASL требует обучения,
# но пайплайн “камера -> Python -> распознавание -> слово в UI” работает полностью.
WORDS_20 = [
    "hello",
    "yes",
    "no",
    "please",
    "thank you",
    "sorry",
    "love",
    "friend",
    "good",
    "bad",
    "help",
    "water",
    "food",
    "more",
    "stop",
    "go",
    "where",
    "when",
    "name",
    "work",
]


TEMPLATES = [
    # thumb, index, middle, ring, pinky
    ((1, 0, 0, 0, 0), "hello"),
    ((0, 1, 0, 0, 1), "yes"),
    ((0, 1, 1, 0, 0), "no"),
    ((1, 1, 0, 0, 0), "please"),
    ((0, 1, 1, 1, 0), "thank you"),
    ((1, 0, 1, 0, 1), "sorry"),
    ((0, 0, 1, 1, 1), "love"),
    ((1, 1, 1, 0, 0), "friend"),
    ((0, 0, 0, 1, 0), "good"),
    ((1, 0, 0, 1, 0), "bad"),
    ((0, 1, 0, 1, 1), "help"),
    ((0, 1, 1, 0, 1), "water"),
    ((1, 0, 1, 1, 0), "food"),
    ((1, 1, 0, 1, 0), "more"),
    ((0, 0, 1, 0, 1), "stop"),
    ((1, 0, 0, 1, 1), "go"),
    ((0, 1, 0, 0, 0), "where"),
    ((0, 0, 1, 0, 0), "when"),
    ((1, 1, 0, 0, 1), "name"),
    ((1, 0, 1, 0, 0), "work"),
]

TEMPLATE_MASKS = []
for bits, word in TEMPLATES[:20]:
    mask = 0
    for i, b in enumerate(bits):
        mask |= (1 if b else 0) << i
    TEMPLATE_MASKS.append((mask, bits, word))


def _bits_from_landmarks(landmarks, handedness_label: str):
    """
    Возвращает (thumb, index, middle, ring, pinky) как биты 0/1.
    thumb определяется по оси X (зависит от Left/Right),
    остальные пальцы — по сравнению tip.y и pip.y.
    """
    lm = landmarks

    # Имена landmarks: thumb tip=4, ip=3
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    if handedness_label == "Right":
        thumb_open = 1 if thumb_tip.x > thumb_ip.x else 0
    else:
        thumb_open = 1 if thumb_tip.x < thumb_ip.x else 0

    def finger_open(tip_idx: int, pip_idx: int) -> int:
        # Для пальцев tip.y “выше” (меньше) pip.y когда палец вытянут
        return 1 if lm[tip_idx].y < lm[pip_idx].y else 0

    index_open = finger_open(8, 6)
    middle_open = finger_open(12, 10)
    ring_open = finger_open(16, 14)
    pinky_open = finger_open(20, 18)
    return (thumb_open, index_open, middle_open, ring_open, pinky_open)


def _predict_word(bits):
    pred_mask = 0
    for i, b in enumerate(bits):
        pred_mask |= (1 if b else 0) << i

    # “Ближайший шаблон” по расстоянию Хэмминга.
    best_word = ""
    best_conf = 0.0
    best_dist = 10

    pred_list = list(bits)
    for tmpl_mask, tmpl_bits, word in TEMPLATE_MASKS:
        dist = sum(1 for i in range(5) if pred_list[i] != tmpl_bits[i])
        if dist < best_dist:
            best_dist = dist
            best_word = word

    # Конфиденс: выше, когда поза ближе к шаблону.
    best_conf = max(0.0, 1.0 - (best_dist / 5.0))
    return best_word, best_conf


# Сглаживание “по времени”: выбираем самое частое слово за короткое окно.
history = deque(maxlen=6)  # items: (word, conf)
last_emit_word = ""
last_emit_at = 0.0


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/recognize")
def api_recognize():
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get("frame", "")
    if not frame_b64:
        return jsonify({"word": "", "confidence": 0.0, "error": "missing_frame"})

    # Разрешаем передачу как “data:image/jpeg;base64,....” или чистую base64-строку.
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",", 1)[1]

    try:
        jpg_bytes = base64.b64decode(frame_b64, validate=True)
    except Exception:
        return jsonify({"word": "", "confidence": 0.0, "error": "invalid_base64"})

    # JPEG → BGR image → обработка MediaPipe.
    img_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"word": "", "confidence": 0.0, "error": "decode_failed"})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with process_lock:
        results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        history.clear()
        return jsonify({"word": "", "confidence": 0.0})

    hand_landmarks = results.multi_hand_landmarks[0]
    handedness_label = "Right"
    if results.multi_handedness:
        handedness_label = results.multi_handedness[0].classification[0].label

    bits = _bits_from_landmarks(hand_landmarks.landmark, handedness_label)
    raw_word, raw_conf = _predict_word(bits)

    with process_lock:
        if raw_word and raw_conf >= 0.35:
            history.append((raw_word, raw_conf))
        else:
            history.append(("", 0.0))

        candidates = [w for (w, _c) in history if w]
        if not candidates:
            last_word = ""
            out_conf = 0.0
        else:
            last_word = Counter(candidates).most_common(1)[0][0]
            out_conf = max((c for (w, c) in history if w == last_word), default=0.0)

    now = time.time()
    global last_emit_word, last_emit_at
    if last_word != last_emit_word or (now - last_emit_at) > 0.2:
        last_emit_word = last_word
        last_emit_at = now

    return jsonify({"word": last_word, "confidence": float(out_conf)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)

