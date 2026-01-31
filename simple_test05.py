from datetime import datetime
import os
import cv2
import mediapipe as mp
import threading
import paramiko
import json
from ultralytics import YOLO
import time

# =======================
# ROI CONFIG
# =======================
ROI_FILE = "roi_boxes.json"
roi_points = []
drawing_done = False

# =======================
# INTRUSION CAPTURE
# =======================
INTRUSION_DIR = "intrusion_images"
os.makedirs(INTRUSION_DIR, exist_ok=True)

intrusion_active = False

# =======================
# RTSP ZERO BUFFER
# =======================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0"
)

RTSP_URL = "rtsp://admin:Techno%40321@192.168.1.2:554/Streaming/Channels/102"
last_frame_time = time.time()



# =======================
# RELAY CONFIG
# =======================
PI_IP = "192.168.1.3"
PI_USER = "invictus"
PI_PASSWORD = "2904"

RELAY_ON_SCRIPT = "/home/invictus/Desktop/relay_off.py"
RELAY_OFF_SCRIPT = "/home/invictus/Desktop/relay_on.py"

# =======================
# SSH CONNECT
# =======================
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(PI_IP, username=PI_USER, password=PI_PASSWORD)

relay_state = "ON"

def relay_on():
    global relay_state
    if relay_state != "ON":
        ssh.exec_command(f"python3 {RELAY_ON_SCRIPT}")
        relay_state = "ON"
        print("🔴 RELAY ON")

def relay_off():
    global relay_state
    if relay_state != "OFF":
        ssh.exec_command(f"python3 {RELAY_OFF_SCRIPT}")
        relay_state = "OFF"
        print("🟢 RELAY OFF")

# =======================
# MEDIAPIPE
# =======================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=0                      # 0-low accuracy/speed, 1-Balanced, 2-Most accurate/low speed
)

# =======================
# YOLO
# =======================
yolo_model = YOLO(r"runs/detect/train/weights/best.pt")  # gloves model

# =======================
# FRAME GRABBER THREAD
# =======================
# cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

latest_frame = None
running = True

def frame_grabber():
    global latest_frame, last_frame_time
    while running:
        cap.grab()
        ret, frame = cap.retrieve()
        if ret:
            latest_frame = frame
            last_frame_time = time.time()


threading.Thread(target=frame_grabber, daemon=True).start()
print("✅ Camera started")

# =======================
# ROI DRAWING
# =======================
def mouse_draw(event, x, y, flags, param):
    global roi_points, drawing_done

    if event == cv2.EVENT_LBUTTONDOWN and not drawing_done:
        roi_points.append([x, y])
        print(f"Point {len(roi_points)}: {x},{y}")

        if len(roi_points) == 8:
            data = {
                "red_box": roi_points[:4],
                "yellow_box": roi_points[4:]
            }
            with open(ROI_FILE, "w") as f:
                json.dump(data, f, indent=4)
            drawing_done = True
            print("✅ ROI saved")

def load_or_draw_roi(frame):
    global drawing_done

    if os.path.exists(ROI_FILE):
        with open(ROI_FILE, "r") as f:
            data = json.load(f)
        drawing_done = True
        print("📁 ROI loaded")
        return data["red_box"], data["yellow_box"]

    print("Draw RED (4) then YELLOW (4)")
    cv2.namedWindow("ROI SETUP")
    cv2.setMouseCallback("ROI SETUP", mouse_draw)

    while not drawing_done:
        temp = frame.copy()
        for i in range(len(roi_points)):
            cv2.circle(temp, tuple(roi_points[i]), 5, (255,0,0), -1)
            if i >= 1:
                cv2.line(temp, tuple(roi_points[i-1]),
                         tuple(roi_points[i]), (255,0,0), 2)
        cv2.imshow("ROI SETUP", temp)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyWindow("ROI SETUP")

    with open(ROI_FILE, "r") as f:
        data = json.load(f)

    return data["red_box"], data["yellow_box"]

def point_inside_box(point, box):
    x, y = point
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys)

def draw_box(frame, box, color):
    for i in range(4):
        cv2.line(frame, tuple(box[i]),
                 tuple(box[(i+1)%4]), color, 2)

# =======================
# ROI SETUP
# =======================
while latest_frame is None:
    pass

red_box, yellow_box = load_or_draw_roi(latest_frame)

# =======================
# MAIN LOOP
# =======================
frame_count = 0
YOLO_HOLD_TIME = 0.4
last_yolo_time = 0
yolo_glove_detected = False
yolo_glove_in_red = False

while True:
    if latest_frame is None:
        continue

    frame = latest_frame.copy()
    h, w, _ = frame.shape

    # ===== MediaPipe =====
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_in_red = False

    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks:
            for pt in lm.landmark:
                px = int(pt.x * w)
                py = int(pt.y * h)
                if point_inside_box((px, py), red_box):
                    hand_in_red = True
                    break
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            if hand_in_red:
                break

    # # ===== YOLO (every 3 frames) =====
    frame_count += 1

    # reset every frame
    yolo_glove_in_red = False

    if frame_count % 1 == 0:  # YOLO every frame

        yolo_result = yolo_model(frame, conf=0.6, verbose=False)[0]

        for box in yolo_result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 1:  # gloves only

                # glove detected → update time
                last_yolo_time = time.time()

                # check RED ROI boundary
                boundary_points = [
                    (x1, y1),
                    (x2, y1),
                    (x2, y2),
                    (x1, y2),
                ]

                for pt in boundary_points:
                    if point_inside_box(pt, red_box):
                        yolo_glove_in_red = True
                        break

                # draw always
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"GLOVES {conf:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

    # ===== YOLO AUTO OFF =====
    if time.time() - last_yolo_time > YOLO_HOLD_TIME:
        yolo_glove_detected = False

    # ===== RELAY =====
    if hand_in_red or yolo_glove_in_red:                   #RELAY
        relay_on()
    else:
        relay_off()

    # ===== INTRUSION IMAGE CAPTURE =====
    intrusion_detected =  hand_in_red or yolo_glove_in_red

    if intrusion_detected and not intrusion_active:
        intrusion_active = True

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"intrusion_{timestamp}.jpg"
        filepath = os.path.join(INTRUSION_DIR, filename)

        cv2.imwrite(filepath, frame)
        print(f"📸 Intrusion captured: {filepath}")

    elif not intrusion_detected:
        intrusion_active = False

    # ===== DRAW ROI =====
    draw_box(frame, red_box, (0,0,255))
    draw_box(frame, yellow_box, (0,255,255))

    cv2.imshow("HAND + YOLO GLOVE DETECTION", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =======================
# CLEANUP
# =======================q
running = False                                  # ⬅️
relay_off()
cap.release()
hands.close()
ssh.close()
cv2.destroyAllWindows()
