import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# ========== 配置部分 ==========

# ---
YOLO_MODEL_PATH = r"C:\Users\lucky\Desktop\shopping26\yolov8n.pt"
ROI_SCALE = 0.45
CONF_THRESHOLD = 0.84

# ---
OCR_CONFIG = \
{
    # 'use_angle_cls': True,
    'lang': 'ch',
    # 'show_log': False
}
FONT_PATH = "simhei.ttf"
FONT_SIZE = 20

# ---
SERIAL_PORT = 'COM4'
BAUDRATE = 9600

# ---
CAMERA_TYPE = 0

# ---
INSTRUCTION_MAP_TMP = \
{
    # 2: 'c',  # 加多宝
    3: 'd',  # 茶pi
    4: 'e',  # 东鹏
    5: 'b',  # 锐澳
}

INSTRUCTION_MAP_GET = \
{
    # 0: 'a',  # 旺仔牛奶
    6: 'a',  # 营养快线
    # 7: 'a',  # 养乐多
    8: 'a',  # 百事可乐
}

INSTRUCTION_MAP_OCR = \
{
    # 2: 'g',  # 加多宝
    3: 'h',  # 茶pi
    4: 'i',  # 东鹏
    5: 'f',  # 锐澳
}

OCR_KEYWORD_MAP = \
{
    "东鹏": 4,
    "锐澳": 5,
    "茶": 3,
    # "加": 2
}

# ========== 全局变量 ==========

global ser
global yolo_model
global ocr_model

current_mode = 'idle'

last_operation_time = 0
operation_intervals = \
{
    'tmp': 8,
    'get': 6,
    'ocr': 9.7
}
last_operable_status = False

sent_instructions = set()


# ========== 初始化函数 ==========
def init_models():
    global yolo_model, ocr_model
    print("正在加载YOLO模型...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("正在加载OCR模型...")
    ocr_model = PaddleOCR(**OCR_CONFIG)
    print("所有模型加载完成")


def init_serial():
    global ser
    print(f"正在连接串口 {SERIAL_PORT}...")
    ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE, bytesize=8, parity='N', stopbits=1, timeout=1)
    print("串口连接成功\n")


# ========== 串口中断处理 ==========
def handle_serial_data():
    global current_mode, last_operation_time
    if ser.in_waiting > 0:
        data = ser.read(ser.in_waiting).decode('ascii', errors='ignore')
        if '0' in data:
            current_mode = 'idle'
        elif '1' in data:
            current_mode = 'tmp'
        elif '2' in data:
            current_mode = 'get'
        elif '3' in data:
            current_mode = 'ocr'
        cv2.destroyAllWindows()
        last_operation_time = 0


# ========== 公共函数 ==========
def get_roi(frame):
    h, w = frame.shape[:2]
    roi_x1 = int(w * (0.6 - ROI_SCALE / 2))
    roi_y1 = int(h * (0.4 - ROI_SCALE / 2))
    roi_x2 = int(w * (0.4 + ROI_SCALE / 2))
    roi_y2 = int(h * (0.6 + ROI_SCALE / 2))
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    return roi, (roi_x1, roi_y1, roi_x2, roi_y2)


def draw_roi(frame, roi_coords):
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    return frame


def get_ocr_roi(frame):
    h, w = frame.shape[:2]
    roi_x1 = int(w * (0.6 - 0.45 / 2))
    roi_y1 = int(h * 0.65)
    roi_x2 = int(w * (0.4 + 0.45 / 2))
    roi_y2 = int(h * 0.85)
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    return roi, (roi_x1, roi_y1, roi_x2, roi_y2)


def draw_ocr_roi(frame, roi_coords):
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    return frame


def send_keyboard_command(key):
    ser.write((key + '\n').encode('utf-8'))
    print(f"黄金矿工发送: {key}")


def reset_instructions():
    global sent_instructions
    sent_instructions = set()
    print("[系统提示] 已重置指令记录")


def can_operate():
    global current_mode, last_operation_time, last_operable_status
    if current_mode == 'idle':
        return True
    if last_operation_time == 0:
        return True
    current_time = time.time()
    interval = operation_intervals.get(current_mode, 0)
    operable = (current_time - last_operation_time) >= interval

    if operable and not last_operable_status:
        print(f"[系统提示] 冷却时间结束，可以执行下一次操作\n")

    last_operable_status = operable
    return operable


# ========== 模式处理函数 ==========
def process_idle_mode(frame):
    cv2.imshow('Idle Mode', frame)


def process_tmp_mode(frame):
    global last_operation_time, last_operable_status

    roi, roi_coords = get_roi(frame)
    frame = draw_roi(frame, roi_coords)

    if can_operate():
        results = yolo_model(roi, imgsz=960, conf=CONF_THRESHOLD, verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                if conf > CONF_THRESHOLD and cls in INSTRUCTION_MAP_TMP:
                    instruction_key = f"TMP_{cls}"  # 创建唯一标识符
                    if instruction_key not in sent_instructions:
                        ser.write((INSTRUCTION_MAP_TMP[cls] + '\n').encode('utf-8'))
                        print(f"检测到TMP物品 {cls}，已发送指令: {INSTRUCTION_MAP_TMP[cls]}")
                        sent_instructions.add(instruction_key)
                        last_operation_time = time.time()
                        last_operable_status = False
                        print(f"[系统提示] 进入冷却时间，请等待{operation_intervals['tmp']}秒")
                    else:
                        print(f"[系统提示] 已发送过TMP物品 {cls} 的指令，不再重复发送")
                    break

    cv2.imshow('TMP Mode', frame)


def process_get_mode(frame):
    global last_operation_time, last_operable_status

    roi, roi_coords = get_roi(frame)
    frame = draw_roi(frame, roi_coords)

    if can_operate():
        results = yolo_model(roi, imgsz=960, conf=CONF_THRESHOLD, verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                if conf > CONF_THRESHOLD and cls in INSTRUCTION_MAP_GET:
                    instruction_key = f"GET_{cls}"  # 创建唯一标识符
                    if instruction_key not in sent_instructions:
                        ser.write((INSTRUCTION_MAP_GET[cls] + '\n').encode('utf-8'))
                        print(f"检测到GET物品 {cls}，已发送指令: {INSTRUCTION_MAP_GET[cls]}")
                        sent_instructions.add(instruction_key)
                        last_operation_time = time.time()
                        last_operable_status = False
                        print(f"[系统提示] 进入冷却时间，请等待{operation_intervals['get']}秒")
                    else:
                        print(f"[系统提示] 已发送过GET物品 {cls} 的指令，不再重复发送")
                    break

    cv2.imshow('GET Mode', frame)


def process_ocr_mode(frame):
    global last_operation_time, last_operable_status

    roi, roi_coords = get_ocr_roi(frame)
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
    frame = draw_ocr_roi(frame, roi_coords)

    if can_operate():
        # 执行OCR（仅在ROI区域内）
        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        result = ocr_model.ocr(img_rgb, cls=True)

        detected_keyword = None

        if result and result[0]:
            # 创建全图副本用于绘制
            frame_with_roi = frame.copy()

            # 创建ROI区域的PIL图像用于绘制文本
            roi_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(roi_pil)
            font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

            for line in result[0]:
                points, (text, conf) = line
                # 检查是否包含关键词
                for keyword in OCR_KEYWORD_MAP:
                    if keyword in text and conf > 0.8:
                        detected_keyword = keyword
                        # 高亮显示识别到的关键词
                        pts = np.array(points, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(roi, [pts], True, (0, 0, 255), 3)
                        draw.text(tuple(map(int, points[0])),f"{text} ({conf:.2f})", font=font, fill=(255, 0, 255))
                        break  # 找到第一个关键词即可

                if detected_keyword:
                    break  # 找到第一个关键词即可

            # 将ROI区域绘制结果合并回原图
            frame_with_roi[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
            frame = frame_with_roi

            # 如果检测到关键词，发送对应指令
            if detected_keyword:
                command_num = OCR_KEYWORD_MAP[detected_keyword]
                instruction_key = f"OCR_{command_num}"  # 创建唯一标识符
                if instruction_key not in sent_instructions:
                    command = INSTRUCTION_MAP_OCR[command_num] + '\n'
                    ser.write(command.encode('utf-8'))
                    print(f"检测到OCR关键词 '{detected_keyword}'，已发送指令: {command.strip()} (对应指令{command_num})")
                    sent_instructions.add(instruction_key)
                    last_operation_time = time.time()
                    last_operable_status = False
                    print(f"[系统提示] 进入冷却时间，请等待{operation_intervals['ocr']}秒")
                else:
                    print(f"[系统提示] 已发送过OCR关键词 '{detected_keyword}' 的指令，不再重复发送")

    cv2.imshow('OCR Mode', frame)


# ========== 主循环 ==========
def main():

    global current_mode

    init_models()
    init_serial()

    cap = cv2.VideoCapture(CAMERA_TYPE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            handle_serial_data()

            ret, frame = cap.read()
            if not ret:
                continue

            if current_mode == 'idle':
                process_idle_mode(frame)
            elif current_mode == 'tmp':
                process_tmp_mode(frame)
            elif current_mode == 'get':
                process_get_mode(frame)
            elif current_mode == 'ocr':
                process_ocr_mode(frame)
            else:
                cv2.imshow('Waiting_Mode', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('x'):
                send_keyboard_command('x')
            elif key == ord('0'):
                current_mode = 'idle'
                cv2.destroyAllWindows()
            elif key == ord('1'):
                current_mode = 'tmp'
                cv2.destroyAllWindows()
            elif key == ord('2'):
                current_mode = 'get'
                cv2.destroyAllWindows()
            elif key == ord('3'):
                current_mode = 'ocr'
                cv2.destroyAllWindows()
            elif key == ord('r'):
                reset_instructions()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()

