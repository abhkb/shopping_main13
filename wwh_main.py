# -*- coding: utf-8 -*-

import os
import torch
torch.set_num_threads(1)
import pickle
import cv2
import face_recognition
import numpy as np
from paddleocr import PaddleOCR
import requests
import re
import json
from datetime import datetime
from ultralytics import YOLO
import time
import serial

import warnings
warnings.filterwarnings('ignore')

# ========== 配置部分 ==========

# 相机配置（双摄像头）
CAMERA_ID_1 = 1
CAMERA_ID_2 = 2
CAMERA_BACKEND = cv2.CAP_MSMF
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# 串口配置
SERIAL_PORT = 'COM3'  # 根据实际修改串口号
BAUDRATE = 9600

# 状态定义
STATE_IDLE = "idle"
STATE_FACE_DETECT = "face_detect"
STATE_SAVE = "save"
STATE_OCR = "ocr"
STATE_LLM = "llm"
STATE_TMP = "tmp"
STATE_PUT = "put"
STATE_NEED = "need"
STATE_GIVE = "give"

# 人脸识别阈值
FACE_DISTANCE_THRESHOLD = 0.5

# 保存路径
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Target_Customer')

# 购物清单文件路径
SHOPPING_LIST_PATH = os.path.join(SAVE_PATH, 'shopping_result.json')

# 购物记录文件路径
SHOPPING_RECORD_PATH = os.path.join(SAVE_PATH, 'shopping_record.json')

# LLM 配置
LLM_MODEL = "deepseek-r1:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# YOLO 配置
YOLO_MODEL_PATH = r"C:\Users\39962\work\shopping26-main13\best.pt"
CONF_THRESHOLD = 0.80

# TMP 模式 物品指令映射表
ROI_SCALE = 0.6
INSTRUCTION_MAP_TMP = \
    {
        0: 'a', #vtnai_tblue
        1: 'a', #vtnai_sblue
        2: 'a', #vtnai_zi
        3: 'a', #vtnai_black
        4: 'a', #vtnai_red
        5: 'c', #wangzai_red
        6: 'c', #wangzai_green
        # 7: '0', #pingke
        8: 'd', #taozi
        9: 'c', #wangzai_yellow
        10: 'b',#baishi
    }

# PUT 模式 物品指令映射表
PUT_OCR_ROI_SCALE = 0.5
PUT_COOLDOWN = 2
PUT_OCR_KEYWORDS = \
{
    "维他": 'x',
    "百事": 'y',
    "旺仔": 'm',
    "RIO": 'n',
}

# NEED 模式配置
NEED_OCR_ROI_SCALE = 0.6
NEED_COOLDOWN = 2
NEED_CONFIDENCE_THRESHOLD = 0.5

# 需要发送的指令序列（依次发送，最多4个）
NEED_COMMANDS = ['e', 'f', 'g', 'h']

# 最大收集数量（最多4个）
MAX_COLLECT_COUNT = 4

# GIVE 模式配置（暂时不使用）
GIVE_COOLDOWN = 2

# 冷却时间配置
TMP_COOLDOWN = 10  # 秒
PUT_COOLDOWN = 2  # 秒

# ========== 全局变量 ==========

# 摄像头对象
cap1 = None
cap2 = None
current_camera = 1
  # 1: 摄像头1, 2: 摄像头2

# 串口对象
ser = None

current_state = STATE_IDLE

has_target = False
target_face_image = None
target_face_encoding = None

ocr_model = None
ocr_result = None
shopping_list = []
shopping_progress = {}

# NEED 模式购物清单（原始物品列表）
original_shopping_list = []  # 存储原始物品列表
need_collected_items = []    # 已收集的物品列表
need_collected_count = {}    # 每种物品已收集的数量

yolo_model = None
sent_instructions = set()
last_tmp_operation_time = 0

last_put_operation_time = 0
put_sent_keywords = set()

last_need_operation_time = 0
sent_need_items = set()  # 记录已经发送指令的物品
manual_trigger_need = False  # NEED模式手动触发标志

last_give_operation_time = 0
give_verified = False

manual_trigger = False


# ==================== 文件操作函数 ====================

def init_target_path():
    """创建目标顾客文件夹"""
    global SAVE_PATH
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print(f"✅ 创建目录: {SAVE_PATH}")


def load_shopping_list():
    """加载购物清单（原始物品列表）"""
    global shopping_list, shopping_progress, original_shopping_list, need_collected_items, need_collected_count, sent_need_items

    if not os.path.exists(SHOPPING_LIST_PATH):
        print(f"❌ 购物清单文件不存在: {SHOPPING_LIST_PATH}")
        print("   请先完成人脸识别和OCR解析")
        return False

    try:
        with open(SHOPPING_LIST_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'shopping_list' in data:
                shopping_list = data['shopping_list']
            else:
                shopping_list = data

        shopping_progress = {}
        
        # 构建原始购物清单（展开重复物品）
        original_shopping_list = []
        for item in shopping_list:
            item_name = item.get('item', '未知')
            quantity = item.get('quantity', 1)
            # 限制同种物品最多2个
            if quantity > 2:
                quantity = 2
            brand = item.get('brand', None)
            for i in range(quantity):
                original_shopping_list.append({
                    "item": item_name,
                    "brand": brand,
                    "collected": False
                })
        
        # 初始化已收集记录
        need_collected_items = []
        need_collected_count = {}
        sent_need_items = set()
        
        print(f"✅ 加载购物清单成功，共 {len(original_shopping_list)} 个目标物品")
        for i, item in enumerate(original_shopping_list):
            if item is not None:
                brand_info = f" (品牌: {item['brand']})" if item['brand'] and item['brand'] != "null" else ""
                print(f"   - 物品{i+1}: {item['item']}{brand_info}")
        
        print(f"\n📌 最多收集 {MAX_COLLECT_COUNT} 个物品，按顺序发送指令: {NEED_COMMANDS}")
        
        load_shopping_record()
        return True
    except Exception as e:
        print(f"❌ 加载购物清单失败: {e}")
        return False


def save_shopping_record():
    """保存购物进度记录"""
    global need_collected_items, need_collected_count, sent_need_items
    
    try:
        record = {
            "collected_items": need_collected_items,
            "collected_count": need_collected_count,
            "sent_commands": list(sent_need_items),
            "total_collected": len(need_collected_items),
            "timestamp": datetime.now().isoformat()
        }

        with open(SHOPPING_RECORD_PATH, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"   💾 购物记录已保存")
    except Exception as e:
        print(f"   ⚠️ 保存购物记录失败: {e}")


def load_shopping_record():
    """加载之前的购物进度记录"""
    global need_collected_items, need_collected_count, sent_need_items

    if not os.path.exists(SHOPPING_RECORD_PATH):
        return

    try:
        with open(SHOPPING_RECORD_PATH, 'r', encoding='utf-8') as f:
            record = json.load(f)
        
        need_collected_items = record.get("collected_items", [])
        need_collected_count = record.get("collected_count", {})
        sent_need_items = set(record.get("sent_commands", []))
        
        print(f"📋 加载历史购物记录:")
        print(f"   - 已收集: {len(need_collected_items)} 个物品")
        print(f"   - 已发送指令: {', '.join(sent_need_items) if sent_need_items else '无'}")
        for item_name, count in need_collected_count.items():
            print(f"   - {item_name}: {count}个")
    except Exception as e:
        print(f"加载购物记录失败: {e}")


def reset_shopping_progress():
    """重置购物进度"""
    global original_shopping_list, need_collected_items, need_collected_count, sent_need_items
    # 重置 NEED 模式相关变量
    if original_shopping_list:
        for item in original_shopping_list:
            if item is not None:
                item['collected'] = False
    need_collected_items = []
    need_collected_count = {}
    sent_need_items.clear()
    save_shopping_record()
    print("✅ 购物进度已重置")


def save_target(original_frame, face_image, face_encoding):
    """保存目标顾客到工程"""
    global has_target, target_face_image, target_face_encoding, SAVE_PATH

    original_path = os.path.join(SAVE_PATH, 'target_original.jpg')
    cv2.imwrite(original_path, original_frame)

    face_path = os.path.join(SAVE_PATH, 'target_face.jpg')
    cv2.imwrite(face_path, face_image)

    encoding_path = os.path.join(SAVE_PATH, 'target_face.pkl')
    with open(encoding_path, 'wb') as f:
        pickle.dump(face_encoding, f)

    target_face_encoding = face_encoding
    target_face_image = face_image
    has_target = True

    print("✅ 目标顾客数据保存完成")


def clear_target():
    """清除目标顾客的数据"""
    global has_target, target_face_image, target_face_encoding, SAVE_PATH

    if os.path.exists(SAVE_PATH):
        for f in os.listdir(SAVE_PATH):
            if f.endswith('.pkl') or f.endswith('.jpg') or f.endswith('.json'):
                file_path = os.path.join(SAVE_PATH, f)
                os.remove(file_path)
                print(f"✓ 已删除: {f}")

    has_target = False
    target_face_image = None
    target_face_encoding = None

    print("✅ 目标顾客数据已清除")


def load_target_encoding():
    """加载已保存的目标人脸编码"""
    global target_face_encoding, has_target

    encoding_path = os.path.join(SAVE_PATH, 'target_face.pkl')
    if os.path.exists(encoding_path):
        try:
            with open(encoding_path, 'rb') as f:
                target_face_encoding = pickle.load(f)
            has_target = True
            print("✅ 已加载目标人脸编码")
            return True
        except Exception as e:
            print(f"加载目标人脸编码失败: {e}")
    return False


# ==================== 相机操作函数（双摄像头）====================

def init_cameras():
    """初始化两个摄像头"""
    global cap1, cap2

    print(f"\n正在打开摄像头 {CAMERA_ID_1}...")
    print(f"正在打开摄像头 {CAMERA_ID_2}...")

    cap1 = cv2.VideoCapture(CAMERA_ID_1, CAMERA_BACKEND)
    cap2 = cv2.VideoCapture(CAMERA_ID_2, CAMERA_BACKEND)

    if not cap1.isOpened():
        print("❌ 无法成功打开摄像头1")
        return False

    if not cap2.isOpened():
        print("❌ 无法成功打开摄像头2")
        cap1.release()
        return False

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    ret1, test_frame1 = cap1.read()
    if ret1 and test_frame1 is not None:
        print("✅ 摄像头1初始化成功")
    else:
        print("❌ 摄像头1测试失败")
        cap1.release()
        cap2.release()
        return False

    ret2, test_frame2 = cap2.read()
    if ret2 and test_frame2 is not None:
        print("✅ 摄像头2初始化成功")
    else:
        print("❌ 摄像头2测试失败")
        cap1.release()
        cap2.release()
        return False

    return True


def switch_camera():
    """切换摄像头"""
    global current_camera
    if current_camera == 1:
        current_camera = 2
        print("\n📷 已切换到摄像头2（物品识别）")
    else:
        current_camera = 1
        print("\n📷 已切换到摄像头1（人脸识别）")


def get_current_frame():
    """获取当前摄像头的画面"""
    global cap1, cap2, current_camera

    if current_camera == 1:
        if cap1 is None or not cap1.isOpened():
            return None, "摄像头1未打开"
        ret, frame = cap1.read()
        if ret:
            # 添加摄像头标识
            cv2.putText(frame, "Camera 1 (Face)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return frame, None
        return None, "读取摄像头1失败"
    else:
        if cap2 is None or not cap2.isOpened():
            return None, "摄像头2未打开"
        ret, frame = cap2.read()
        if ret:
            cv2.putText(frame, "Camera 2 (Object)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return frame, None
        return None, "读取摄像头2失败"


def release_cameras():
    """释放摄像头"""
    global cap1, cap2
    if cap1 is not None:
        cap1.release()
        print("✅ 摄像头1已释放")
    if cap2 is not None:
        cap2.release()
        print("✅ 摄像头2已释放")


# ==================== 串口操作函数 ====================

def init_serial():
    """初始化串口"""
    global ser
    print(f"\n正在连接串口 {SERIAL_PORT}...")
    try:
        ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE,
                           bytesize=8, parity='N', stopbits=1, timeout=1)
        print("✅ 串口连接成功")
        return True
    except Exception as e:
        print(f"❌ 串口连接失败: {e}")
        return False


def send_serial_command(command):
    """发送串口指令"""
    global ser
    if ser and ser.is_open:
        ser.write((command + '\n').encode('utf-8'))
        print(f"📤 发送指令: {command}")
    else:
        print(f"⚠️ 串口未连接，无法发送指令: {command}")


def handle_serial_data():
    """处理串口接收的数据，用于模式切换"""
    global current_state, last_tmp_operation_time, last_put_operation_time
    global last_need_operation_time, last_give_operation_time, current_camera, manual_trigger
    global manual_trigger_need, sent_need_items

    if ser is None or not ser.is_open:
        return

    if ser.in_waiting > 0:
        try:
            data = ser.read(ser.in_waiting).decode('ascii', errors='ignore')
            print(f"[串口接收] {data}")

            if '0' in data:
                current_state = STATE_IDLE
                cv2.destroyAllWindows()
                print(f"\n[串口] 状态转换: -> {STATE_IDLE}")
            elif '1' in data:
                current_state = STATE_TMP
                cv2.destroyAllWindows()
                print(f"\n[串口] 状态转换: -> {STATE_TMP}")
            elif '3' in data:
                current_state = STATE_PUT
                cv2.destroyAllWindows()
                print(f"\n[串口] 状态转换: -> {STATE_PUT}")
            elif 'o' in data:
                if current_state == STATE_PUT:
                    manual_trigger = True
                    print(f"\n[串口] 🔔 触发OCR识别请求...")
                elif current_state == STATE_NEED:
                    manual_trigger_need = True
                    print(f"\n[串口] 🔔 触发NEED模式OCR识别请求...")
                else:
                    print(f"\n[串口] ⚠️ 当前状态为 {current_state}，无法触发OCR识别（需要PUT或NEED模式）")
            elif '2' in data:
                if not original_shopping_list:
                    print(f"\n[串口] ⚠️ 购物清单为空，请先完成人脸识别和OCR解析")
                else:
                    current_state = STATE_NEED
                    cv2.destroyAllWindows()
                    print(f"\n[串口] 状态转换: -> {STATE_NEED}")
            elif 'g' in data:
                if not has_target:
                    print(f"\n[串口] ⚠️ 未设置目标顾客，请先保存人脸数据")
                else:
                    current_state = STATE_GIVE
                    cv2.destroyAllWindows()
                    print(f"\n[串口] 状态转换: -> {STATE_GIVE}")
            elif 's' in data:
                current_state = STATE_SAVE
                cv2.destroyAllWindows()
                print(f"\n[串口] 状态转换: -> {STATE_SAVE}")
            elif 't' in data:
                switch_camera()
            elif 'r' in data:
                reset_tmp_instructions()
                reset_put_keywords()
                reset_need_items()
                reset_shopping_progress()
                clear_target()
                print(f"\n[串口] 已重置所有记录")

            # 重置冷却时间
            last_tmp_operation_time = 0
            last_put_operation_time = 0
            last_need_operation_time = 0
            last_give_operation_time = 0

        except Exception as e:
            print(f"串口数据解析错误: {e}")


def release_serial():
    """释放串口"""
    global ser
    if ser is not None and ser.is_open:
        ser.close()
        print("✅ 串口已释放")


# ==================== 人脸识别函数 ====================

def detect_faces(frame):
    """检测图片中的人脸 返回位置和编码"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return face_locations, face_encodings


def get_best_face(frame):
    """获取面积最大的人脸 进行简单滤波"""
    face_locations, face_encodings = detect_faces(frame)

    if len(face_locations) == 0:
        return None, None, None

    areas = [(right - left) * (bottom - top) for (top, right, bottom, left) in face_locations]
    best_index = areas.index(max(areas))

    top, right, bottom, left = face_locations[best_index]
    face_encoding = face_encodings[best_index] if face_encodings else None

    margin = 30
    h, w = frame.shape[:2]
    top_crop = max(0, top - margin)
    bottom_crop = min(h, bottom + margin)
    left_crop = max(0, left - margin)
    right_crop = min(w, right + margin)
    face_image = frame[top_crop:bottom_crop, left_crop:right_crop]

    return face_encoding, face_image, (top, right, bottom, left)


def compare_faces(face_encoding):
    """比较当前人脸与目标人脸"""
    global target_face_encoding

    if target_face_encoding is None:
        print("⚠️ 未加载目标人脸")
        return False, 1.0

    distance = np.linalg.norm(target_face_encoding - face_encoding)
    is_match = distance <= FACE_DISTANCE_THRESHOLD

    return is_match, distance


# ==================== OCR处理函数 ====================

def init_ocr():
    """初始化 PaddleOCR"""
    global ocr_model
    print("\n正在初始化 PaddleOCR ...")
    ocr_model = PaddleOCR(lang='ch')
    print("✅ PaddleOCR 初始化完成")


def ocr_from_image(image_path):
    """从图片中提取文字"""
    global ocr_model

    try:
        result = ocr_model.ocr(image_path)

        extracted_text = []

        if result and isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and 'rec_texts' in item:
                    texts = item.get('rec_texts', [])
                    confidences = item.get('rec_scores', [])
                    for text, conf in zip(texts, confidences):
                        if text and text.strip():
                            extracted_text.append({
                                'text': text,
                                'confidence': float(conf) if conf else 1.0
                            })
                elif isinstance(item, list) and len(item) > 0:
                    for line in item:
                        if len(line) >= 2:
                            if isinstance(line[1], tuple) or isinstance(line[1], list):
                                text = line[1][0] if len(line[1]) > 0 else ''
                                conf = line[1][1] if len(line[1]) > 1 else 1.0
                            else:
                                text = line[1] if len(line) > 1 else ''
                                conf = 1.0
                            if text and text.strip():
                                extracted_text.append({
                                    'text': text,
                                    'confidence': float(conf)
                                })

        if not extracted_text:
            print("❌ 未识别到文字")
            return None, None

        raw_text = ' '.join([item['text'] for item in extracted_text])

        print(f"✅ OCR识别完成，共识别 {len(extracted_text)} 个文本块")
        print(f"📝 原始识别结果: {raw_text}")

        return raw_text, extracted_text

    except Exception as e:
        print(f"❌ OCR识别失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def ocr_from_frame(roi):
    """从ROI区域识别文字"""
    global ocr_model

    if roi is None or roi.size == 0:
        return []

    try:
        if len(roi.shape) == 3:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        else:
            roi_rgb = roi

        result = ocr_model.ocr(roi_rgb)

        extracted_text = []

        if result:
            for page_result in result:
                if page_result is None:
                    continue
                if isinstance(page_result, list):
                    for line in page_result:
                        if line and len(line) >= 2:
                            rec_result = line[1]
                            if isinstance(rec_result, (list, tuple)) and len(rec_result) >= 2:
                                text = rec_result[0]
                                if text and isinstance(text, str) and text.strip():
                                    extracted_text.append(text.strip())
                            elif isinstance(rec_result, str) and rec_result.strip():
                                extracted_text.append(rec_result.strip())
                elif isinstance(page_result, dict):
                    if 'rec_texts' in page_result:
                        texts = page_result.get('rec_texts', [])
                        for text in texts:
                            if text and text.strip():
                                extracted_text.append(text.strip())

        return extracted_text

    except Exception as e:
        print(f"❌ OCR 错误: {e}")
        return []


# ==================== LLM处理函数 ====================

def call_llm(prompt):
    """调用本地大模型"""
    try:
        response = requests.post(OLLAMA_URL,
                                 json={
                                     'model': LLM_MODEL,
                                     'prompt': prompt,
                                     'stream': False,
                                     'options': {'temperature': 0.1, 'num_predict': 1024}
                                 },
                                 timeout=120)

        if response.status_code == 200:
            print(f"✅ 调用大模型成功")
            return response.json()['response']
        return None

    except Exception as e:
        print(f"❌ 调用大模型失败: {e}")
        return None


def check_ocr_text(raw_text):
    """大模型检查文本"""
    prompt = f"""你是一个OCR文本清洗助手。请清理以下OCR识别的文字：

                 原始OCR文字：{raw_text}

                 清洗要求：
                 1. 去除不必要的空格和换行
                 2. 不要去除任何的标点符号
                 3. 将断开的词语连接起来
                 4. 修正明显的OCR识别错误
                 5. 保持原始语义不变
                 6. 只输出清洗后的文本，不要任何解释

                 输出清洗后的文本："""

    response = call_llm(prompt)

    if response:
        cleaned = response.strip().strip('"').strip("'")
        print(f"📝 检查完成: {cleaned[:100]}...")
        return cleaned

    return raw_text


def parse_shopping_list(checked_text):
    """大模型解析购物清单 - 提取6个物品，同种最多重复2次"""
    prompt = f"""你是一个JSON输出器。只输出JSON数组，不要输出任何其他内容。

                 从以下文本提取提取所有提到的商品。

                 OCR识别文本：{checked_text}

                 输出格式：[{{"item": "商品名", "quantity": 数量, "brand": "品牌或null"}}]

                 规则：
                 1. 商品名必须从文本中提取，不要凭空创造，而且商品名只会出现在数量词之后，例如：“2瓶可乐”中的“可乐”；而不会出现在数量词之前，例如“可乐2瓶”中的“可乐”。数量词可以是数字或者中文数字，例如“2”或者“两”
                 2. 如果文本中提到品牌，必须提取到brand字段，如果没有提到品牌，brand字段为null
                 3. quantity只能是1或2，不能超过2
                 4. 最终输出最多只有6个目标
                 5. 输出是一个JSON数组，不要有任何其他内容
                 *. 如果是物品是“纯牛奶”，品牌绑定“光明”
                 *. 如果是物品是“薯片”，品牌绑定“乐事”
                 *. 如果是物品是“可乐”，品牌绑定“百事”

                 现在输出JSON："""

    response = call_llm(prompt)
    if response:
        json_match = re.search(r'\[[\s\S]*?\]', response)
        if json_match:
            try:
                result = json.loads(json_match.group())
                # 合并重复物品并限制数量为2
                merged = {}
                for item in result:
                    item_name = item.get('item', '')
                    brand = item.get('brand', None)
                    quantity = item.get('quantity', 1)
                    if quantity > 2:
                        quantity = 2
                    key = f"{item_name}_{brand}" if brand else item_name
                    if key in merged:
                        new_qty = merged[key]['quantity'] + quantity
                        if new_qty > 2:
                            new_qty = 2
                        merged[key]['quantity'] = new_qty
                    else:
                        merged[key] = item
                        merged[key]['quantity'] = quantity
                return list(merged.values())
            except Exception as e:
                print(f"JSON解析失败: {e}")

    return None


# ==================== 物品识别函数 ====================

def init_yolo():
    """初始化 YOLO 模型"""
    global yolo_model
    print("\n正在加载 YOLO 模型...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO 模型加载完成")


def get_roi(frame):
    """获取 ROI 区域"""
    h, w = frame.shape[:2]
    roi_x1 = int(w * (0.6 - ROI_SCALE / 2))
    roi_y1 = int(h * (0.4 - ROI_SCALE / 2))
    roi_x2 = int(w * (0.4 + ROI_SCALE / 2))
    roi_y2 = int(h * (0.6 + ROI_SCALE / 2))
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    return roi, (roi_x1, roi_y1, roi_x2, roi_y2)


def draw_roi(frame, roi_coords):
    """绘制 ROI 区域"""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    return frame


def can_operate_tmp():
    """检查 TMP 模式是否可以操作（冷却时间）"""
    global last_tmp_operation_time
    current_time = time.time()
    if last_tmp_operation_time == 0:
        return True
    return (current_time - last_tmp_operation_time) >= TMP_COOLDOWN


def reset_tmp_instructions():
    """重置 TMP 模式已发送指令记录"""
    global sent_instructions
    sent_instructions.clear()
    print("✅ TMP 模式指令记录已重置")


# ==================== PUT 模式函数 ====================

def get_put_roi(frame):
    """获取 PUT 模式的 ROI 区域"""
    h, w = frame.shape[:2]
    roi_x1 = int(w * (0.5 - PUT_OCR_ROI_SCALE / 2))
    roi_y1 = int(h * (0.5 - PUT_OCR_ROI_SCALE / 2))
    roi_x2 = int(w * (0.5 + PUT_OCR_ROI_SCALE / 2))
    roi_y2 = int(h * (0.5 + PUT_OCR_ROI_SCALE / 2))
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    return roi, (roi_x1, roi_y1, roi_x2, roi_y2)


def draw_put_roi(frame, roi_coords):
    """绘制 PUT 模式的 ROI 区域"""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    cv2.putText(frame, "OCR ROI", (roi_x1, roi_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame


def can_operate_put():
    """检查 PUT 模式是否可以操作（冷却时间）"""
    global last_put_operation_time
    current_time = time.time()
    if last_put_operation_time == 0:
        return True
    return (current_time - last_put_operation_time) >= PUT_COOLDOWN


def reset_put_keywords():
    """重置 PUT 模式已发送关键词记录"""
    global put_sent_keywords, last_put_operation_time
    put_sent_keywords.clear()
    last_put_operation_time = 0
    print("✅ PUT 模式关键词记录已重置")


def process_put_mode(frame):
    """PUT 模式：按 'o' 键或串口接收 'o' 手动触发 OCR 识别"""
    global last_put_operation_time, put_sent_keywords, manual_trigger

    roi, roi_coords = get_put_roi(frame)
    frame = draw_put_roi(frame, roi_coords)

    current_time = time.time()

    if last_put_operation_time > 0:
        remaining = max(0, PUT_COOLDOWN - (current_time - last_put_operation_time))
        cv2.putText(frame, f"Cooldown: {remaining:.1f}s", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if manual_trigger and can_operate_put():
        manual_trigger = False

        print("\n" + "=" * 40)
        print("🔍 正在识别商品...")
        print("=" * 40)

        scale = 0.5
        small_roi = cv2.resize(roi, (0, 0), fx=scale, fy=scale)

        # 保存ROI区域图片用于调试（可选）
        debug_path = os.path.join(SAVE_PATH, 'put_roi_debug.jpg')
        cv2.imwrite(debug_path, small_roi)
        print(f"📷 ROI区域已保存: {debug_path}")

        texts = ocr_from_frame(small_roi)

        if texts:
            recognized_text = ' '.join(texts)
            print(f"📝 识别到的文字: {recognized_text}")

            matched = False
            for keyword, message in PUT_OCR_KEYWORDS.items():
                if keyword in recognized_text and keyword not in put_sent_keywords:
                    # 发送串口指令
                    send_serial_command(message)
                    put_sent_keywords.add(keyword)
                    last_put_operation_time = time.time()
                    print(f"🎯 关键词 '{keyword}' 识别成功，发送指令: {message}")
                    matched = True
                    break
                elif keyword in recognized_text and keyword in put_sent_keywords:
                    print(f"⏭️ 关键词 '{keyword}' 已识别过，跳过")
                    matched = True
                    break
            
            if not matched:
                print("❌ 未匹配到任何关键词")
        else:
            print("❌ OCR 识别失败或未识别到文字")

        last_put_operation_time = time.time()
        print("=" * 40 + "\n")

    cv2.imshow(current_state.upper(), frame)


# ==================== NEED 模式函数 ====================

def get_need_roi(frame):
    """获取 NEED 模式的 ROI 区域"""
    h, w = frame.shape[:2]
    roi_x1 = int(w * (0.5 - NEED_OCR_ROI_SCALE / 2))
    roi_y1 = int(h * (0.5 - NEED_OCR_ROI_SCALE / 2))
    roi_x2 = int(w * (0.5 + NEED_OCR_ROI_SCALE / 2))
    roi_y2 = int(h * (0.5 + NEED_OCR_ROI_SCALE / 2))
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    return roi, (roi_x1, roi_y1, roi_x2, roi_y2)


def draw_need_roi(frame, roi_coords):
    """绘制 NEED 模式的 ROI 区域"""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
    cv2.putText(frame, "NEED ROI", (roi_x1, roi_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame


def can_operate_need():
    """检查 NEED 模式是否可以操作（冷却时间）"""
    global last_need_operation_time
    current_time = time.time()
    if last_need_operation_time == 0:
        return True
    return (current_time - last_need_operation_time) >= NEED_COOLDOWN


def reset_need_items():
    """重置 NEED 模式已发送物品记录"""
    global sent_need_items, last_need_operation_time
    sent_need_items.clear()
    last_need_operation_time = 0
    print("✅ NEED 模式物品记录已重置")


def get_next_command():
    """获取下一个可用的串口指令（按顺序e,f,g,h）"""
    used_count = len(sent_need_items)
    if used_count < len(NEED_COMMANDS):
        return NEED_COMMANDS[used_count]
    return None


def match_shopping_item_direct(recognized_text):
    """
    直接从原始购物清单中匹配物品
    优先级：品牌匹配 > 物品名匹配
    """
    global original_shopping_list, need_collected_count
    
    if not original_shopping_list:
        print("⚠️ 购物清单为空")
        return None
    
    print(f"🔍 匹配文本: {recognized_text}")
    
    # 获取还未收集完成的目标物品
    remaining_targets = []
    for item in original_shopping_list:
        if item is not None:
            item_name = item['item']
            collected_count = need_collected_count.get(item_name, 0)
            total_needed = sum(1 for x in original_shopping_list if x is not None and x['item'] == item_name)
            if collected_count < total_needed:
                remaining_targets.append(item)
    
    if not remaining_targets:
        print("✅ 所有目标物品已收集完毕")
        return None
    
    print(f"📋 剩余目标物品: {[item['item'] for item in remaining_targets]}")
    
    # 第一轮：优先匹配品牌（如果购物清单中有指定品牌）
    for target in remaining_targets:
        brand = target.get('brand')
        item_name = target['item']
        
        # 如果有品牌要求，先匹配品牌
        if brand and brand != "null":
            if brand in recognized_text:
                print(f"✅ 品牌匹配成功: 品牌 '{brand}' -> 物品 '{item_name}'")
                return target
    
    # 第二轮：匹配物品名
    for target in remaining_targets:
        item_name = target['item']
        if item_name in recognized_text:
            print(f"✅ 物品名匹配成功: '{item_name}'")
            return target
    
    # 第三轮：模糊匹配（包含关系）
    for target in remaining_targets:
        item_name = target['item']
        if item_name in recognized_text or recognized_text in item_name:
            print(f"✅ 模糊匹配成功: '{item_name}'")
            return target
    
    # 第四轮：关键词匹配
    for target in remaining_targets:
        item_name = target['item']
        if len(item_name) >= 2:
            keyword = item_name[:2]
            if keyword in recognized_text:
                print(f"✅ 关键词匹配成功: '{keyword}' -> '{item_name}'")
                return target
    
    print("❌ 未匹配到任何剩余目标物品")
    return None


def mark_item_collected_direct(item_name, brand=None):
    """直接标记物品为已收集"""
    global need_collected_items, need_collected_count
    
    if item_name not in need_collected_count:
        need_collected_count[item_name] = 0
    
    need_collected_count[item_name] += 1
    need_collected_items.append({
        "item": item_name,
        "brand": brand,
        "timestamp": datetime.now().isoformat()
    })
    
    print(f"✅ 已标记物品 '{item_name}' 为已收集 (第{need_collected_count[item_name]}个)")


def get_collection_progress():
    """获取收集进度"""
    collected = len(need_collected_items)
    total = min(MAX_COLLECT_COUNT, len([x for x in original_shopping_list if x is not None]))
    return collected, total


def process_need_mode(frame):
    """NEED 模式：OCR识别物品，从6个物品中匹配，匹配成功直接发送e,f,g,h指令，最多收集4个"""
    global last_need_operation_time, sent_need_items, manual_trigger_need
    global original_shopping_list, need_collected_items, need_collected_count

    # 获取 ROI 区域
    roi, roi_coords = get_need_roi(frame)
    frame = draw_need_roi(frame, roi_coords)

    # 显示冷却时间
    current_time = time.time()
    if last_need_operation_time > 0:
        remaining = max(0, NEED_COOLDOWN - (current_time - last_need_operation_time))
        cv2.putText(frame, f"Cooldown: {remaining:.1f}s", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 显示已发送指令
    if sent_need_items:
        sent_text = f"Sent: {', '.join(list(sent_need_items))}"
        cv2.putText(frame, sent_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示收集进度
    collected, total = get_collection_progress()
    progress_text = f"Progress: {collected}/{total} (Max: {MAX_COLLECT_COUNT})"
    cv2.putText(frame, progress_text, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # 显示剩余待匹配的物品（未收集的）
    remaining_items = []
    for item in original_shopping_list:
        if item is not None:
            item_name = item['item']
            collected_count = need_collected_count.get(item_name, 0)
            total_needed = sum(1 for x in original_shopping_list if x is not None and x['item'] == item_name)
            if collected_count < total_needed:
                brand_info = f"({item['brand']})" if item.get('brand') and item['brand'] != "null" else ""
                remaining_items.append(f"{item_name}{brand_info}")
    
    if remaining_items:
        y_offset = 140
        cv2.putText(frame, "Need to collect:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        for i, item_text in enumerate(remaining_items[:4]):
            status = f"  {i+1}. {item_text}"
            cv2.putText(frame, status, (15, y_offset + (i+1) * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # 显示匹配提示
    cv2.putText(frame, "Place item in ROI and press 'o' to scan & collect", (10, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 检查是否触发识别
    trigger = manual_trigger_need
    if trigger and can_operate_need():
        manual_trigger_need = False
        
        # 检查是否已经达到最大收集数量
        if len(need_collected_items) >= MAX_COLLECT_COUNT:
            print("\n" + "=" * 50)
            print("⚠️ 已达到最大收集数量，不再处理新的识别请求")
            print(f"   已收集 {len(need_collected_items)}/{MAX_COLLECT_COUNT} 个物品")
            print("=" * 50 + "\n")
            cv2.imshow(current_state.upper(), frame)
            return
        
        print("\n" + "=" * 50)
        print("🔍 NEED模式：正在识别物品...")
        print("=" * 50)

        # 预处理 ROI
        scale = 0.5
        small_roi = cv2.resize(roi, (0, 0), fx=scale, fy=scale)
        
        # 保存调试图片
        debug_path = os.path.join(SAVE_PATH, 'need_roi_debug.jpg')
        cv2.imwrite(debug_path, small_roi)
        print(f"📷 ROI区域已保存: {debug_path}")

        # OCR 识别
        texts = ocr_from_frame(small_roi)

        if texts:
            recognized_text = ' '.join(texts)
            print(f"📝 识别到的文字: {recognized_text}")

            # 匹配购物清单
            matched_item = match_shopping_item_direct(recognized_text)
            
            if matched_item:
                item_name = matched_item['item']
                brand = matched_item.get('brand')
                
                # 检查该物品是否已经收集完成
                collected_count = need_collected_count.get(item_name, 0)
                total_needed = sum(1 for x in original_shopping_list if x is not None and x['item'] == item_name)
                
                if collected_count >= total_needed:
                    print(f"⏭️ 物品 '{item_name}' 已经收集完成，跳过")
                else:
                    # 获取下一个可用的指令
                    next_command = get_next_command()
                    
                    if next_command:
                        # 发送串口指令
                        send_serial_command(next_command)
                        sent_need_items.add(next_command)
                        mark_item_collected_direct(item_name, brand)
                        last_need_operation_time = time.time()
                        
                        # 获取品牌信息
                        brand_info = f" (品牌: {brand})" if brand and brand != "null" else ""
                        
                        collected, total = get_collection_progress()
                        print(f"🎯 匹配成功: '{item_name}'{brand_info} -> 发送指令 '{next_command}'")
                        print(f"📊 收集进度: {collected}/{total}")
                        
                        # 保存进度
                        save_shopping_record()
                        
                        # 检查是否已经收集完所有物品
                        if collected >= total or len(need_collected_items) >= MAX_COLLECT_COUNT:
                            print("\n🎉 恭喜！所有目标物品已全部收集完成！")
                            print(f"💡 共收集 {len(need_collected_items)} 个物品，指令已发送: {', '.join(sent_need_items)}")
                    else:
                        print("⚠️ 没有可用的指令了（已达上限4个）")
            else:
                print("❌ 未匹配到购物清单中的物品")
        else:
            print("❌ OCR 识别失败或未识别到文字")

        print("=" * 50 + "\n")

    cv2.imshow(current_state.upper(), frame)


# ==================== 其他模式处理函数 ====================

def process_idle_mode(frame):
    cv2.imshow(current_state.upper(), frame)


def process_face_detect_mode(frame):
    face_locations, _ = detect_faces(frame)
    for (top, right, bottom, left) in face_locations:
        color = (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.imshow(current_state.upper(), frame)
    return face_locations


def process_save_mode(frame):
    global has_target, current_state

    print("\n" + "=" * 40)
    print("正在保存目标顾客...")
    print("=" * 40)

    face_locations = detect_faces(frame)[0]

    if len(face_locations) == 0:
        print("❌ 没有人脸 无法保存")
        return

    face_encoding, face_image, _ = get_best_face(frame)

    if face_encoding is not None:
        save_target(frame, face_image, face_encoding)
        has_target = True
    else:
        print("❌ 人脸特征提取失败")

    if has_target:
        current_state = STATE_OCR
        print(f"\n状态转换: 当前状态 -> {STATE_OCR}")
    else:
        current_state = STATE_IDLE
        print(f"\n状态转换: 当前状态 -> {STATE_IDLE}")


def process_ocr_mode():
    global current_state, ocr_result

    print("\n" + "=" * 40)
    print("正在提取顾客需求...")
    print("=" * 40)

    target_image_path = os.path.join(SAVE_PATH, 'target_original.jpg')

    if not os.path.exists(target_image_path):
        print(f"❌ 请先保存目标顾客数据")
        return False

    raw_text, _ = ocr_from_image(target_image_path)

    ocr_result = raw_text

    if ocr_result:
        print(f"\n状态转换: {STATE_OCR} -> {STATE_LLM}")
        current_state = STATE_LLM
    else:
        print(f"\n状态转换: {STATE_OCR} -> {STATE_IDLE}")
        current_state = STATE_IDLE


def process_llm_mode(raw_text):
    global current_state

    print("\n" + "=" * 40)
    print("正在检查顾客需求...")
    print("=" * 40)

    checked_text = check_ocr_text(raw_text)

    print("\n" + "=" * 40)
    print("正在确认顾客需求...")
    print("=" * 40)

    parsed_result = parse_shopping_list(checked_text)

    if parsed_result:
        print("🔍 顾客需求分析结果：")
        print(json.dumps(parsed_result, ensure_ascii=False, indent=2))

        result_path = os.path.join(SAVE_PATH, 'shopping_result.json')
        result_record = {
            "created_at": datetime.now().isoformat(),
            "raw_ocr_text": raw_text,
            "checked_text": checked_text,
            "shopping_list": parsed_result,
            "model_used": LLM_MODEL
        }
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_record, f, ensure_ascii=False, indent=2)

        print(f"✅ 顾客需求已保存")
        load_shopping_list()

        current_state = STATE_IDLE
        print(f"\n状态转换: 当前状态 -> {STATE_IDLE}")

        return True
    else:
        print("❌ 顾客需求分析失败")
        return False


def process_tmp_mode(frame):
    """TMP 模式：YOLO 物品识别 - 相同指令只发送一次"""
    global last_tmp_operation_time

    roi, roi_coords = get_roi(frame)
    frame = draw_roi(frame, roi_coords)  

    if can_operate_tmp():
        results = yolo_model(roi, imgsz=640, conf=CONF_THRESHOLD, verbose=False)

        detected = False
        for r in results:
            for box in r.boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()

                if conf > CONF_THRESHOLD and cls in INSTRUCTION_MAP_TMP:
                    # 获取该类别对应的串口指令
                    command = INSTRUCTION_MAP_TMP[cls]
                    
                    # 使用指令作为key，而不是类别ID
                    instruction_key = f"TMP_CMD_{command}"

                    if instruction_key not in sent_instructions:
                        send_serial_command(command)
                        print(f"🎯 检测到物品 {cls}，指令: {command} (置信度: {conf:.2%})")
                        sent_instructions.add(instruction_key)
                        last_tmp_operation_time = time.time()
                        detected = True
                        print(f"⏰ 进入冷却时间 {TMP_COOLDOWN} 秒")
                    else:
                        print(f"⏭️ 指令 '{command}' 已发送过，跳过 (当前物品类别: {cls})(置信度: {conf:.2%})")

                    if detected:
                        break
            if detected:
                break

    cv2.putText(frame, f"Cooldown: {max(0, TMP_COOLDOWN - (time.time() - last_tmp_operation_time)):.1f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow(current_state.upper(), frame)


# ========== 主循环 ==========

def main():
    global current_state, has_target, target_face_image, target_face_encoding
    global manual_trigger, current_camera, manual_trigger_need

    init_target_path()
    clear_target()
    reset_tmp_instructions()
    reset_put_keywords()
    reset_need_items()

    init_ocr()
    init_yolo()

    # 初始化双摄像头
    if not init_cameras():
        print("❌ 摄像头初始化失败，程序退出")
        return

    init_serial()

    # 默认使用摄像头2（物品识别）
    current_camera = 2

    load_shopping_list()
    load_target_encoding()

    try:
        while True:
            handle_serial_data()

            # 获取当前摄像头画面
            frame, error = get_current_frame()
            if error:
                continue
            if frame is None:
                continue

            # 状态处理
            if current_state == STATE_IDLE:
                process_idle_mode(frame)
            elif current_state == STATE_FACE_DETECT:
                process_face_detect_mode(frame)
            elif current_state == STATE_SAVE:
                process_save_mode(frame)
            elif current_state == STATE_OCR:
                process_ocr_mode()
            elif current_state == STATE_LLM:
                process_llm_mode(ocr_result)
            elif current_state == STATE_TMP:
                process_tmp_mode(frame)
            elif current_state == STATE_PUT:
                process_put_mode(frame)
            elif current_state == STATE_NEED:
                process_need_mode(frame)
            elif current_state == STATE_GIVE:
                # GIVE模式暂未实现
                cv2.imshow(current_state.upper(), frame)
            else:
                cv2.imshow('Waiting', frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n退出程序...")
                break
            elif key == ord('0'):
                current_state = STATE_IDLE
                cv2.destroyAllWindows()
                print(f"\n状态转换: -> {STATE_IDLE}")
            elif key == ord('f'):
                current_state = STATE_FACE_DETECT
                cv2.destroyAllWindows()
                print(f"\n状态转换: -> {STATE_FACE_DETECT}")
            elif key == ord('s'):
                current_state = STATE_SAVE
                cv2.destroyAllWindows()
                print(f"\n状态转换: -> {STATE_SAVE}")
            elif key == ord('1'):
                current_state = STATE_TMP
                cv2.destroyAllWindows()
                print(f"\n状态转换: -> {STATE_TMP}")
            elif key == ord('3'):
                current_state = STATE_PUT
                cv2.destroyAllWindows()
                print(f"\n状态转换: -> {STATE_PUT}")
            elif key == ord('2'):
                if not original_shopping_list:
                    print(f"\n⚠️ 购物清单为空，请先完成人脸识别和OCR解析")
                else:
                    current_state = STATE_NEED
                    cv2.destroyAllWindows()
                    print(f"\n状态转换: -> {STATE_NEED}")
            elif key == ord('g'):
                if not has_target:
                    print(f"\n⚠️ 未设置目标顾客，请先保存人脸数据")
                else:
                    current_state = STATE_GIVE
                    cv2.destroyAllWindows()
                    print(f"\n状态转换: -> {STATE_GIVE}")
            elif key == ord('o'):
                if current_state == STATE_PUT:
                    manual_trigger = True
                    print(f"\n🔔 触发OCR识别请求...")
                elif current_state == STATE_NEED:
                    manual_trigger_need = True
                    print(f"\n🔔 触发NEED模式OCR识别请求...")
                else:
                    print(f"\n⚠️ 当前状态为 {current_state}，无法触发OCR识别（需要PUT或NEED模式）")
            elif key == ord('r'):
                reset_tmp_instructions()
                reset_put_keywords()
                reset_need_items()
                reset_shopping_progress()
                clear_target()
                print(f"\n✅ 已重置所有记录")
            elif key == ord('t'):
                # 按键切换摄像头
                switch_camera()
            # 手动发送串口指令（调试用）- NEED模式使用e,f,g,h
            elif key == ord('e'):
                send_serial_command('e')
            elif key == ord('f'):
                send_serial_command('f')
            elif key == ord('g'):
                send_serial_command('g')
            elif key == ord('h'):
                send_serial_command('h')
            # PUT模式使用的指令
            elif key == ord('x'):
                send_serial_command('x')
            elif key == ord('y'):
                send_serial_command('y')
            elif key == ord('m'):
                send_serial_command('m')
            elif key == ord('n'):
                send_serial_command('n')
            # TMP模式使用的指令
            elif key == ord('a'):
                send_serial_command('a')
            elif key == ord('b'):
                send_serial_command('b')
            elif key == ord('c'):
                send_serial_command('c')
            elif key == ord('d'):
                send_serial_command('d')

    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        release_serial()
        release_cameras()
        cv2.destroyAllWindows()
        print("✅ 程序已退出")


if __name__ == "__main__":
    main()