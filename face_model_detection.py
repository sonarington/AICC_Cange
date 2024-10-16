import tensorflow as tf
from PIL import Image
import numpy as np
from flask_cors import CORS
import base64
import sys
import os
from flask import Flask, request, jsonify
import io
import json
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 오류 메시지만 표시

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

# 얼굴형 클래스 라벨 정의
classes = ['Round', 'Square','Oblong']

# 모델 로드
model = tf.keras.models.load_model('face_shape_cnn_model.keras')

# 이미지 불러오기 및 전처리
def analyze_face(image):
    image = image.resize((100, 100)) # 이미지 리사이징
    image_array = np.array(image) / 255.0 # 0~1 사이 값으로 정규화
    image_array = np.expand_dims(image_array, axis=0)

    # 분석 예측
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions)

    # 결과 반환
    predicted_class = classes[predicted_index]
    result = {"predicted_class": predicted_class}
    # return result
    return json.dumps(result)

if __name__ == "__main__":
    buffer = sys.stdin.buffer.read()
    nparr = np.frombuffer(buffer, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    analysis_result = analyze_face(Image.fromarray(image))
    
    # print("analysis result:", analysis_result)
    # print(analysis_result)
    # 결과 출력
    sys.stdout.write(analysis_result)