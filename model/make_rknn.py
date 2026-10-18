"""Полный маршрут best.pt -> best_i8_split.rknn для RK3566.

Запускать на x86_64 Linux (Python 3.10-3.12):
    pip install rknn-toolkit2==2.3.2 "onnx==1.17.0" "numpy==1.26.4" "setuptools<81"
    pip install ultralytics==8.4.155        # нужен только для шага 1
    python make_rknn.py

Шаг 2 - ключевой. Без разделения выходов int8-модель выдаёт НОЛЬ детекций:
координаты рамок (0..640) и вероятности классов (0..1) лежат в одном тензоре,
общий масштаб квантизации берётся по рамкам, и все вероятности схлопываются в 0.
"""
import onnx
from onnx import helper, TensorProto
from ultralytics import YOLO
from rknn.api import RKNN

# 1. экспорт в ONNX
YOLO('best.pt').export(format='onnx', opset=12, simplify=True, imgsz=640, dynamic=False)

# 2. разделение выхода Concat на два тензора: рамки и вероятности
m = onnx.load('best.onnx'); g = m.graph
BOX, CLS = '/model.22/Mul_2_output_0', '/model.22/Sigmoid_output_0'
del g.output[:]
g.output.extend([
    helper.make_tensor_value_info(BOX, TensorProto.FLOAT, [1, 4, 8400]),
    helper.make_tensor_value_info(CLS, TensorProto.FLOAT, [1, 3, 8400]),
])
onnx.checker.check_model(m)
onnx.save(m, 'best_split.onnx')

# 3. квантизация под RK3566 по dataset.txt (296 кадров, letterbox 640x640)
rknn = RKNN(verbose=False)
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
            target_platform='rk3566',      # НЕ rk3588, как в README пайплайна
            quantized_dtype='w8a8', optimization_level=3)
rknn.load_onnx(model='best_split.onnx')
rknn.build(do_quantization=True, dataset='dataset.txt')
rknn.export_rknn('best_i8_split.rknn')
rknn.release()
print('готово: best_i8_split.rknn')
