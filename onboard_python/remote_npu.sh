set -e
cd "$HOME/model_detection"
echo "=== окружение NPU ==="
python3 -V
if [ ! -d .venv_npu ]; then python3 -m venv .venv_npu; fi
.venv_npu/bin/pip install -q --upgrade pip
.venv_npu/bin/pip install -q "numpy<2" opencv-python-headless
.venv_npu/bin/pip install -q rknn/rknn_toolkit_lite2-*.whl
echo "=== версии ==="
.venv_npu/bin/python -c "import numpy,cv2,rknnlite;print('numpy',numpy.__version__,'| cv2',cv2.__version__)"
echo "=== драйвер NPU ==="
sudo -n dmesg 2>/dev/null | grep -i "Initialized rknpu" | tail -1 || echo "(dmesg без sudo недоступен - не критично)"
echo "=== запуск ==="
cd "$HOME/model_detection"
LD_LIBRARY_PATH="$HOME/model_detection/rknn" .venv_npu/bin/python npu_detect.py
