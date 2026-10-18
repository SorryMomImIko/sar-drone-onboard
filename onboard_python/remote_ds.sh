set -e
cd "$HOME/model_detection"
echo "=== прогон по выборке на NPU ==="
LD_LIBRARY_PATH="$HOME/model_detection/rknn" .venv_npu/bin/python npu_dataset.py
echo
echo "=== замер CPU платы, onnxruntime ==="
nproc
.venv_npu/bin/pip install -q onnxruntime 2>&1 | tail -1
.venv_npu/bin/python bench_laptop.py || echo "(замер CPU не прошёл)"
