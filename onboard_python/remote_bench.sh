set -e
cd "$HOME/model_detection"
.venv_npu/bin/pip install -q onnxruntime 2>&1 | tail -2
echo "=== CPU платы, onnxruntime, те же условия что на ноуте ==="
nproc
.venv_npu/bin/python bench_laptop.py
