set -e
cd "$HOME/model_detection"
echo "=== ArUco: замер профилей на плате ==="
LD_LIBRARY_PATH="$PWD/rknn" .venv_npu/bin/python aruco/profile_bench.py
echo
echo "=== ArUco: дальность по разметке ==="
cd aruco && LD_LIBRARY_PATH="$HOME/model_detection/rknn" ../.venv_npu/bin/python aruco_eval.py && cd ..
echo
echo "=== Совмещённый прогон: люди на NPU + ArUco на CPU ==="
LD_LIBRARY_PATH="$PWD/rknn" .venv_npu/bin/python detect_both.py aruco/images
