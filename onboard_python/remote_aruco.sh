set -e
cd "$HOME/model_detection/aruco"
echo "=== версия OpenCV на плате ==="
../.venv_npu/bin/python -c "import cv2; print(cv2.__version__, 'aruco:', hasattr(cv2,'aruco'))"
echo
echo "=== замер профилей на плате ==="
../.venv_npu/bin/python profile_bench.py
echo
echo "=== прогон детектора (профиль по умолчанию) ==="
../.venv_npu/bin/python aruco_eval.py
