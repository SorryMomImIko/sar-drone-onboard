set -e
cd "$HOME/sar_yolo_pipeline"
echo "=== проверка зависимостей ==="
MISSING=""
command -v g++   >/dev/null || MISSING="$MISSING build-essential"
command -v cmake >/dev/null || MISSING="$MISSING cmake"
command -v pkg-config >/dev/null || MISSING="$MISSING pkg-config"
pkg-config --exists opencv4        || MISSING="$MISSING libopencv-dev"
pkg-config --exists gstreamer-1.0  || MISSING="$MISSING libgstreamer1.0-dev"
pkg-config --exists gstreamer-app-1.0 || MISSING="$MISSING libgstreamer-plugins-base1.0-dev"
if [ -n "$MISSING" ]; then
  echo "не хватает:$MISSING - ставлю (нужен пароль sudo)"
  sudo dpkg --configure -a || true
  sudo apt-get update -qq || true
  sudo apt-get install -y -qq --allow-downgrades --allow-change-held-packages $MISSING librga-dev \
    || for p in $MISSING; do sudo apt-get install -y -qq --allow-downgrades "$p" || echo "  не удалось: $p"; done
else
  echo "всё на месте, apt не нужен"
fi
for c in g++ cmake pkg-config; do command -v $c >/dev/null && echo "  есть: $c" || echo "  НЕТ: $c"; done
pkg-config --exists gstreamer-1.0 && echo "  есть: gstreamer-1.0" || echo "  НЕТ: gstreamer-1.0"
pkg-config --exists opencv4 && echo "  есть: opencv4" || echo "  НЕТ: opencv4"
pkg-config --exists gstreamer-video-1.0 && echo "  есть: gstreamer-video-1.0" || echo "  нет gstreamer-video-1.0 (ожидаемо, обойдём)"

echo
echo "=== сборка ==="
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DRKNN_ROOT="$PWD/third_party/rknpu2"
cmake --build build -j4
ls -la build/sar_yolo_pipeline
echo
echo "=== готовлю тестовую последовательность кадров ==="
rm -rf test_seq && mkdir -p test_seq
i=0
for f in "$HOME/model_detection/calib"/*.jpg; do
  cp "$f" "$(printf "test_seq/%06d.jpg" $i)"
  i=$((i+1))
done
echo "кадров: $i"
echo
echo "=== прогон пайплайна без камеры (60 секунд) ==="
export LD_LIBRARY_PATH="$PWD/third_party/rknpu2/lib:$LD_LIBRARY_PATH"
timeout 60 ./build/sar_yolo_pipeline \
  "multifilesrc location=$PWD/test_seq/%06d.jpg index=0 caps=image/jpeg,framerate=15/1 ! jpegdec ! videoconvert ! video/x-raw,format=NV12 ! queue max-size-buffers=1 leaky=downstream ! appsink name=sink" \
  || echo "(завершено по таймауту - это ожидаемо)"
