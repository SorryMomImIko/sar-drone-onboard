#pragma once
// TelemetrySourceFactory.hpp - выбор протокола по названию.
//
// ЕДИНСТВЕННОЕ МЕСТО, которое придётся тронуть при смене протокола на
// полётнике. Новый протокол - это новый файл рядом с MavlinkThread.hpp,
// реализующий ITelemetrySource, и одна строка здесь.

#include "MspThread.hpp"
#ifdef SAR_WITH_MAVLINK
#include "MavlinkThread.hpp"
#endif
#include "TelemetrySource.hpp"

#include <memory>
#include <string>

namespace sar {

inline std::unique_ptr<ITelemetrySource> make_telemetry_source(
        const std::string& protocol, AttitudeBuffer& att, PositionBuffer& pos) {
#ifdef SAR_WITH_MAVLINK
    // ArduPilot / PX4 - собирается только с -DSAR_WITH_MAVLINK=ON
    if (protocol == "mavlink")
        return std::unique_ptr<ITelemetrySource>(new MavlinkThread(att, pos));
#endif

    // Betaflight / INAV по USB (Foxeer F405 V2 и подобные)
    if (protocol == "msp")
        return std::unique_ptr<ITelemetrySource>(new MspThread(att, pos));

    // Сюда добавляются другие протоколы, например:
    //   if (protocol == "crsf") return ... CrsfThread(att, pos);
    // Что именно писать - определяется дампом с реального полётника,
    // см. tools/sniff_serial.py.

    return nullptr;
}

} // namespace sar
