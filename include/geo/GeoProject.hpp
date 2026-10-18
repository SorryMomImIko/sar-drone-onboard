// GeoProject.hpp — пиксель кадра -> широта/долгота точки на земле.
// Порт geoproject.py на C++, без зависимостей (своя матрица 3x3).
//
// Система координат NED: x на север, y на восток, z ВНИЗ.
#pragma once

#include <cmath>
#include <optional>
#include "TelemetryBuffers.hpp"

namespace sar {

constexpr double R_EARTH = 6378137.0;

struct Vec3 { double x = 0, y = 0, z = 0; };

struct Mat3 {
    double m[3][3]{};
    static Mat3 identity() {
        Mat3 r; r.m[0][0] = r.m[1][1] = r.m[2][2] = 1.0; return r;
    }
    Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) r.m[i][j] += m[i][k] * o.m[k][j];
        return r;
    }
    Vec3 operator*(const Vec3& v) const {
        return { m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
                 m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
                 m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z };
    }
};

// Внутренние параметры камеры — из cv2.calibrateCamera.
struct CameraIntrinsics {
    double fx = 1065.1, fy = 1065.1;   // пиксели
    double cx = 640.0,  cy = 360.0;
    int    width = 1280, height = 720;

    static CameraIntrinsics from_fov(double fov_h_deg, int w, int h) {
        CameraIntrinsics c;
        c.fx = c.fy = (w / 2.0) / std::tan(fov_h_deg * M_PI / 180.0 / 2.0);
        c.cx = w / 2.0; c.cy = h / 2.0; c.width = w; c.height = h;
        return c;
    }

    // Пиксель -> единичный луч в системе камеры (OpenCV: x вправо, y вниз, z вперёд).
    Vec3 ray(double u, double v) const {
        Vec3 d{ (u - cx) / fx, (v - cy) / fy, 1.0 };
        const double n = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
        return { d.x / n, d.y / n, d.z / n };
    }
};

// Ориентация корпуса в мире: Rz(yaw) * Ry(pitch) * Rx(roll).
inline Mat3 R_ned_from_body(double roll, double pitch, double yaw) {
    const double cr = std::cos(roll),  sr = std::sin(roll);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const double cy = std::cos(yaw),   sy = std::sin(yaw);

    Mat3 Rx; Rx.m[0][0]=1; Rx.m[1][1]=cr; Rx.m[1][2]=-sr; Rx.m[2][1]=sr; Rx.m[2][2]=cr;
    Mat3 Ry; Ry.m[0][0]=cp; Ry.m[0][2]=sp; Ry.m[1][1]=1; Ry.m[2][0]=-sp; Ry.m[2][2]=cp;
    Mat3 Rz; Rz.m[0][0]=cy; Rz.m[0][1]=-sy; Rz.m[1][0]=sy; Rz.m[1][1]=cy; Rz.m[2][2]=1;
    return Rz * Ry * Rx;
}

// Как камера привинчена к дрону. tilt_deg = 0 — строго вниз, верх кадра вперёд.
// tilt_deg > 0 — наклон вперёд по курсу.
inline Mat3 R_body_from_cam(double tilt_deg = 0.0, double pan_deg = 0.0) {
    Mat3 R0;                       // x_cam->y_body, y_cam->-x_body, z_cam->z_body
    R0.m[0][0]=0; R0.m[0][1]=-1; R0.m[0][2]=0;
    R0.m[1][0]=1; R0.m[1][1]= 0; R0.m[1][2]=0;
    R0.m[2][0]=0; R0.m[2][1]= 0; R0.m[2][2]=1;

    const double t = tilt_deg * M_PI / 180.0;
    Mat3 Rt; Rt.m[0][0]=std::cos(t); Rt.m[0][2]=std::sin(t);
             Rt.m[1][1]=1;
             Rt.m[2][0]=-std::sin(t); Rt.m[2][2]=std::cos(t);

    const double p = pan_deg * M_PI / 180.0;
    Mat3 Rp; Rp.m[0][0]=std::cos(p); Rp.m[0][1]=-std::sin(p);
             Rp.m[1][0]=std::sin(p); Rp.m[1][1]= std::cos(p);
             Rp.m[2][2]=1;
    return Rp * Rt * R0;
}

struct GeoPoint {
    double lat = 0, lon = 0;
    double slant_range_m = 0;    // наклонная дальность до точки
};

// Главная функция. nullopt, если луч уходит выше горизонта.
inline std::optional<GeoPoint> pixel_to_ground(double u, double v,
                                               const Pose& pose,
                                               const CameraIntrinsics& cam,
                                               const Mat3& R_mount) {
    const Vec3 d_cam  = cam.ray(u, v);
    const Vec3 d_body = R_mount * d_cam;
    const Vec3 d_ned  = R_ned_from_body(pose.roll, pose.pitch, pose.yaw) * d_body;

    if (d_ned.z <= 1e-6) return std::nullopt;     // не смотрит вниз — земли не встретит

    const double h = pose.alt_for_geo();
    const double t = h / d_ned.z;
    const double north = t * d_ned.x;
    const double east  = t * d_ned.y;

    GeoPoint g;
    g.lat = pose.lat + (north / R_EARTH) * 180.0 / M_PI;
    g.lon = pose.lon + (east / (R_EARTH * std::cos(pose.lat * M_PI / 180.0))) * 180.0 / M_PI;
    g.slant_range_m = t;
    return g;
}

inline double ground_distance(double lat1, double lon1, double lat2, double lon2) {
    const double dn = (lat2 - lat1) * M_PI / 180.0 * R_EARTH;
    const double de = (lon2 - lon1) * M_PI / 180.0 * R_EARTH * std::cos(lat1 * M_PI / 180.0);
    return std::sqrt(dn * dn + de * de);
}

} // namespace sar
