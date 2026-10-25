"""
实验四 进阶：为 3D 查看器中加载的模型增加光线追踪
=========================================================
CPU 上建 BVH，节点和三角形上传到纹理缓冲，片元着色器里遍历 BVH 求交
（Möller–Trumbore），场景是一个带墙壁的房间：左红墙、右绿墙、白后墙、
石板地面、天花板中间的矩形面光源，带软阴影、反射、折射。

功能：
    - 材质：漫反射 / 镜面 / 玻璃 / 金属
    - 参数：反射次数、阴影采样、超采样、光源强度、曝光、视场角
    - 视角：正视 / 俯视 / 等轴测 / 复位（1 / 2 / 3 / R）
    - 选中与移动：单击模型选中（橙色高亮），WASD 平移、Q/E 升降，
      移动会同步写回查看器里的模型位置

操作：
    鼠标拖动旋转（很慢的速率，0.26 度/像素），滚轮缩放
    单击模型 = 选中，WASD/QE = 移动选中的模型
=========================================================
"""
import math

import numpy as np
from PyQt5.QtCore import QObject, Qt, QTimer
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                             QOpenGLWidget, QPushButton, QSlider, QVBoxLayout,
                             QWidget)

from OpenGL.GL import *

import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from plugins.base_plugin import Base3DPlugin


# ------------------------------------------------------------------ 着色器
VS_SRC = """
#version 330 core
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

FS_SRC = """
#version 330 core

uniform samplerBuffer uNodes;
uniform samplerBuffer uTris;
uniform vec2  uRes;
uniform vec3  uCamPos;
uniform vec3  uCamTarget;
uniform float uFov;
uniform int   uMat;              // 0 漫反射 1 镜面 2 玻璃 3 金属
uniform int   uRefl;
uniform int   uShadows;
uniform int   uShadowSamples;
uniform int   uAA;
uniform float uLightPower;
uniform float uExposure;
uniform int   uSel;              // 选中的模型编号，-1 表示没有

uniform vec3  uRoomMin;
uniform vec3  uRoomMax;
uniform vec3  uLightC;
uniform vec2  uLightHalf;

out vec4 fragColor;

struct Hit {
    float t;
    vec3  p;
    vec3  n;
    int   type;      // 1 模型  2 房间
    int   emissive;
    int   id;        // 模型编号
};

// ---------------------------------------------------------------- BVH
float hitTri(int i, vec3 ro, vec3 rd, out vec3 nOut, out int idOut) {
    vec4 v0 = texelFetch(uTris, i * 6 + 0);
    vec4 v1 = texelFetch(uTris, i * 6 + 1);
    vec4 v2 = texelFetch(uTris, i * 6 + 2);
    vec3 e1 = v1.xyz - v0.xyz;
    vec3 e2 = v2.xyz - v0.xyz;
    vec3 pv = cross(rd, e2);
    float det = dot(e1, pv);
    if (abs(det) < 1e-9) return 1e30;
    float inv = 1.0 / det;
    vec3 tv = ro - v0.xyz;
    float u = dot(tv, pv) * inv;
    if (u < 0.0 || u > 1.0) return 1e30;
    vec3 qv = cross(tv, e1);
    float v = dot(rd, qv) * inv;
    if (v < 0.0 || u + v > 1.0) return 1e30;
    float t = dot(e2, qv) * inv;
    if (t <= 1e-4) return 1e30;
    vec4 n0 = texelFetch(uTris, i * 6 + 3);
    vec3 n1 = texelFetch(uTris, i * 6 + 4).xyz;
    vec3 n2 = texelFetch(uTris, i * 6 + 5).xyz;
    vec3 n = n0.xyz * (1.0 - u - v) + n1 * u + n2 * v;
    if (dot(n, n) < 1e-12) n = cross(e1, e2);
    nOut = normalize(n);
    idOut = int(n0.w + 0.5);
    return t;
}

bool hitModel(vec3 ro, vec3 rd, out Hit h) {
    float best = 1e30;
    bool found = false;
    int stack[32];
    int sp = 0;
    stack[sp++] = 0;
    vec3 inv = 1.0 / rd;

    while (sp > 0) {
        int ni = stack[--sp];
        vec4 a = texelFetch(uNodes, ni * 2 + 0);
        vec4 b = texelFetch(uNodes, ni * 2 + 1);

        vec3 t0 = (a.xyz - ro) * inv;
        vec3 t1 = (b.xyz - ro) * inv;
        vec3 tmin = min(t0, t1);
        vec3 tmaxv = max(t0, t1);
        float tn = max(max(tmin.x, tmin.y), tmin.z);
        float tf = min(min(tmaxv.x, tmaxv.y), tmaxv.z);
        if (tn > tf || tf < 1e-4 || tn > best) continue;

        int count = int(b.w + 0.5);
        if (count > 0) {
            int start = int(a.w + 0.5);
            for (int k = 0; k < count; ++k) {
                vec3 n;
                int id;
                float t = hitTri(start + k, ro, rd, n, id);
                if (t < best) {
                    best = t;
                    h.t = t;
                    h.p = ro + rd * t;
                    h.n = (dot(n, rd) > 0.0) ? -n : n;
                    h.type = 1;
                    h.emissive = 0;
                    h.id = id;
                    found = true;
                }
            }
        } else {
            int left = int(a.w + 0.5);
            stack[sp++] = left;
            stack[sp++] = left + 1;
        }
    }
    return found;
}

bool occluded(vec3 ro, vec3 rd, float maxt) {
    int stack[32];
    int sp = 0;
    stack[sp++] = 0;
    vec3 inv = 1.0 / rd;

    while (sp > 0) {
        int ni = stack[--sp];
        vec4 a = texelFetch(uNodes, ni * 2 + 0);
        vec4 b = texelFetch(uNodes, ni * 2 + 1);

        vec3 t0 = (a.xyz - ro) * inv;
        vec3 t1 = (b.xyz - ro) * inv;
        vec3 tmin = min(t0, t1);
        vec3 tmaxv = max(t0, t1);
        float tn = max(max(tmin.x, tmin.y), tmin.z);
        float tf = min(min(tmaxv.x, tmaxv.y), tmaxv.z);
        if (tn > tf || tf < 1e-4 || tn > maxt) continue;

        int count = int(b.w + 0.5);
        if (count > 0) {
            int start = int(a.w + 0.5);
            for (int k = 0; k < count; ++k) {
                vec3 n;
                int id;
                float t = hitTri(start + k, ro, rd, n, id);
                if (t < maxt) return true;
            }
        } else {
            int left = int(a.w + 0.5);
            stack[sp++] = left;
            stack[sp++] = left + 1;
        }
    }
    return false;
}

// ---------------------------------------------------------------- 房间
bool hitRoom(vec3 ro, vec3 rd, out Hit h) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (uRoomMin - ro) * inv;
    vec3 t1 = (uRoomMax - ro) * inv;
    vec3 tmaxv = max(t0, t1);
    float tf = min(min(tmaxv.x, tmaxv.y), tmaxv.z);
    if (tf < 1e-4) return false;

    h.t = tf;
    h.p = ro + rd * tf;
    h.type = 2;
    if (tf == tmaxv.x)      h.n = vec3(-sign(rd.x), 0.0, 0.0);
    else if (tf == tmaxv.y) h.n = vec3(0.0, -sign(rd.y), 0.0);
    else                    h.n = vec3(0.0, 0.0, -sign(rd.z));

    h.emissive = 0;
    h.id = -1;
    if (h.n.y < -0.5 &&
        abs(h.p.x - uLightC.x) <= uLightHalf.x &&
        abs(h.p.z - uLightC.z) <= uLightHalf.y) {
        h.emissive = 1;
    }
    return true;
}

vec3 wallAlbedo(Hit h) {
    vec3 n = h.n;
    if (n.y > 0.5) {
        float c = mod(floor(h.p.x * 1.5) + floor(h.p.z * 1.5), 2.0);
        return mix(vec3(0.42, 0.41, 0.40), vec3(0.62, 0.61, 0.59), c);
    }
    if (n.y < -0.5) return vec3(0.66);
    if (n.x > 0.5)  return vec3(0.70, 0.10, 0.10);
    if (n.x < -0.5) return vec3(0.10, 0.55, 0.14);
    if (n.z > 0.5)  return vec3(0.88);
    return vec3(0.38);
}

bool traceScene(vec3 ro, vec3 rd, out Hit h) {
    Hit hm;
    Hit hr;
    bool a = hitModel(ro, rd, hm);
    bool b = hitRoom(ro, rd, hr);
    if (a && (!b || hm.t < hr.t)) { h = hm; return true; }
    if (b) { h = hr; return true; }
    return false;
}

// ---------------------------------------------------------------- 着色
vec3 shadeDiffuse(vec3 p, vec3 n, vec3 albedo, vec2 seed) {
    vec3 col = albedo * 0.12;

    int nS = uShadowSamples;
    float sum = 0.0;
    for (int i = 0; i < 32; ++i) {
        if (i >= nS) break;
        // 分层采样：u 均匀分 N 层，v 用黄金比例错开
        float su = (float(i) + 0.5) / float(nS);
        float sv = fract(0.5 + float(i) * 0.6180339887 + seed.x * 0.0137);
        vec3 lp = vec3(uLightC.x + (su * 2.0 - 1.0) * uLightHalf.x,
                       uLightC.y,
                       uLightC.z + (sv * 2.0 - 1.0) * uLightHalf.y);
        vec3 L = lp - p;
        float d2 = dot(L, L);
        float d = sqrt(d2);
        L /= d;
        float ndl = dot(n, L);
        if (ndl <= 0.0) continue;
        if (uShadows == 1 && occluded(p + n * 5e-3, L, d - 1e-2)) continue;
        sum += ndl / (1.0 + 0.10 * d2);
    }
    col += albedo * sum * (uLightPower * 5.0 / float(nS));
    return col;
}

float fresnel(vec3 nGeo, vec3 rd, float ior) {
    bool exiting = dot(rd, nGeo) > 0.0;
    vec3 n = exiting ? -nGeo : nGeo;
    float etai = exiting ? ior : 1.0;
    float etat = exiting ? 1.0 : ior;
    float cosi = clamp(dot(-rd, n), 0.0, 1.0);
    float sint = etai / etat * sqrt(max(0.0, 1.0 - cosi * cosi));
    if (sint >= 1.0) return 1.0;
    float cost = sqrt(max(0.0, 1.0 - sint * sint));
    float r0 = (etai - etat) / (etai + etat);
    r0 *= r0;
    float c = 1.0 - (exiting ? cost : cosi);
    return r0 + (1.0 - r0) * pow(c, 5.0);
}

vec3 modelAlbedo(int id) {
    if (uMat == 2) return vec3(0.94, 0.96, 0.99);
    if (uMat == 3) return vec3(0.96, 0.82, 0.56);
    vec3 base = vec3(0.80, 0.80, 0.82);
    if (id == uSel) base = mix(base, vec3(1.0, 0.52, 0.14), 0.55);
    return base;
}

vec3 trace(vec3 ro, vec3 rd, vec2 seed) {
    vec3 col = vec3(0.0);
    vec3 thr = vec3(1.0);
    bool escaped = false;

    for (int b = 0; b < 6; ++b) {
        if (b > uRefl) break;
        Hit h;
        if (!traceScene(ro, rd, h)) {
            col += thr * vec3(0.02, 0.02, 0.03);
            escaped = true;
            break;
        }

        if (h.type == 2) {
            if (h.emissive == 1) {
                col += thr * vec3(7.0);
                break;
            }
            col += thr * shadeDiffuse(h.p, h.n, wallAlbedo(h),
                                      seed + vec2(float(b) * 13.0, 0.0));
            break;
        }

        vec3 albedo = modelAlbedo(h.id);
        if (uMat == 0) {
            col += thr * shadeDiffuse(h.p, h.n, albedo, seed + vec2(float(b) * 13.0, 0.0));
            break;
        } else if (uMat == 1) {
            col += thr * albedo * 0.06;
            rd = reflect(rd, h.n);
            thr *= albedo;
        } else if (uMat == 3) {
            col += thr * albedo * 0.08;
            rd = reflect(rd, h.n);
            thr *= albedo * vec3(0.96, 0.90, 0.72);
        } else {
            bool inside = dot(rd, h.n) > 0.0;
            vec3 n = inside ? -h.n : h.n;
            float f = fresnel(h.n, rd, 1.5);
            vec3 refl = reflect(rd, h.n);
            vec3 refr = refract(rd, n, inside ? 1.5 : 1.0 / 1.5);
            col += thr * albedo * 0.04 * f;
            if (dot(refr, refr) < 1e-6) {
                rd = refl;
            } else {
                rd = (f > 0.5) ? refl : refr;
                thr *= mix(vec3(1.0), albedo, 0.2);
            }
        }
        ro = h.p + rd * 2e-3;
    }

    if (!escaped) {
        col += thr * vec3(0.04, 0.04, 0.05);
    }
    return col;
}

void main() {
    vec3 forward = normalize(uCamTarget - uCamPos);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, forward);
    float scale = tan(radians(uFov) * 0.5);

    vec3 col = vec3(0.0);
    int n = uAA;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            if (x >= n || y >= n) continue;
            vec2 jitter = (vec2(float(x), float(y)) + 0.5) / float(n);
            vec2 pix = gl_FragCoord.xy + jitter - 0.5;
            vec2 ndc = (pix * 2.0 - uRes) / uRes.y;
            vec3 rd = normalize(forward + right * ndc.x * scale + up * ndc.y * scale);
            vec2 seed = gl_FragCoord.xy + vec2(float(x), float(y)) * 41.0;
            col += trace(uCamPos, rd, seed);
        }
    }
    col /= float(n * n);

    col *= uExposure;
    col = col / (col + vec3(1.0));
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
    fragColor = vec4(col, 1.0);
}
"""


def compile_shader(src, kind):
    shader = glCreateShader(kind)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader


def build_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


# ------------------------------------------------------------------ BVH
def build_bvh(tris, leaf_size=8):
    tri_min = tris.min(axis=1)
    tri_max = tris.max(axis=1)
    cent = (tri_min + tri_max) * 0.5

    nodes = []
    order = []

    def fill(slot, idx):
        lo = tri_min[idx].min(axis=0)
        hi = tri_max[idx].max(axis=0)
        nodes[slot] = [lo[0], lo[1], lo[2], 0.0, hi[0], hi[1], hi[2], 0.0]

        if len(idx) <= leaf_size:
            start = len(order)
            order.extend(int(t) for t in idx)
            nodes[slot][3] = start
            nodes[slot][7] = len(idx)
            return

        axis = int(np.argmax(hi - lo))
        values = cent[idx, axis]
        mid = len(idx) // 2
        part = np.argpartition(values, mid)
        left_idx = idx[part[:mid]]
        right_idx = idx[part[mid:]]

        left = len(nodes)
        nodes.append(None)
        nodes.append(None)
        nodes[slot][3] = left
        fill(left, left_idx)
        fill(left + 1, right_idx)

    nodes.append(None)
    fill(0, np.arange(len(tris)))
    data = np.array(nodes, dtype=np.float32).reshape(-1, 2, 4)
    return data.reshape(-1, 4), np.array(order, dtype=np.int64)


# ------------------------------------------------------------------ 收集场景
def collect_scene(ctx, max_tris=20000):
    chunks = []
    model_ids = []
    models = []
    total = 0
    lo = np.array([1e30, 1e30, 1e30], dtype=np.float32)
    hi = np.array([-1e30, -1e30, -1e30], dtype=np.float32)

    for model in getattr(ctx, "models", []):
        polydata = model.get("polydata")
        if polydata is None:
            continue

        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(polydata)
        tri.Update()
        data = tri.GetOutput()

        if data.GetNumberOfPolys() > 8000:
            dec = vtk.vtkQuadricDecimation()
            dec.SetInputData(data)
            dec.SetTargetReduction(0.9)
            dec.Update()
            data = dec.GetOutput()

        pts = vtk_to_numpy(data.GetPoints().GetData()).astype(np.float32)
        if pts.shape[0] < 3:
            continue
        polys = vtk_to_numpy(data.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        nrm = data.GetPointData().GetNormals()
        normals = vtk_to_numpy(nrm).astype(np.float32) if nrm is not None else None

        mat = model["actor"].GetMatrix()
        rot = np.array([[mat.GetElement(r, c) for c in range(3)] for r in range(3)],
                       dtype=np.float32)
        trans = np.array([mat.GetElement(r, 3) for r in range(3)], dtype=np.float32)
        if not np.any(rot):
            rot = np.eye(3, dtype=np.float32)

        world = pts @ rot.T + trans
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))

        if normals is not None:
            wn = normals @ rot.T
            wn /= np.maximum(np.linalg.norm(wn, axis=1, keepdims=True), 1e-8)
            tri_n = wn[polys]
        else:
            face = np.cross(world[polys[:, 1]] - world[polys[:, 0]],
                            world[polys[:, 2]] - world[polys[:, 0]])
            face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-8)
            tri_n = np.repeat(face[:, None, :], 3, axis=1)

        tri_v = world[polys]
        count = len(polys)
        if total + count > max_tris:
            keep = max(1, int(count * (max_tris - total) / max(count, 1)))
            idx = np.linspace(0, count - 1, keep).astype(np.int64)
            tri_v, tri_n = tri_v[idx], tri_n[idx]
            count = keep

        mid = len(models)
        models.append(model)
        model_ids.append(np.full(count, mid, dtype=np.float32))
        total += count

        chunks.append(np.concatenate([tri_v, tri_n], axis=1).reshape(-1, 3))
        if total >= max_tris:
            break

    if not chunks:
        return None

    verts = np.concatenate(chunks, axis=0).astype(np.float32)
    ids = np.concatenate(model_ids, axis=0)
    tris = verts.reshape(-1, 6, 3)

    node_data, order = build_bvh(tris[:, :3, :])
    tris = tris[order]
    ids = ids[order]

    tri_data = np.ones((len(tris), 6, 4), dtype=np.float32)
    tri_data[:, :, :3] = tris
    tri_data[:, 3:, 3] = 0.0
    tri_data[:, 3, 3] = ids          # 法线 texel 的 w 存模型编号

    center = (lo + hi) * 0.5
    size = hi - lo
    half_w = max(float(size[0]), float(size[2]), 2.0) * 0.55 + 0.9
    half_d = half_w * 1.55
    height = max(float(size[1]) * 1.25 + 0.9, 2.6)

    floor_y = float(lo[1]) - 0.01
    room_min = [center[0] - half_w, floor_y, center[2] - half_d]
    room_max = [center[0] + half_w, floor_y + height, center[2] + half_d]
    light_c = [center[0], floor_y + height - 0.02, center[2] + half_d * 0.15]
    light_half = [half_w * 0.34, half_d * 0.24]

    return {
        "tris": tri_data.reshape(-1, 4),
        "nodes": node_data.reshape(-1, 4),
        "count": len(tris),
        "model_ids": ids,
        "models": models,
        "center": center,
        "size": size,
        "room_min": np.array(room_min, dtype=np.float32),
        "room_max": np.array(room_max, dtype=np.float32),
        "light_c": np.array(light_c, dtype=np.float32),
        "light_half": np.array(light_half, dtype=np.float32),
        "half_d": half_d,
        "tris_xyz": tris[:, :3, :],
    }


MAT_NAMES = ["漫反射", "镜面", "玻璃", "金属"]


class _MenuRelay(QObject):
    """插件本身不是 QObject，用这个把“视图”菜单的 triggered 转出去"""

    def __init__(self, plugin, action):
        super().__init__(plugin.ctx.main_window)
        self.plugin = plugin
        self.action = action

    def handle(self, checked=False):
        self.plugin._on_view_action(self.action)


# ------------------------------------------------------------------ 覆盖层
class RayTraceView(QOpenGLWidget):
    """独立的 OpenGL 上下文，盖在 VTK 视口上"""

    def __init__(self, parent, scene, move_cb=None, select_cb=None):
        super().__init__(parent)

        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.scene = scene
        self.move_cb = move_cb
        self.select_cb = select_cb
        self.tri_count = scene["count"]
        self.selected_model = -1

        self.center = scene["center"].copy()
        radius = float(np.linalg.norm(scene["size"])) * 0.5
        self.distance = min(float(scene["half_d"]) * 0.78, max(radius * 2.7, 1.6))
        self.yaw = 0.35
        self.pitch = 0.14
        self.fov = 50.0

        self.material = 0
        self.refl = 1
        self.shadows = True
        self.shadow_samples = 16
        self.aa = 2
        self.light_power = 1.0
        self.exposure = 1.0

        self.program = 0
        self.vao = 0
        self.node_tbo = self.node_tex = 0
        self.tri_tbo = self.tri_tex = 0
        self.loc = {}
        self.last_ms = 0.0
        self.dirty = True
        self.dragging = False
        self.drag_moved = 0

        self.hud = QLabel(self)
        self.hud.setStyleSheet("color:#cfe3ff; background:rgba(10,16,28,170);"
                               "padding:4px 8px; border-radius:4px;")
        self.hud.move(10, 10)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(120)

    # ---------------------------------------------------------- GL
    def initializeGL(self):
        self.program = build_program(VS_SRC, FS_SRC)
        for name in ("uNodes", "uTris", "uRes", "uCamPos", "uCamTarget", "uFov",
                     "uMat", "uRefl", "uShadows", "uShadowSamples", "uAA",
                     "uLightPower", "uExposure", "uSel",
                     "uRoomMin", "uRoomMax", "uLightC", "uLightHalf"):
            self.loc[name] = glGetUniformLocation(self.program, name)
        self.vao = glGenVertexArrays(1)
        self._upload()

    def _upload(self):
        tris = self.scene["tris"].astype(np.float32)
        nodes = self.scene["nodes"].astype(np.float32)

        if not self.tri_tbo:
            self.tri_tbo = glGenBuffers(1)
            self.tri_tex = glGenTextures(1)
        glBindBuffer(GL_TEXTURE_BUFFER, self.tri_tbo)
        glBufferData(GL_TEXTURE_BUFFER, tris.nbytes, tris, GL_STATIC_DRAW)
        glBindTexture(GL_TEXTURE_BUFFER, self.tri_tex)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, self.tri_tbo)

        if not self.node_tbo:
            self.node_tbo = glGenBuffers(1)
            self.node_tex = glGenTextures(1)
        glBindBuffer(GL_TEXTURE_BUFFER, self.node_tbo)
        glBufferData(GL_TEXTURE_BUFFER, nodes.nbytes, nodes, GL_STATIC_DRAW)
        glBindTexture(GL_TEXTURE_BUFFER, self.node_tex)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, self.node_tbo)

    def update_scene(self, scene):
        """模型移动后重建 BVH 并重新上传"""
        self.scene = scene
        self.tri_count = scene["count"]
        self.makeCurrent()
        self._upload()
        self.doneCurrent()
        self.dirty = True

    def resizeGL(self, w, h):
        pass

    def paintGL(self):
        import time
        dpr = self.devicePixelRatioF()
        w = max(1, int(round(self.width() * dpr)))
        h = max(1, int(round(self.height() * dpr)))
        glViewport(0, 0, w, h)

        cam = self.camera_position()
        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, self.node_tex)
        glUniform1i(self.loc["uNodes"], 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_BUFFER, self.tri_tex)
        glUniform1i(self.loc["uTris"], 1)

        glUniform2f(self.loc["uRes"], float(w), float(h))
        glUniform3f(self.loc["uCamPos"], *cam)
        glUniform3f(self.loc["uCamTarget"], *self.center)
        glUniform1f(self.loc["uFov"], self.fov)
        glUniform1i(self.loc["uMat"], self.material)
        glUniform1i(self.loc["uRefl"], self.refl)
        glUniform1i(self.loc["uShadows"], 1 if self.shadows else 0)
        glUniform1i(self.loc["uShadowSamples"], self.shadow_samples)
        glUniform1i(self.loc["uAA"], 1 if self.dragging else self.aa)
        glUniform1f(self.loc["uLightPower"], self.light_power)
        glUniform1f(self.loc["uExposure"], self.exposure)
        glUniform1i(self.loc["uSel"], self.selected_model)
        glUniform3f(self.loc["uRoomMin"], *self.scene["room_min"])
        glUniform3f(self.loc["uRoomMax"], *self.scene["room_max"])
        glUniform3f(self.loc["uLightC"], *self.scene["light_c"])
        glUniform2f(self.loc["uLightHalf"], *self.scene["light_half"])

        t0 = time.perf_counter()
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glFinish()
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.dirty = False

        name = ""
        if 0 <= self.selected_model < len(self.scene["models"]):
            path = self.scene["models"][self.selected_model].get("path", "")
            name = " | 选中 " + path.replace("\\", "/").split("/")[-1]
        self.hud.setText(f"光追 | 三角 {self.tri_count} | {MAT_NAMES[self.material]}"
                         f" | 阴影 {self.shadow_samples} | {self.last_ms:.0f} ms{name}")

    # ---------------------------------------------------------- 交互
    def camera_position(self):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        return self.center + np.array([self.distance * cp * sy,
                                       self.distance * sp,
                                       self.distance * cp * cy], dtype=np.float32)

    def _ray_from_pixel(self, x, y):
        dpr = self.devicePixelRatioF()
        w = max(1.0, self.width() * dpr)
        h = max(1.0, self.height() * dpr)
        px = x * dpr
        py = (self.height() - y) * dpr
        ndc = np.array([(px * 2.0 - w) / h, (py * 2.0 - h) / h], dtype=np.float32)

        cam = self.camera_position()
        forward = self.center - cam
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        scale = math.tan(math.radians(self.fov) * 0.5)
        d = forward + right * ndc[0] * scale + up * ndc[1] * scale
        d /= np.linalg.norm(d)
        return cam, d

    def pick(self, x, y):
        """CPU 上用同一批三角形做一次拾取，返回模型编号"""
        cam, d = self._ray_from_pixel(x, y)
        tris = self.scene["tris_xyz"].astype(np.float64)
        v0 = tris[:, 0]
        e1 = tris[:, 1] - v0
        e2 = tris[:, 2] - v0

        pv = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, pv)
        ok = np.abs(det) > 1e-9
        inv = np.zeros_like(det)
        inv[ok] = 1.0 / det[ok]

        tv = cam - v0
        u = np.einsum("ij,ij->i", tv, pv) * inv
        qv = np.cross(tv, e1)
        v = (qv @ d) * inv
        tt = np.einsum("ij,ij->i", e2, qv) * inv

        mask = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (tt > 1e-4)
        if not np.any(mask):
            return -1
        idx = int(np.argmin(np.where(mask, tt, 1e30)))
        return int(self.scene["model_ids"][idx])

    def mousePressEvent(self, event):
        self.last_pos = event.pos()
        self.dragging = True
        self.drag_moved = 0

    def mouseReleaseEvent(self, event):
        self.dragging = False
        if self.drag_moved < 4:
            hit = self.pick(event.x(), event.y())
            self.selected_model = hit
            if self.select_cb is not None:
                self.select_cb(hit)
            self.dirty = True
        self.update()

    def mouseMoveEvent(self, event):
        if not self.dragging:
            return
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        self.last_pos = event.pos()
        self.drag_moved += abs(dx) + abs(dy)
        # 0.26 度/像素
        self.yaw -= dx * 0.0045
        self.pitch = max(-0.35, min(1.45, self.pitch - dy * 0.0035))
        self.dirty = True

    def wheelEvent(self, event):
        step = 0.97 if event.angleDelta().y() > 0 else 1.03
        self.distance = max(0.25, min(float(self.scene["half_d"]) * 0.92,
                                     self.distance * step))
        self.dirty = True

    def keyPressEvent(self, event):
        key = event.key()
        step = 0.12
        if key in (Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D, Qt.Key_Q, Qt.Key_E):
            if self.selected_model < 0:
                return
            delta = {"W": (0, 0, -step), "S": (0, 0, step),
                     "A": (-step, 0, 0), "D": (step, 0, 0),
                     "Q": (0, -step, 0), "E": (0, step, 0)}[
                {Qt.Key_W: "W", Qt.Key_S: "S", Qt.Key_A: "A",
                 Qt.Key_D: "D", Qt.Key_Q: "Q", Qt.Key_E: "E"}[key]]
            if self.move_cb is not None:
                self.move_cb(*delta)
            return
        super().keyPressEvent(event)

    def set_view(self, kind):
        base = min(float(self.scene["half_d"]) * 0.78,
                   max(float(np.linalg.norm(self.scene["size"])) * 1.35, 1.6))
        if kind == "front":
            self.yaw, self.pitch, self.distance = 0.0, 0.02, base
        elif kind == "top":
            self.yaw, self.pitch, self.distance = 0.0, 1.42, base * 1.25
        elif kind == "iso":
            self.yaw, self.pitch, self.distance = 0.68, 0.55, base * 1.1
        else:
            self.yaw, self.pitch = 0.35, 0.14
            self.distance = base
        self.dirty = True

    def _tick(self):
        if self.dirty:
            self.update()
        self.hud.adjustSize()


# ------------------------------------------------------------------ 插件
class Exp4RayTracePlugin(Base3DPlugin):
    plugin_cls_name = "exp4_raytrace"
    plugin_name = "实验四·光线追踪预览"
    plugin_desc = "BVH + 片元着色器光线追踪：房间、软阴影、反射折射、选中移动"

    def __init__(self, viewer_ctx):
        super().__init__(viewer_ctx)
        self.view = None
        self.sync_timer = None
        self.rebuild_timer = None

    # ---------------------------------------------------------- UI
    def create_ui(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(3)

        tip = QLabel("房间 + 天花板面光源，BVH 光线追踪。单击模型选中，WASD/QE 移动，"
                     "1/2/3 换视角。")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        self.btn_toggle = QPushButton("开启光线追踪预览")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.toggled.connect(self._toggle)
        layout.addWidget(self.btn_toggle)

        row = QHBoxLayout()
        row.addWidget(QLabel("材质"))
        self.cmb_mat = QComboBox()
        self.cmb_mat.addItems(MAT_NAMES)
        self.cmb_mat.currentIndexChanged.connect(self._params_changed)
        row.addWidget(self.cmb_mat)

        row.addWidget(QLabel("反射"))
        self.cmb_refl = QComboBox()
        self.cmb_refl.addItems(["0", "1", "2", "3"])
        self.cmb_refl.setCurrentIndex(1)
        self.cmb_refl.currentIndexChanged.connect(self._params_changed)
        row.addWidget(self.cmb_refl)

        row.addWidget(QLabel("阴影"))
        self.cmb_shadow = QComboBox()
        self.cmb_shadow.addItems(["4", "8", "16", "32"])
        self.cmb_shadow.setCurrentIndex(2)
        self.cmb_shadow.currentIndexChanged.connect(self._params_changed)
        row.addWidget(self.cmb_shadow)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        self.chk_shadow = QCheckBox("软阴影")
        self.chk_shadow.setChecked(True)
        self.chk_shadow.toggled.connect(self._params_changed)
        row2.addWidget(self.chk_shadow)

        row2.addWidget(QLabel("超采样"))
        self.cmb_aa = QComboBox()
        self.cmb_aa.addItems(["1x", "2x"])
        self.cmb_aa.setCurrentIndex(1)
        self.cmb_aa.currentIndexChanged.connect(self._params_changed)
        row2.addWidget(self.cmb_aa)
        layout.addLayout(row2)

        layout.addWidget(QLabel("光源强度"))
        self.sld_light = QSlider(Qt.Horizontal)
        self.sld_light.setRange(20, 300)
        self.sld_light.setValue(100)
        self.sld_light.valueChanged.connect(self._params_changed)
        layout.addWidget(self.sld_light)

        layout.addWidget(QLabel("曝光"))
        self.sld_exposure = QSlider(Qt.Horizontal)
        self.sld_exposure.setRange(40, 250)
        self.sld_exposure.setValue(100)
        self.sld_exposure.valueChanged.connect(self._params_changed)
        layout.addWidget(self.sld_exposure)

        layout.addWidget(QLabel("视场角"))
        self.sld_fov = QSlider(Qt.Horizontal)
        self.sld_fov.setRange(25, 80)
        self.sld_fov.setValue(50)
        self.sld_fov.valueChanged.connect(self._params_changed)
        layout.addWidget(self.sld_fov)

        self.lbl_state = QLabel("未开启")
        self.lbl_state.setWordWrap(True)
        layout.addWidget(self.lbl_state)
        return panel

    def _params_changed(self):
        if self.view is None:
            return
        self.view.material = self.cmb_mat.currentIndex()
        self.view.refl = self.cmb_refl.currentIndex()
        self.view.shadow_samples = int(self.cmb_shadow.currentText())
        self.view.shadows = self.chk_shadow.isChecked()
        self.view.aa = self.cmb_aa.currentIndex() + 1
        self.view.light_power = self.sld_light.value() / 100.0
        self.view.exposure = self.sld_exposure.value() / 100.0
        self.view.fov = float(self.sld_fov.value())
        self.view.dirty = True

    # ---------------------------------------------------------- 开关
    def _toggle(self, checked):
        if checked:
            self._open()
        else:
            self._close()

    def _open(self):
        self.ctx.vtk_widget.GetRenderWindow().Render()
        scene = collect_scene(self.ctx)
        if scene is None:
            self.ctx.info_display.append("[光追] 场景里还没有模型，先加载一个 OBJ")
            self.btn_toggle.setChecked(False)
            return

        host = self.ctx.vtk_widget
        self.view = RayTraceView(host, scene, move_cb=self._move_selected,
                                 select_cb=self._on_picked)
        self.view.setGeometry(0, 0, host.width(), host.height())
        self.view.show()
        self.view.raise_()
        self.view.setFocus()

        if self.sync_timer is None:
            self.sync_timer = QTimer(self.ctx.main_window)
            self.sync_timer.timeout.connect(self._sync_geometry)
        self.sync_timer.start(150)

        self._hook_framework()
        # 打开时同步一下场景面板当前选中的模型
        self._on_list_changed(self.ctx.main_window.model_list.currentRow())

        self._params_changed()
        self.ctx.info_display.append(
            f"[光追] 已开启：三角形 {scene['count']}，单击模型可选中，WASD/QE 移动")
        self.lbl_state.setText("已开启：单击选中，WASD/QE 移动，1/2/3 视角")
        self.btn_toggle.setText("退出光线追踪预览")

    # ---------------------------------------------------------- 复用框架已有功能
    def _hook_framework(self):
        """把框架“视图”菜单和场景面板接到光追画面上，避免插件里再重复一套"""
        win = self.ctx.main_window
        self._hooked_actions = []
        self._hooked_spins = []
        self._hooked_buttons = []

        for menu_action in win.menuBar().actions():
            menu = menu_action.menu()
            if menu is None or "视图" not in menu_action.text():
                continue
            for act in menu.actions():
                relay = _MenuRelay(self, act)
                act.triggered.connect(relay.handle)
                self._hooked_actions.append((act, relay))

        for spin in (win.spin_x, win.spin_y, win.spin_z):
            spin.valueChanged.connect(self._on_scene_changed)
            self._hooked_spins.append(spin)

        win.model_list.currentRowChanged.connect(self._on_list_changed)

        from PyQt5.QtWidgets import QPushButton
        for btn in win.findChildren(QPushButton):
            if btn.text() in ("自动排列", "删除选中", "清空"):
                btn.clicked.connect(self._on_scene_changed)
                self._hooked_buttons.append(btn)

    def _unhook_framework(self):
        for act, relay in getattr(self, "_hooked_actions", []):
            try:
                act.triggered.disconnect(relay.handle)
            except TypeError:
                pass
        for spin in getattr(self, "_hooked_spins", []):
            try:
                spin.valueChanged.disconnect(self._on_scene_changed)
            except TypeError:
                pass
        for btn in getattr(self, "_hooked_buttons", []):
            try:
                btn.clicked.disconnect(self._on_scene_changed)
            except TypeError:
                pass
        try:
            self.ctx.main_window.model_list.currentRowChanged.disconnect(self._on_list_changed)
        except TypeError:
            pass
        self._hooked_actions = []
        self._hooked_spins = []
        self._hooked_buttons = []

    def _on_view_action(self, action):
        if self.view is None:
            return
        text = action.text().replace("&", "") if action is not None else ""
        if "正视" in text:
            self.view.set_view("front")
        elif "俯视" in text:
            self.view.set_view("top")
        elif "等轴测" in text:
            self.view.set_view("iso")
        elif "适配" in text or "重置" in text:
            self.view.set_view("reset")
        elif "自动摆正" in text or "统一模型尺寸" in text:
            self._schedule_rebuild()
        else:
            return
        self.view.dirty = True

    def _on_scene_changed(self, *_args):
        self._schedule_rebuild()

    def _on_list_changed(self, row):
        if self.view is None or row is None or row < 0:
            return
        scene = self.view.scene
        if row < len(self.ctx.models):
            model = self.ctx.models[row]
            try:
                self.view.selected_model = scene["models"].index(model)
            except ValueError:
                self.view.selected_model = -1
            self.view.dirty = True

    def _schedule_rebuild(self):
        if self.view is None:
            return
        if self.rebuild_timer is None:
            self.rebuild_timer = QTimer(self.ctx.main_window)
            self.rebuild_timer.setSingleShot(True)
            self.rebuild_timer.timeout.connect(self._rebuild)
        self.rebuild_timer.start(90)

    def _sync_geometry(self):
        if self.view is None:
            return
        host = self.ctx.vtk_widget
        self.view.setGeometry(0, 0, host.width(), host.height())

    def _close(self):
        self._unhook_framework()
        if self.sync_timer is not None:
            self.sync_timer.stop()
        if self.rebuild_timer is not None:
            self.rebuild_timer.stop()
        if self.view is not None:
            self.view.hide()
            self.view.deleteLater()
            self.view = None
            self.ctx.vtk_widget.update()
            self.ctx.vtk_widget.GetRenderWindow().Render()
            self.ctx.info_display.append("[光追] 已退出预览")
        self.lbl_state.setText("未开启")
        self.btn_toggle.setText("开启光线追踪预览")

    # ---------------------------------------------------------- 选中 / 移动
    def _on_picked(self, model_index):
        scene = self.view.scene if self.view is not None else None
        if scene is None or model_index < 0:
            self.lbl_state.setText("已开启：单击选中，WASD/QE 移动")
            return
        model = scene["models"][model_index]
        name = model.get("path", "").replace("\\", "/").split("/")[-1]
        self.lbl_state.setText(f"已选中：{name}，WASD/QE 移动")
        self.ctx.info_display.append(f"[光追] 选中 {name}")
        try:
            row = self.ctx.models.index(model)
            self.ctx.main_window.model_list.setCurrentRow(row)
        except ValueError:
            pass

    def _move_selected(self, dx, dy, dz):
        if self.view is None or self.view.selected_model < 0:
            return
        scene = self.view.scene
        model = scene["models"][self.view.selected_model]
        model["pos"][0] += dx
        model["pos"][1] += dy
        model["pos"][2] += dz
        # 同步修改查看器里的 actor 位置
        try:
            self.ctx.main_window._apply_transform(model)
        except Exception:
            pass

        if self.rebuild_timer is None:
            self.rebuild_timer = QTimer(self.ctx.main_window)
            self.rebuild_timer.setSingleShot(True)
            self.rebuild_timer.timeout.connect(self._rebuild)
        self.rebuild_timer.start(90)

    def _rebuild(self):
        if self.view is None:
            return
        scene = collect_scene(self.ctx)
        if scene is None:
            return
        sel = self.view.selected_model
        self.view.update_scene(scene)
        self.view.selected_model = sel

    # ---------------------------------------------------------- 钩子
    def on_model_loaded(self, obj_path, actor, polydata):
        if self.view is not None:
            self._close()
            self.btn_toggle.setChecked(False)
            self.ctx.info_display.append("[光追] 场景有变化，请重新开启预览")

    def on_destroy(self):
        if self.sync_timer is not None:
            self.sync_timer.stop()
        if self.rebuild_timer is not None:
            self.rebuild_timer.stop()
        if self.view is not None:
            self.view.hide()
            self.view.deleteLater()
            self.view = None
