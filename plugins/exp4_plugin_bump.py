"""
实验四 进阶：为 3D 查看器中加载的模型增加凹凸映射
=========================================================
做法是扰动顶点法线：在一个程序化的三维高度场上取每个顶点的梯度，
把梯度中垂直于法线的分量加到法线上，再归一化。几何不动，但光照算出来
表面就像有凹凸（和片元级 bump mapping 是同一套原理，只是这里按顶点算）。

为了让低模也有细节，插件会先对模型做一次线性细分再扰动法线。

用法：
    右侧面板调“凹凸强度 / 细节尺度”，点“应用到选中”或“应用到全部”；
    “恢复原状”把模型还原成原始网格。
=========================================================
"""
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton,
                             QSlider, QVBoxLayout, QWidget)

import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from plugins.base_plugin import Base3DPlugin


def _subdivide(polydata, max_tris=40000):
    """三角形化 + 线性细分一级，让法线扰动有足够的顶点"""
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.Update()
    data = tri.GetOutput()

    if data.GetNumberOfPolys() * 4 <= max_tris:
        try:
            sub = vtk.vtkLinearSubdivisionFilter()
            sub.SetInputData(data)
            sub.SetNumberOfSubdivisions(1)
            sub.Update()
            out = sub.GetOutput()
            if out.GetNumberOfPoints() > data.GetNumberOfPoints():
                return out
        except Exception:
            pass
    return data


def _normals_of(polydata):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.ComputePointNormalsOn()
    normals.Update()
    return normals.GetOutput()


# 几组不同方向的波，叠起来接近各向同性的凹凸
_WAVES = [
    (np.array([1.00, 0.62, 0.38]), 1.00, 0.0),
    (np.array([0.55, -1.15, 0.90]), 0.62, 1.7),
    (np.array([1.90, 1.25, -0.70]), 0.38, 4.1),
    (np.array([-0.80, 0.45, 1.60]), 0.24, 2.3),
]


def _height_gradient(p, scale):
    """程序化高度场的解析梯度：h = Σ a·sin(k·p)，梯度 = Σ a·k·cos(k·p)"""
    grad = np.zeros_like(p)
    height = np.zeros(len(p))
    for k, amp, phase in _WAVES:
        arg = (p @ k) * scale + phase
        height += amp * np.sin(arg)
        grad += amp * np.cos(arg)[:, None] * (k[None, :] * scale)
    return grad, height


def bump_normals(polydata, strength=0.6, scale=6.0):
    """返回带扰动法线的新 polydata"""
    mesh = _normals_of(_subdivide(polydata))
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)
    normals = vtk_to_numpy(mesh.GetPointData().GetNormals()).astype(np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)

    grad, _ = _height_gradient(points, scale)
    # 只保留切向分量，法线方向上的分量对明暗没意义
    tangential = grad - normals * np.sum(grad * normals, axis=1, keepdims=True)
    bumped = normals - strength * tangential
    bumped /= np.maximum(np.linalg.norm(bumped, axis=1, keepdims=True), 1e-9)

    arr = numpy_to_vtk(bumped.astype(np.float64), deep=True)
    arr.SetName("Normals")
    mesh.GetPointData().SetNormals(arr)
    mesh.Modified()
    return mesh


class Exp4BumpPlugin(Base3DPlugin):
    plugin_cls_name = "exp4_bump"
    plugin_name = "实验四·凹凸映射"
    plugin_desc = "扰动模型法线，让表面出现凹凸细节（可调强度与尺度）"

    def __init__(self, viewer_ctx):
        super().__init__(viewer_ctx)
        self.strength = 0.35
        self.scale = 7.0
        self.applied = {}          # id(model) -> 已应用的 polydata

    # ---------------------------------------------------------- UI
    def create_ui(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        tip = QLabel("用程序化高度场扰动法线：几何不变，靠光照表现凹凸。"
                     "低模会先细分一级再算，效果更明显。")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        row = QHBoxLayout()
        row.addWidget(QLabel("强度"))
        self.sld_strength = QSlider(Qt.Horizontal)
        self.sld_strength.setRange(0, 200)
        self.sld_strength.setValue(int(self.strength * 100))
        self.sld_strength.valueChanged.connect(self._on_strength)
        row.addWidget(self.sld_strength, 1)
        self.lbl_strength = QLabel(f"{self.strength:.2f}")
        row.addWidget(self.lbl_strength)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("尺度"))
        self.sld_scale = QSlider(Qt.Horizontal)
        self.sld_scale.setRange(10, 300)
        self.sld_scale.setValue(int(self.scale * 10))
        self.sld_scale.valueChanged.connect(self._on_scale)
        row2.addWidget(self.sld_scale, 1)
        self.lbl_scale = QLabel(f"{self.scale:.1f}")
        row2.addWidget(self.lbl_scale)
        layout.addLayout(row2)

        btn_row = QHBoxLayout()
        btn_cur = QPushButton("应用到选中")
        btn_cur.clicked.connect(self.apply_to_current)
        btn_row.addWidget(btn_cur)
        btn_all = QPushButton("应用到全部")
        btn_all.clicked.connect(self.apply_to_all)
        btn_row.addWidget(btn_all)
        layout.addLayout(btn_row)

        btn_reset = QPushButton("恢复原状")
        btn_reset.clicked.connect(self.restore_all)
        layout.addWidget(btn_reset)

        self.chk_auto = QCheckBox("加载模型时自动应用")
        self.chk_auto.setChecked(False)
        layout.addWidget(self.chk_auto)
        return panel

    # ---------------------------------------------------------- 参数
    def _on_strength(self, value):
        self.strength = value / 100.0
        self.lbl_strength.setText(f"{self.strength:.2f}")

    def _on_scale(self, value):
        self.scale = value / 10.0
        self.lbl_scale.setText(f"{self.scale:.1f}")

    # ---------------------------------------------------------- 应用
    def _targets(self, current_only):
        models = list(getattr(self.ctx, "models", []))
        if not current_only:
            return models
        actor = getattr(self.ctx, "current_actor", None)
        return [m for m in models if m.get("actor") is actor]

    def _apply_model(self, model):
        polydata = model.get("polydata")
        if polydata is None:
            return False
        try:
            bumped = bump_normals(polydata, self.strength, self.scale)
        except Exception as exc:
            self.ctx.info_display.append(f"[凹凸] 处理失败：{exc}")
            return False

        mapper = model["actor"].GetMapper()
        if mapper.GetInput() is not bumped:
            mapper.SetInputData(bumped)
        mapper.Modified()
        model["actor"].Modified()
        self.applied[id(model)] = True
        return True

    def apply_to_current(self):
        targets = self._targets(True)
        if not targets:
            self.ctx.info_display.append("[凹凸] 没有选中的模型")
            return
        if self._apply_model(targets[0]):
            self.ctx.info_display.append(
                f"[凹凸] 已应用：强度 {self.strength:.2f}，尺度 {self.scale:.1f}")
            self._render()

    def apply_to_all(self):
        targets = self._targets(False)
        if not targets:
            self.ctx.info_display.append("[凹凸] 场景里还没有模型")
            return
        ok = sum(1 for m in targets if self._apply_model(m))
        self.ctx.info_display.append(
            f"[凹凸] 已应用到 {ok} 个模型（强度 {self.strength:.2f}，尺度 {self.scale:.1f}）")
        self._render()

    def restore_all(self):
        count = 0
        for model in getattr(self.ctx, "models", []):
            if id(model) in self.applied:
                mapper = model["actor"].GetMapper()
                mapper.SetInputData(model["polydata"])
                mapper.Modified()
                model["actor"].Modified()
                del self.applied[id(model)]
                count += 1
        self.ctx.info_display.append(f"[凹凸] 已恢复 {count} 个模型")
        self._render()

    def _render(self):
        win = self.ctx.vtk_widget.GetRenderWindow()
        if win is not None:
            win.Render()

    # ---------------------------------------------------------- 钩子
    def on_model_loaded(self, obj_path, actor, polydata):
        if self.chk_auto.isChecked():
            for model in getattr(self.ctx, "models", []):
                if model.get("actor") is actor:
                    self._apply_model(model)
                    break
            self._render()

    def on_destroy(self):
        self.restore_all()
