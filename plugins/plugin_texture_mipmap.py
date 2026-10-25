"""
实验二 进阶任务（一）：纹理滤波与 Mipmap 插件
==========================================================
在 3D 查看器框架里复刻实验二编程任务 1 的核心内容：
    用 4 种“插值 × Mipmap”组合给模型贴同一张纹理，观察
    最近点采样的马赛克/锯齿、双线性的摩尔纹、Mipmap 的远处平滑效果。

与 OpenGL 版本的对应关系（VTK 的 vtkTexture 只暴露 Interpolate / Mipmap 两个开关，
内部据此决定 OpenGL 的 GL_TEXTURE_MIN_FILTER / MAG_FILTER）：
    Interpolate=0, Mipmap=0  ->  GL_NEAREST                放大马赛克，缩小锯齿
    Interpolate=1, Mipmap=0  ->  GL_LINEAR                 放大平滑，缩小仍闪烁
    Interpolate=0, Mipmap=1  ->  GL_NEAREST_MIPMAP_NEAREST 按视距选层，层内最近点
    Interpolate=1, Mipmap=1  ->  GL_LINEAR_MIPMAP_LINEAR   三线性，最平滑

插件标识：texture_mipmap（在 plugin_config.yaml 中启用）
"""
import os

import vtk
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
                             QLabel, QSlider, QCheckBox, QFrame)
from PyQt5.QtCore import Qt

from plugins.base_plugin import Base3DPlugin

# (下拉框显示文本, 是否插值, 是否 Mipmap, 现象说明)
#
# 说明：VTK 的 vtkTexture 只暴露 Interpolate / Mipmap 两个开关，实测只有 3 种
# 真正有效的组合（Mipmap 链只在 Interpolate=1 时才生成，“Mipmap + 最近点”在
# VTK 里与纯最近点等效，故不列出）。完整 6 种 GL 组合由实验二的 OpenGL 程序
# texture_opengl.py 演示，本插件用于在 3D 查看器中快速对比效果。
SAMPLING_MODES = [
    ("最近点采样  GL_NEAREST", False, False,
     "放大呈马赛克；远处纹理剧烈闪烁（锯齿）"),
    ("双线性插值  GL_LINEAR", True, False,
     "放大平滑；远处仍会闪烁、出现摩尔纹"),
    ("Mipmap 三线性  GL_LINEAR_MIPMAP_LINEAR", True, True,
     "层内双线性 + 层间线性，远处不再闪烁；配合各向异性过滤斜视也清晰"),
]

# 纹理坐标的生成方式
PROJECTIONS = [
    ("保留模型自带 UV", "keep"),
    ("平面投影", "plane"),
    ("圆柱投影", "cylinder"),
    ("球面投影", "sphere"),
]


class MipmapTexturePlugin(Base3DPlugin):
    """纹理滤波 / Mipmap 插件"""

    plugin_cls_name = "texture_mipmap"
    plugin_name = "texture_mipmap"
    plugin_desc = "纹理滤波与 Mipmap 插件（实验2 进阶一）"

    def __init__(self, viewer_ctx):
        super().__init__(viewer_ctx)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.texture_dir = os.path.normpath(os.path.join(base_dir, "../data/texture"))
        self.base_dir = os.path.normpath(os.path.join(base_dir, ".."))

        self.texture_paths = []
        self.selected_texture = None
        self.anisotropy = 8
        self.repeat_tiling = True
        self.current_texture = None       # 最近一次应用的 vtkTexture

        self.combo_texture = None
        self.combo_projection = None
        self.combo_sampling = None
        self.slider_aniso = None
        self.lbl_aniso = None
        self.chk_repeat = None
        self.chk_plane = None
        self.plane_actor = None
        self.lbl_status = None

    # ------------------------------------------------------------------ UI
    def create_ui(self):
        self.plugin_panel = QFrame()
        self.plugin_panel.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self.plugin_panel)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("纹理滤波 / Mipmap（实验2 进阶）")
        title.setStyleSheet("font-weight:bold;")
        layout.addWidget(title)

        # 纹理选择
        row_tex = QHBoxLayout()
        row_tex.addWidget(QLabel("纹理"))
        self.combo_texture = QComboBox()
        self.combo_texture.currentIndexChanged.connect(self.on_texture_changed)
        row_tex.addWidget(self.combo_texture, 1)
        btn_refresh = QPushButton("刷新")
        btn_refresh.setFixedWidth(52)
        btn_refresh.clicked.connect(self.refresh_texture_list)
        row_tex.addWidget(btn_refresh)
        layout.addLayout(row_tex)

        # 投影方式
        row_proj = QHBoxLayout()
        row_proj.addWidget(QLabel("投影"))
        self.combo_projection = QComboBox()
        for name, key in PROJECTIONS:
            self.combo_projection.addItem(name, key)
        self.combo_projection.currentIndexChanged.connect(lambda _=0: self.apply_texture(auto=True))
        row_proj.addWidget(self.combo_projection, 1)
        layout.addLayout(row_proj)

        # 采样方式
        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("采样"))
        self.combo_sampling = QComboBox()
        for name, _interp, _mip, _desc in SAMPLING_MODES:
            self.combo_sampling.addItem(name)
        self.combo_sampling.setCurrentIndex(len(SAMPLING_MODES) - 1)   # 默认三线性
        self.combo_sampling.currentIndexChanged.connect(self.on_sampling_changed)
        row_mode.addWidget(self.combo_sampling, 1)
        layout.addLayout(row_mode)

        # 各向异性过滤
        row_aniso = QHBoxLayout()
        row_aniso.addWidget(QLabel("各向异性"))
        self.slider_aniso = QSlider(Qt.Horizontal)
        self.slider_aniso.setRange(1, 16)
        self.slider_aniso.setValue(self.anisotropy)
        self.slider_aniso.valueChanged.connect(self.on_aniso_changed)
        row_aniso.addWidget(self.slider_aniso, 1)
        self.lbl_aniso = QLabel(f"{self.anisotropy}x")
        self.lbl_aniso.setFixedWidth(32)
        row_aniso.addWidget(self.lbl_aniso)
        layout.addLayout(row_aniso)

        # 重复平铺
        self.chk_repeat = QCheckBox("纹理重复平铺（Repeat）")
        self.chk_repeat.setChecked(self.repeat_tiling)
        self.chk_repeat.stateChanged.connect(self.on_repeat_changed)
        layout.addWidget(self.chk_repeat)

        # 对比地面：把同一张纹理平铺 24×24 铺在地上，远处强烈缩小，
        # 这样 Mipmap 与普通插值的区别一眼就能看出来
        self.chk_plane = QCheckBox("显示 Mipmap 对比地面")
        self.chk_plane.setChecked(False)
        self.chk_plane.stateChanged.connect(self.on_plane_toggled)
        layout.addWidget(self.chk_plane)

        btn_apply = QPushButton("应用到当前模型")
        btn_apply.clicked.connect(lambda _=0: self.apply_texture())
        layout.addWidget(btn_apply)

        self.lbl_status = QLabel("提示：先加载 OBJ 模型，再选择纹理")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#555;")
        layout.addWidget(self.lbl_status)

        self.refresh_texture_list()
        return self.plugin_panel

    # -------------------------------------------------------------- 纹理列表
    def refresh_texture_list(self):
        self.texture_paths = []
        if os.path.isdir(self.texture_dir):
            for name in sorted(os.listdir(self.texture_dir)):
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.texture_paths.append(os.path.join(self.texture_dir, name))
        self.combo_texture.blockSignals(True)
        self.combo_texture.clear()
        for path in self.texture_paths:
            self.combo_texture.addItem(os.path.basename(path))
        self.combo_texture.blockSignals(False)

        if self.texture_paths:
            self.selected_texture = self.texture_paths[0]
            self._set_status(f"找到 {len(self.texture_paths)} 张纹理：{self.texture_dir}")
        else:
            self.selected_texture = None
            self._set_status(f"未找到纹理图片：{self.texture_dir}")

    # --------------------------------------------------------------- 事件
    def on_texture_changed(self, index):
        if 0 <= index < len(self.texture_paths):
            self.selected_texture = self.texture_paths[index]
        self.apply_texture(auto=True)

    def on_sampling_changed(self, index):
        _name, _interp, _mip, desc = SAMPLING_MODES[index]
        self.apply_texture(auto=True)
        self._set_status(desc)

    def on_aniso_changed(self, value):
        self.anisotropy = int(value)
        self.lbl_aniso.setText(f"{self.anisotropy}x")
        if self.current_texture is not None:
            self.current_texture.SetMaximumAnisotropicFiltering(self.anisotropy)
            self._render()

    def on_repeat_changed(self, state):
        self.repeat_tiling = (state == Qt.Checked)
        if self.current_texture is not None:
            self.current_texture.SetRepeat(1 if self.repeat_tiling else 0)
            self.current_texture.SetEdgeClamp(0 if self.repeat_tiling else 1)
            self._render()

    # --------------------------------------------------------------- 应用
    def apply_texture(self, auto=False):
        actor = self.ctx.current_actor
        polydata = self.ctx.current_polydata
        if actor is None or polydata is None:
            if not auto:
                self._set_status("请先在左侧文件树里加载一个 OBJ 模型")
            return
        if not self.selected_texture:
            self._set_status("没有可用纹理")
            return

        path = self.selected_texture
        lower = path.lower()
        if lower.endswith((".jpg", ".jpeg")):
            reader = vtk.vtkJPEGReader()
        else:
            reader = vtk.vtkPNGReader()
        reader.SetFileName(path)
        reader.Update()
        if reader.GetOutput().GetNumberOfPoints() == 0:
            self._set_status(f"纹理读取失败：{os.path.basename(path)}")
            return

        mode = self.combo_sampling.currentIndex()
        _name, interp, mip, desc = SAMPLING_MODES[mode]

        texture = vtk.vtkTexture()
        texture.SetInputConnection(reader.GetOutputPort())
        texture.SetInterpolate(1 if interp else 0)
        texture.SetMipmap(1 if mip else 0)
        texture.SetMaximumAnisotropicFiltering(self.anisotropy)
        texture.SetRepeat(1 if self.repeat_tiling else 0)
        texture.SetEdgeClamp(0 if self.repeat_tiling else 1)

        # 纹理坐标：优先保留模型自带 UV，否则按选择的投影方式生成
        mapped, uv_note = self._build_uv(polydata, self.combo_projection.currentData())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mapped)
        mapper.SetScalarVisibility(False)
        actor.SetMapper(mapper)
        actor.SetTexture(texture)

        self.current_texture = texture
        self._update_plane(texture)
        self._render()
        self._set_status(
            f"{os.path.basename(path)}｜{desc}\nUV：{uv_note}｜各向异性 {self.anisotropy}x"
            f"｜{'重复' if self.repeat_tiling else '边缘钳制'}")

    def _build_uv(self, polydata, kind):
        """返回 (带 UV 的 polydata, 说明文字)"""
        has_uv = polydata.GetPointData().GetTCoords() is not None
        if kind == "keep":
            if has_uv:
                return polydata, "模型自带 UV"
            kind = "plane"          # 模型没有 UV 时退回平面投影

        if kind == "plane":
            gen = vtk.vtkTextureMapToPlane()
            note = "平面投影"
        elif kind == "cylinder":
            gen = vtk.vtkTextureMapToCylinder()
            gen.PreventSeamOn()
            note = "圆柱投影"
        else:
            gen = vtk.vtkTextureMapToSphere()
            gen.PreventSeamOn()
            note = "球面投影"
        gen.SetInputData(polydata)
        gen.Update()
        return gen.GetOutput(), note

    # ------------------------------------------------------------ 对比地面
    def on_plane_toggled(self, _state=None):
        self._update_plane(self.current_texture)
        self._render()

    def _update_plane(self, texture):
        """把同一张纹理以 24×24 平铺到一块大地面（UV 超出 0~1 即重复），
        地平线附近纹理被强烈缩小，4 种采样方式的差别在这里最明显。"""
        renderer = self.ctx.vtk_renderer
        if renderer is None:
            return
        if texture is None or not self.chk_plane.isChecked():
            if self.plane_actor is not None:
                renderer.RemoveActor(self.plane_actor)
                self.plane_actor = None
            return
        if self.plane_actor is None:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self._build_plane(self.ctx.current_polydata))
            mapper.SetScalarVisibility(False)
            self.plane_actor = vtk.vtkActor()
            self.plane_actor.SetMapper(mapper)
            self.plane_actor.PickableOff()
        self.plane_actor.SetTexture(texture)
        renderer.RemoveActor(self.plane_actor)
        renderer.AddActor(self.plane_actor)

    def _build_plane(self, model_polydata, tiles=12):
        """生成一块带重复 UV 的地面"""
        if model_polydata is not None and model_polydata.GetNumberOfPoints() > 0:
            b = model_polydata.GetBounds()
            diag = model_polydata.GetLength() or 1.0
            y = b[2] - diag * 0.02
            half = max(diag * 0.75, 2.5)
            cx, cz = (b[0] + b[1]) / 2.0, (b[4] + b[5]) / 2.0
        else:
            y, half, cx, cz = 0.0, 6.0, 0.0, 0.0

        src = vtk.vtkPlaneSource()
        src.SetOrigin(cx - half, y, cz - half)
        src.SetPoint1(cx + half, y, cz - half)
        src.SetPoint2(cx - half, y, cz + half)
        src.SetXResolution(1)
        src.SetYResolution(1)
        src.Update()
        polydata = src.GetOutput()

        tcoords = vtk.vtkFloatArray()
        tcoords.SetNumberOfComponents(2)
        tcoords.SetName("TextureCoordinates")
        for u, v in ((0.0, 0.0), (float(tiles), 0.0),
                     (float(tiles), float(tiles)), (0.0, float(tiles))):
            tcoords.InsertNextTuple2(u, v)
        polydata.GetPointData().SetTCoords(tcoords)
        return polydata

    # --------------------------------------------------------------- 钩子
    def on_model_loaded(self, obj_path, actor, polydata):
        # 新模型加载后自动套用上一次的纹理设置
        if self.selected_texture:
            self.apply_texture(auto=True)

    def on_destroy(self):
        renderer = self.ctx.vtk_renderer
        if renderer is not None and self.plane_actor is not None:
            renderer.RemoveActor(self.plane_actor)
        self.plane_actor = None
        self.current_texture = None

    # --------------------------------------------------------------- 工具
    def _render(self):
        if self.ctx.vtk_widget is not None:
            self.ctx.vtk_widget.GetRenderWindow().Render()

    def _set_status(self, text):
        if self.lbl_status is not None:
            self.lbl_status.setText(text)
        if self.ctx.info_display is not None:
            self.ctx.info_display.append(f"[Mipmap] {text.splitlines()[0]}")



