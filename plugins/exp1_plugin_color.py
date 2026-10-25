import os
import colorsys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QSlider, QPushButton, QColorDialog, QGroupBox, 
    QGridLayout, QScrollArea, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
import vtk
from plugins.base_plugin import Base3DPlugin


# ---------------- 1. 自定义控件：单击数值并横向拖拽调节 ----------------
class DraggableValueLabel(QLabel):
    """单击数值并水平拖动鼠标增减数值 (0.00 ~ 1.00)"""
    valueChanged = pyqtSignal(float)

    def __init__(self, value=0.1, min_val=0.0, max_val=1.0, step=0.005, parent=None):
        super().__init__(parent)
        self.val = value
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.is_dragging = False
        self.last_mouse_x = 0

        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                color: #00e5ff;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 6px;
                min-width: 50px;
            }
            QLabel:hover {
                border: 1px solid #00e5ff;
                background-color: #383838;
            }
        """)
        self._refresh_text()

    def set_value(self, new_val):
        self.val = max(self.min_val, min(self.max_val, new_val))
        self._refresh_text()

    def _refresh_text(self):
        self.setText(f"{self.val:.2f}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.last_mouse_x = event.globalPos().x()
            self.setCursor(Qt.SizeHorCursor)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            dx = event.globalPos().x() - self.last_mouse_x
            self.last_mouse_x = event.globalPos().x()
            
            delta = dx * self.step
            new_val = max(self.min_val, min(self.max_val, self.val + delta))
            if abs(new_val - self.val) > 1e-4:
                self.val = new_val
                self._refresh_text()
                self.valueChanged.emit(self.val)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.setCursor(Qt.ArrowCursor)


# ---------------- 2. 核心插件实现 ----------------
class ColorPlugin(Base3DPlugin):
    plugin_cls_name = "color_control"
    plugin_name = "color_control"
    plugin_desc = "实验1：HSV/RGB变换与3D色彩实体"

    def __init__(self, viewer_ctx):
        super().__init__(viewer_ctx)
        self.h = 0.0
        self.s = 1.0
        self.v = 0.9
        self.r, self.g, self.b = colorsys.hsv_to_rgb(self.h, self.s, self.v)
        self.cube_opacity = 0.88

        self._is_syncing = False
        self.current_mode = "cube"  # 可选: "model", "triangle", "cube"

        # VTK 实体对象引用
        self.tri_actors = []
        self.cube_actors = []
        self.marker_actor = None
        self.cube_solid_actor = None
        self.tri_hsv_actor = None
        self.tri_rgb_actor = None

    def create_ui(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.panel = QWidget()
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ====== 1. 视口显示内容选择 ======
        view_group = QGroupBox("视口显示内容选择")
        view_layout = QVBoxLayout()
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("任务 2：RGB 渐变彩色立方体", "cube")
        self.combo_mode.addItem("任务 1：色彩三角形 (HSV & RGB)", "triangle")
        self.combo_mode.addItem("当前 3D 模型 (OBJ)", "model")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        view_layout.addWidget(self.combo_mode)

        btn_reset_cam = QPushButton("复位中央视口视角")
        btn_reset_cam.clicked.connect(self._reset_view_camera)
        view_layout.addWidget(btn_reset_cam)
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # ====== 2. HSV 控制区 ======
        hsv_group = QGroupBox("HSV 控制区 (单击数值横向拖拽)")
        hsv_layout = QGridLayout()

        hsv_layout.addWidget(QLabel("H (色相)"), 0, 0, Qt.AlignCenter)
        hsv_layout.addWidget(QLabel("S (饱和度)"), 0, 1, Qt.AlignCenter)
        hsv_layout.addWidget(QLabel("V (明度)"), 0, 2, Qt.AlignCenter)

        self.lbl_h = DraggableValueLabel(self.h)
        self.lbl_s = DraggableValueLabel(self.s)
        self.lbl_v = DraggableValueLabel(self.v)

        hsv_layout.addWidget(self.lbl_h, 1, 0)
        hsv_layout.addWidget(self.lbl_s, 1, 1)
        hsv_layout.addWidget(self.lbl_v, 1, 2)

        self.slider_h = self._create_slider(self.h)
        self.slider_s = self._create_slider(self.s)
        self.slider_v = self._create_slider(self.v)

        hsv_layout.addWidget(self.slider_h, 2, 0)
        hsv_layout.addWidget(self.slider_s, 2, 1)
        hsv_layout.addWidget(self.slider_v, 2, 2)

        self.lbl_h.valueChanged.connect(self._on_hsv_lbl_changed)
        self.lbl_s.valueChanged.connect(self._on_hsv_lbl_changed)
        self.lbl_v.valueChanged.connect(self._on_hsv_lbl_changed)
        self.slider_h.valueChanged.connect(self._on_hsv_slider_changed)
        self.slider_s.valueChanged.connect(self._on_hsv_slider_changed)
        self.slider_v.valueChanged.connect(self._on_hsv_slider_changed)

        hsv_group.setLayout(hsv_layout)
        layout.addWidget(hsv_group)

        # ====== 3. RGB 控制区 ======
        rgb_group = QGroupBox("RGB 控制区 (单击数值横向拖拽)")
        rgb_layout = QGridLayout()

        rgb_layout.addWidget(QLabel("R (红)"), 0, 0, Qt.AlignCenter)
        rgb_layout.addWidget(QLabel("G (绿)"), 0, 1, Qt.AlignCenter)
        rgb_layout.addWidget(QLabel("B (蓝)"), 0, 2, Qt.AlignCenter)

        self.lbl_r = DraggableValueLabel(self.r)
        self.lbl_g = DraggableValueLabel(self.g)
        self.lbl_b = DraggableValueLabel(self.b)

        rgb_layout.addWidget(self.lbl_r, 1, 0)
        rgb_layout.addWidget(self.lbl_g, 1, 1)
        rgb_layout.addWidget(self.lbl_b, 1, 2)

        self.slider_r = self._create_slider(self.r)
        self.slider_g = self._create_slider(self.g)
        self.slider_b = self._create_slider(self.b)

        rgb_layout.addWidget(self.slider_r, 2, 0)
        rgb_layout.addWidget(self.slider_g, 2, 1)
        rgb_layout.addWidget(self.slider_b, 2, 2)

        self.lbl_r.valueChanged.connect(self._on_rgb_lbl_changed)
        self.lbl_g.valueChanged.connect(self._on_rgb_lbl_changed)
        self.lbl_b.valueChanged.connect(self._on_rgb_lbl_changed)
        self.slider_r.valueChanged.connect(self._on_rgb_slider_changed)
        self.slider_g.valueChanged.connect(self._on_rgb_slider_changed)
        self.slider_b.valueChanged.connect(self._on_rgb_slider_changed)

        rgb_group.setLayout(rgb_layout)
        layout.addWidget(rgb_group)

        # ====== 4. 立方体材质与常用调色 ======
        tool_group = QGroupBox("立方体材质与预设色彩")
        tool_layout = QGridLayout()

        tool_layout.addWidget(QLabel("立方体透明度:"), 0, 0)
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(20, 100)
        self.slider_opacity.setValue(int(self.cube_opacity * 100))
        self.slider_opacity.valueChanged.connect(self._on_cube_opacity_changed)
        tool_layout.addWidget(self.slider_opacity, 0, 1, 1, 2)

        btn_picker = QPushButton("打开调色板")
        btn_picker.clicked.connect(self._open_picker)
        tool_layout.addWidget(btn_picker, 1, 0, 1, 3)

        presets = [
            ("纯红", (1.0, 0.0, 0.0)),
            ("纯绿", (0.0, 1.0, 0.0)),
            ("纯蓝", (0.0, 0.0, 1.0)),
            ("黄色", (1.0, 1.0, 0.0)),
            ("青色", (0.0, 1.0, 1.0)),
            ("纯白", (1.0, 1.0, 1.0))
        ]
        for i, (name, rgb_c) in enumerate(presets):
            btn = QPushButton(name)
            btn.clicked.connect(lambda ch, c=rgb_c: self._apply_preset_rgb(*c))
            tool_layout.addWidget(btn, 2 + i // 3, i % 3)

        tool_group.setLayout(tool_layout)
        layout.addWidget(tool_group)

        layout.addStretch()
        scroll.setWidget(self.panel)

        # 初始化 3D 视口实体
        self._init_vtk_entities()
        return scroll

    def _create_slider(self, val):
        s = QSlider(Qt.Horizontal)
        s.setRange(0, 100)
        s.setValue(int(val * 100))
        return s

    # ---------------- 3. VTK 3D 实体构建 ----------------
    def _init_vtk_entities(self):
        """仅构建 VTK 对象，不重复 AddActor"""
        if self.tri_actors and self.cube_actors:
            return

        # ========= 构建任务 1：三角形实体与文字 =========
        self.tri_hsv_actor = self._create_solid_triangle((-0.85, 0.0, 0.0))
        self.tri_rgb_actor = self._create_solid_triangle((0.85, 0.0, 0.0))
        self.txt_hsv_actor = self._create_3d_text("HSV", (-1.0, 0.8, 0.0))
        self.txt_rgb_actor = self._create_3d_text("RGB", (0.7, 0.8, 0.0))
        self.tri_actors = [self.tri_hsv_actor, self.tri_rgb_actor, self.txt_hsv_actor, self.txt_rgb_actor]

        # ========= 构建任务 2：实心渐变彩色立方体 =========
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("RGBColors")

        vertex_coords = [
            (0.0, 0.0, 0.0),  # 0: 黑
            (1.0, 0.0, 0.0),  # 1: 红
            (1.0, 1.0, 0.0),  # 2: 黄
            (0.0, 1.0, 0.0),  # 3: 绿
            (0.0, 0.0, 1.0),  # 4: 蓝
            (1.0, 0.0, 1.0),  # 5: 品红
            (1.0, 1.0, 1.0),  # 6: 白
            (0.0, 1.0, 1.0)   # 7: 青
        ]
        for p in vertex_coords:
            points.InsertNextPoint(p)
            colors.InsertNextTuple3(int(p[0] * 255), int(p[1] * 255), int(p[2] * 255))

        triangles = [
            (0, 3, 2), (0, 2, 1),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (2, 3, 7), (2, 7, 6),
            (0, 4, 7), (0, 7, 3),
            (1, 2, 6), (1, 6, 5)
        ]
        cells = vtk.vtkCellArray()
        for t in triangles:
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, t[0])
            tri.GetPointIds().SetId(1, t[1])
            tri.GetPointIds().SetId(2, t[2])
            cells.InsertNextCell(tri)

        cube_poly = vtk.vtkPolyData()
        cube_poly.SetPoints(points)
        cube_poly.SetPolys(cells)
        cube_poly.GetPointData().SetScalars(colors)

        cube_mapper = vtk.vtkPolyDataMapper()
        cube_mapper.SetInputData(cube_poly)
        cube_mapper.SetScalarVisibility(True)

        self.cube_solid_actor = vtk.vtkActor()
        self.cube_solid_actor.SetMapper(cube_mapper)
        self.cube_solid_actor.GetProperty().SetRepresentationToSurface()
        self.cube_solid_actor.GetProperty().SetInterpolationToGouraud()
        self.cube_solid_actor.GetProperty().SetAmbient(0.9)
        self.cube_solid_actor.GetProperty().SetDiffuse(0.1)
        self.cube_solid_actor.GetProperty().SetOpacity(self.cube_opacity)
        self.cube_solid_actor.GetProperty().BackfaceCullingOff()

        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(cube_poly)
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        self.cube_outline_actor = vtk.vtkActor()
        self.cube_outline_actor.SetMapper(outline_mapper)
        self.cube_outline_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.cube_outline_actor.GetProperty().SetLineWidth(1.5)

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.045)
        sphere.SetPhiResolution(24)
        sphere.SetThetaResolution(24)
        marker_mapper = vtk.vtkPolyDataMapper()
        marker_mapper.SetInputConnection(sphere.GetOutputPort())
        self.marker_actor = vtk.vtkActor()
        self.marker_actor.SetMapper(marker_mapper)
        self.marker_actor.GetProperty().SetSpecular(0.5)

        self.cube_actors = [self.cube_solid_actor, self.cube_outline_actor, self.marker_actor]

        # 初次挂载并切换为默认立方体模式
        self._switch_mode("cube")

    def _create_solid_triangle(self, center):
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(center[0] - 0.55, center[1] - 0.5, 0.0)
        pts.InsertNextPoint(center[0] + 0.55, center[1] - 0.5, 0.0)
        pts.InsertNextPoint(center[0], center[1] + 0.6, 0.0)

        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, 0)
        tri.GetPointIds().SetId(1, 1)
        tri.GetPointIds().SetId(2, 2)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(tri)

        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToSurface()
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        actor.GetProperty().BackfaceCullingOff()
        return actor

    def _create_3d_text(self, text, pos):
        txt = vtk.vtkVectorText()
        txt.SetText(text)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(txt.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetScale(0.18, 0.18, 0.18)
        actor.SetPosition(pos[0], pos[1], pos[2])
        actor.GetProperty().SetColor(1, 1, 1)
        return actor

    # ---------------- 4. 模式切换与自愈重新挂载 ----------------
    def _on_mode_changed(self, idx):
        mode_key = self.combo_mode.currentData()
        self._switch_mode(mode_key)

    def _switch_mode(self, mode):
        """核心修复：解决 RemoveAllViewProps 之后 Actor 丢失的问题"""
        self.current_mode = mode
        if not self.ctx or not self.ctx.vtk_renderer:
            return

        renderer = self.ctx.vtk_renderer
        is_cube = (mode == "cube")
        is_tri = (mode == "triangle")
        is_model = (mode == "model")

        # 1. 管理立方体 Actor（若被内核 RemoveAllViewProps 移出，则在此自动重新 AddActor）
        for a in self.cube_actors:
            if is_cube:
                if not renderer.HasViewProp(a):
                    renderer.AddActor(a)
                a.SetVisibility(True)
            else:
                a.SetVisibility(False)

        # 2. 管理三角形 Actor（若被内核移出，则自动重新 AddActor）
        for a in self.tri_actors:
            if is_tri:
                if not renderer.HasViewProp(a):
                    renderer.AddActor(a)
                a.SetVisibility(True)
            else:
                a.SetVisibility(False)

        # 3. 管理加载的 OBJ 模型（内核支持多模型，这里统一控制显隐）
        models = getattr(self.ctx, "models", None)
        model_actors = [m["actor"] for m in models] if models else []
        if not model_actors and self.ctx.current_actor:
            model_actors = [self.ctx.current_actor]
        for actor in model_actors:
            if is_model and not renderer.HasViewProp(actor):
                renderer.AddActor(actor)
            actor.SetVisibility(is_model)

        # 4. 隐藏/显示底座加载模型时附带的坐标轴（vtkAxesActor），避免与任务1/2穿插
        for prop in renderer.GetViewProps():
            if isinstance(prop, vtk.vtkAxesActor):
                prop.SetVisibility(is_model)

        # 5. 刷新数据并自适应相机视角
        self._sync_all_data()
        self._reset_view_camera()

    def _reset_view_camera(self):
        """依据当前显示的内容类型自适应重置相机"""
        if not self.ctx or not self.ctx.vtk_renderer:
            return
        cam = self.ctx.vtk_renderer.GetActiveCamera()
        if self.current_mode == "cube":
            cam.SetFocalPoint(0.5, 0.5, 0.5)
            cam.SetPosition(2.2, 2.0, 2.5)
            cam.SetViewUp(0, 0, 1)
        elif self.current_mode == "triangle":
            cam.SetFocalPoint(0.0, 0.0, 0.0)
            cam.SetPosition(0.0, 0.0, 3.5)
            cam.SetViewUp(0, 1, 0)
        else:
            window = getattr(self.ctx, "main_window", None)
            if window is not None and hasattr(window, "fit_camera"):
                window.fit_camera()      # 只按模型包围盒适配，避免被天空盒带偏
            else:
                self.ctx.vtk_renderer.ResetCamera()

        self.ctx.vtk_renderer.ResetCameraClippingRange()
        self.ctx.vtk_widget.GetRenderWindow().Render()

    # ---------------- 5. 色彩计算与多向同步 ----------------
    def _on_hsv_lbl_changed(self):
        if self._is_syncing: return
        self.h, self.s, self.v = self.lbl_h.val, self.lbl_s.val, self.lbl_v.val
        self._convert_hsv_to_rgb()

    def _on_hsv_slider_changed(self):
        if self._is_syncing: return
        self.h = self.slider_h.value() / 100.0
        self.s = self.slider_s.value() / 100.0
        self.v = self.slider_v.value() / 100.0
        self._convert_hsv_to_rgb()

    def _convert_hsv_to_rgb(self):
        self._is_syncing = True
        self.r, self.g, self.b = colorsys.hsv_to_rgb(self.h, self.s, self.v)
        self._sync_all_data()
        self._is_syncing = False

    def _on_rgb_lbl_changed(self):
        if self._is_syncing: return
        self.r, self.g, self.b = self.lbl_r.val, self.lbl_g.val, self.lbl_b.val
        self._convert_rgb_to_hsv()

    def _on_rgb_slider_changed(self):
        if self._is_syncing: return
        self.r = self.slider_r.value() / 100.0
        self.g = self.slider_g.value() / 100.0
        self.b = self.slider_b.value() / 100.0
        self._convert_rgb_to_hsv()

    def _convert_rgb_to_hsv(self):
        self._is_syncing = True
        self.h, self.s, self.v = colorsys.rgb_to_hsv(self.r, self.g, self.b)
        self._sync_all_data()
        self._is_syncing = False

    def _apply_preset_rgb(self, r, g, b):
        self.r, self.g, self.b = r, g, b
        self._convert_rgb_to_hsv()

    def _open_picker(self):
        c = QColorDialog.getColor(QColor(int(self.r*255), int(self.g*255), int(self.b*255)), self.panel)
        if c.isValid():
            self.r, self.g, self.b = c.redF(), c.greenF(), c.blueF()
            self._convert_rgb_to_hsv()

    def _on_cube_opacity_changed(self, val):
        self.cube_opacity = val / 100.0
        if self.cube_solid_actor:
            self.cube_solid_actor.GetProperty().SetOpacity(self.cube_opacity)
            self.ctx.vtk_widget.GetRenderWindow().Render()

    def _sync_all_data(self):
        self.lbl_h.set_value(self.h)
        self.lbl_s.set_value(self.s)
        self.lbl_v.set_value(self.v)
        self.slider_h.setValue(int(self.h * 100))
        self.slider_s.setValue(int(self.s * 100))
        self.slider_v.setValue(int(self.v * 100))

        self.lbl_r.set_value(self.r)
        self.lbl_g.set_value(self.g)
        self.lbl_b.set_value(self.b)
        self.slider_r.setValue(int(self.r * 100))
        self.slider_g.setValue(int(self.g * 100))
        self.slider_b.setValue(int(self.b * 100))

        if not self.ctx or not self.ctx.vtk_renderer:
            return

        # 任务 1：三角形着色
        if self.current_mode == "triangle" and self.tri_hsv_actor and self.tri_rgb_actor:
            self.tri_hsv_actor.GetProperty().SetColor(self.r, self.g, self.b)
            self.tri_rgb_actor.GetProperty().SetColor(self.r, self.g, self.b)

        # 任务 2：色彩立方体标记点实时位移与变色
        elif self.current_mode == "cube" and self.marker_actor:
            self.marker_actor.SetPosition(self.r, self.g, self.b)
            self.marker_actor.GetProperty().SetColor(self.r, self.g, self.b)

        # 模式 3：3D OBJ 模型上色
        elif self.current_mode == "model" and self.ctx.current_actor:
            actor = self.ctx.current_actor
            if actor.GetMapper():
                actor.GetMapper().SetScalarVisibility(False)
            if actor.GetTexture():
                actor.SetTexture(None)
            prop = actor.GetProperty()
            prop.SetColor(self.r, self.g, self.b)
            prop.SetAmbient(0.3)
            prop.SetDiffuse(0.7)

        self.ctx.vtk_widget.GetRenderWindow().Render()

        if hasattr(self.ctx, "info_display") and self.ctx.info_display:
            msg = f"[色彩联动] HSV: ({self.h:.2f}, {self.s:.2f}, {self.v:.2f}) <==> RGB: ({self.r:.2f}, {self.g:.2f}, {self.b:.2f})"
            self.ctx.info_display.setText(msg)

    # ---------------- 插件生命周期钩子 ----------------
    def on_model_loaded(self, obj_path: str, actor, polydata):
        """加载 OBJ 时阻断事件风暴，自动切换至模型选项"""
        self.combo_mode.blockSignals(True)
        self.combo_mode.setCurrentIndex(2)  # 切到 '当前 3D 模型' 选项
        self.combo_mode.blockSignals(False)
        self._switch_mode("model")