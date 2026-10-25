'''
框架内核：基础 3D 查看器底座（多模型版）
------------------------------------------------------------------
相比原版的关键改动：
  1. 支持同时显示多个模型：ctx.models 维护模型列表，加载不再清空场景；
  2. 相机适配只按“模型包围盒”计算（不再被天空盒等大尺寸物体带偏），
     并新增 适应/正视/俯视/等轴测/重置 等视图命令；
  3. 坐标轴改为左下角的 vtkOrientationMarkerWidget（不再跟着模型包围盒飘）；
  4. 右侧改成“场景面板 + 可滚动的插件面板（可折叠）”；
  5. 模型加载时自动做几何预处理：补法线、居中、统一尺寸、自动摆正（Z 轴朝上）。
作者：蔡洪斌（内核），多模型与交互改造
'''
import math
import os

import vtk

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, QFileDialog, QPushButton,
                             QTextEdit, QTreeView, QFileSystemModel, QSplitter, QMenu,
                             QAction, QGroupBox, QScrollArea, QListWidget, QDoubleSpinBox,
                             QLabel, QGridLayout, QFrame)
from PyQt5.QtCore import QDir, Qt, QEvent, QObject


class _ViewportClickFilter(QObject):
    """在 Qt 层截获鼠标点击，用来做视口点选。

    为什么不直接监听 VTK 的 LeftButtonReleaseEvent：
    实测本机 VTK 里 vtkRenderWindowInteractor.LeftButtonReleaseEvent() 不会触发
    观察者（只有 InvokeEvent 才会），所以 QVTK 转发过来的“松开”收不到，
    点选永远不会发生。改为在 Qt 侧拦截鼠标按下/松开，再换算成 VTK 的显示坐标。
    """

    def __init__(self, core):
        super().__init__(core)
        self.core = core

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self.core._press_qt_pos = event.pos()
        elif etype == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            press = getattr(self.core, "_press_qt_pos", None)
            self.core._press_qt_pos = None
            if press is not None:
                moved = (event.pos() - press).manhattanLength()
                if moved <= 5:            # 几乎没移动，才算“点选”，拖动仍是旋转
                    self.core._pick_at_qt(event.pos())
        return False


class ViewerContext:
    """全局上下文，插件共享状态"""

    def __init__(self):
        self.main_window = None
        self.vtk_renderer = None
        self.vtk_widget = None
        self.info_display = None
        self.right_extension_panel = None
        self.right_extension_layout = None
        self.current_obj_path = None
        self.current_actor = None
        self.current_polydata = None
        self.models = []          # [{id, path, actor, polydata, ...}]
        self.plugins = []


class VTKViewerCore(QMainWindow):
    MODEL_TARGET_SIZE = 2.0        # 统一尺寸后模型包围盒对角线长度
    LAYOUT_SPACING = 3.0           # 自动排布时模型中心间距

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Model Viewer Framework")
        self.setGeometry(100, 100, 1600, 880)

        self.ctx = ViewerContext()
        self.ctx.main_window = self
        self.plugin_instances = []
        self._model_seq = 0

        self.normalize_size = True      # 统一模型尺寸
        self.auto_upright = True        # 自动摆正（这些 OBJ 多为 Z 轴朝上）
        self.show_axes = True
        self.show_grid = True

        self.axes_actor = None
        self.orientation_widget = None
        self.ground_actor = None

        self.create_menu()
        self.build_base_layout()
        self.init_vtk()

    # ================================================================ 界面
    def create_menu(self):
        menubar = self.menuBar()

        file_menu = QMenu("File", self)
        open_folder_action = QAction("Open Folder", self)
        open_folder_action.triggered.connect(self.open_folder)
        clear_action = QAction("清空场景", self)
        clear_action.triggered.connect(self.clear_scene)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_folder_action)
        file_menu.addAction(clear_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        menubar.addMenu(file_menu)

        view_menu = QMenu("视图", self)
        act_fit = QAction("适配全部 (F)", self)
        act_fit.setShortcut("F")
        act_fit.triggered.connect(lambda: self.fit_camera())
        act_front = QAction("正视 (1)", self)
        act_front.setShortcut("1")
        act_front.triggered.connect(lambda: self.set_view("front"))
        act_top = QAction("俯视 (2)", self)
        act_top.setShortcut("2")
        act_top.triggered.connect(lambda: self.set_view("top"))
        act_iso = QAction("等轴测 (3)", self)
        act_iso.setShortcut("3")
        act_iso.triggered.connect(lambda: self.set_view("iso"))
        act_reset = QAction("重置视角 (R)", self)
        act_reset.setShortcut("R")
        act_reset.triggered.connect(lambda: self.set_view("iso"))

        self.act_upright = QAction("自动摆正（Z 轴朝上）", self)
        self.act_upright.setCheckable(True)
        self.act_upright.setChecked(True)
        self.act_upright.toggled.connect(self.set_auto_upright)

        self.act_normalize = QAction("统一模型尺寸", self)
        self.act_normalize.setCheckable(True)
        self.act_normalize.setChecked(True)
        self.act_normalize.toggled.connect(self.set_normalize_size)

        self.act_axes = QAction("显示坐标轴", self)
        self.act_axes.setCheckable(True)
        self.act_axes.setChecked(True)
        self.act_axes.toggled.connect(self.set_axes_visible)

        view_menu.addAction(act_fit)
        view_menu.addSeparator()
        view_menu.addAction(act_front)
        view_menu.addAction(act_top)
        view_menu.addAction(act_iso)
        view_menu.addAction(act_reset)
        view_menu.addSeparator()
        view_menu.addAction(self.act_upright)
        view_menu.addAction(self.act_normalize)
        view_menu.addAction(self.act_axes)

        self.act_grid = QAction("显示地面网格", self)
        self.act_grid.setCheckable(True)
        self.act_grid.setChecked(True)
        self.act_grid.toggled.connect(self.set_grid_visible)
        view_menu.addAction(self.act_grid)
        menubar.addMenu(view_menu)

    def build_base_layout(self):
        """基础布局：文件树 | VTK 窗口 | 右侧（场景面板 + 插件滚动区）"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_v_layout = QVBoxLayout(central_widget)
        main_v_layout.setSpacing(4)

        top_h_splitter = QSplitter(Qt.Horizontal)
        main_v_layout.addWidget(top_h_splitter, stretch=8)

        self.ctx.info_display = QTextEdit()
        self.ctx.info_display.setReadOnly(True)
        self.ctx.info_display.setMaximumHeight(140)
        main_v_layout.addWidget(self.ctx.info_display, stretch=1)

        # 左侧文件树
        self.tree_view = QTreeView()
        self.file_model = QFileSystemModel()
        self.file_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files)
        self.file_model.setNameFilters(["*.obj"])
        self.file_model.setNameFilterDisables(False)
        self.tree_view.setModel(self.file_model)
        self.tree_view.clicked.connect(self.on_tree_item_clicked)
        for col in range(1, self.file_model.columnCount()):
            self.tree_view.hideColumn(col)
        self.file_model.setRootPath("")
        self.tree_view.setRootIndex(self.file_model.index(""))
        top_h_splitter.addWidget(self.tree_view)

        # 中间 VTK 窗口
        self.ctx.vtk_widget = QVTKRenderWindowInteractor(self)
        self.ctx.vtk_renderer = vtk.vtkRenderer()
        self.ctx.vtk_renderer.SetBackground(0.16, 0.16, 0.19)
        self.ctx.vtk_widget.GetRenderWindow().AddRenderer(self.ctx.vtk_renderer)
        top_h_splitter.addWidget(self.ctx.vtk_widget)

        # 右侧：场景面板 + 插件滚动区
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)

        right_layout.addWidget(self._build_scene_panel())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        self.ctx.right_extension_layout = QVBoxLayout(content)
        self.ctx.right_extension_layout.setContentsMargins(0, 0, 0, 0)
        self.ctx.right_extension_layout.setSpacing(6)
        self.ctx.right_extension_layout.addStretch(1)
        scroll.setWidget(content)
        right_layout.addWidget(scroll, 1)
        self.ctx.right_extension_panel = scroll

        self.right_container = right_container
        top_h_splitter.addWidget(right_container)

        top_h_splitter.setSizes([int(0.20 * self.width()), int(0.58 * self.width()),
                                 int(0.22 * self.width())])
        top_h_splitter.setStretchFactor(1, 6)
        top_h_splitter.setStretchFactor(2, 4)

    def _build_scene_panel(self):
        """右侧顶部的“场景 / 模型”面板：模型列表 + 位置微调 + 常用操作"""
        box = QGroupBox("场景 / 模型")
        grid = QGridLayout(box)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(4)

        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(92)
        self.model_list.currentRowChanged.connect(self._on_model_selected)
        grid.addWidget(self.model_list, 0, 0, 1, 6)

        self.spin_x = self._make_spin()
        self.spin_y = self._make_spin()
        self.spin_z = self._make_spin()
        grid.addWidget(QLabel("X"), 1, 0)
        grid.addWidget(self.spin_x, 1, 1)
        grid.addWidget(QLabel("Y"), 1, 2)
        grid.addWidget(self.spin_y, 1, 3)
        grid.addWidget(QLabel("Z"), 1, 4)
        grid.addWidget(self.spin_z, 1, 5)

        btn_arrange = QPushButton("自动排列")
        btn_arrange.clicked.connect(self.arrange_models)
        btn_delete = QPushButton("删除选中")
        btn_delete.clicked.connect(self.delete_selected_model)
        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self.clear_scene)
        btn_fit = QPushButton("适配视图")
        btn_fit.clicked.connect(lambda: self.fit_camera())

        grid.addWidget(btn_arrange, 2, 0, 1, 2)
        grid.addWidget(btn_delete, 2, 2, 1, 2)
        grid.addWidget(btn_clear, 2, 4, 1, 1)
        grid.addWidget(btn_fit, 2, 5, 1, 1)
        return box

    def _make_spin(self):
        spin = QDoubleSpinBox()
        spin.setRange(-100.0, 100.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.2)
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(self._apply_selected_position)
        return spin

    # ================================================================ VTK 初始化
    def init_vtk(self):
        self.ctx.vtk_widget.Initialize()
        self.ctx.vtk_widget.Start()
        self._setup_interactor_style()
        self._setup_orientation_marker()
        self._build_ground_grid()
        self._setup_picking()

    def _setup_interactor_style(self):
        """轨道球交互：调低旋转灵敏度，便于精确摆到想要的角度"""
        try:
            style = vtk.vtkInteractorStyleTrackballCamera()
            style.SetMotionFactor(0.6)
            style.SetMouseWheelMotionFactor(1.0)
            self.ctx.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        except Exception as e:
            print(f"[内核] 交互样式设置失败: {e}")

    def _setup_orientation_marker(self):
        """左下角常驻坐标轴（不随模型大小/位置变化，也不再遮挡模型）"""
        axes = vtk.vtkAxesActor()
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.03)
        axes.SetConeRadius(0.12)
        axes.SetNormalizedShaftLength(0.85, 0.85, 0.85)
        axes.SetNormalizedTipLength(0.15, 0.15, 0.15)
        axes.SetNormalizedLabelPosition(1.15, 1.15, 1.15)
        for cap in (axes.GetXAxisCaptionActor2D(),
                    axes.GetYAxisCaptionActor2D(),
                    axes.GetZAxisCaptionActor2D()):
            prop = cap.GetCaptionTextProperty()
            prop.SetFontSize(11)
            prop.ShadowOff()
            prop.BoldOff()
        self.axes_actor = axes

        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(axes)
        self.orientation_widget.SetInteractor(
            self.ctx.vtk_widget.GetRenderWindow().GetInteractor())
        self.orientation_widget.SetViewport(0.0, 0.0, 0.15, 0.18)
        self.orientation_widget.SetEnabled(1)
        self.orientation_widget.InteractiveOff()

    def _build_ground_grid(self):
        """地面参考网格：给用户一个地平线参照，旋转/摆正时更容易判断方向"""
        src = vtk.vtkPlaneSource()
        src.SetOrigin(-1.0, 0.0, -1.0)
        src.SetPoint1(1.0, 0.0, -1.0)
        src.SetPoint2(-1.0, 0.0, 1.0)
        src.SetXResolution(20)
        src.SetYResolution(20)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetRepresentationToWireframe()
        prop.SetColor(0.46, 0.48, 0.56)
        prop.SetLighting(False)
        actor.PickableOff()
        actor.SetVisibility(0)
        self.ground_actor = actor
        self.ctx.vtk_renderer.AddActor(actor)

    def _update_ground(self):
        """网格放在所有模型的最低处，尺寸跟随场景"""
        if self.ground_actor is None:
            return
        bounds = self.model_bounds()
        if bounds is None:
            self.ground_actor.SetVisibility(0)
            return
        width = max(bounds[1] - bounds[0], bounds[5] - bounds[4], 2.0)
        scale = width * 0.75 + 0.5
        self.ground_actor.SetScale(scale, 1.0, scale)
        self.ground_actor.SetPosition((bounds[0] + bounds[1]) / 2.0,
                                      bounds[2] - 0.02,
                                      (bounds[4] + bounds[5]) / 2.0)
        self.ground_actor.SetVisibility(1 if self.show_grid else 0)

    def _setup_picking(self):
        """在视口里点击模型即选中；鼠标事件在 Qt 层拦截（见 _ViewportClickFilter）"""
        self._press_qt_pos = None
        self._click_filter = _ViewportClickFilter(self)
        self.ctx.vtk_widget.installEventFilter(self._click_filter)

        # 选中高亮：橙色包围盒线框，独立 actor，不会被插件改材质影响
        outline = vtk.vtkOutlineSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(outline.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(1.0, 0.62, 0.10)
        prop.SetLineWidth(2.0)
        prop.SetLighting(False)
        actor.PickableOff()
        actor.SetVisibility(0)
        self._outline_source = outline
        self.selection_actor = actor
        self.ctx.vtk_renderer.AddActor(actor)

    def _pick_at_qt(self, qpoint):
        """把 Qt 控件坐标换算成 VTK 显示坐标后拾取（考虑高 DPI 缩放）"""
        widget = self.ctx.vtk_widget
        try:
            size = widget.GetRenderWindow().GetSize()
        except Exception:
            return
        w = max(widget.width(), 1)
        h = max(widget.height(), 1)
        sx = size[0] / float(w)
        sy = size[1] / float(h)
        x = qpoint.x() * sx
        y = (h - qpoint.y() - 1) * sy
        self._pick_at(x, y)

    def _pick_at(self, x, y):
        """按屏幕坐标拾取模型；命中则同步选中列表行并高亮"""
        picker = vtk.vtkPropPicker()
        picker.Pick(int(x), int(y), 0, self.ctx.vtk_renderer)
        prop = picker.GetViewProp()
        for row, model in enumerate(self.ctx.models):
            if model["actor"] is prop:
                if self.model_list.currentRow() == row:
                    self._update_selection_outline(row)
                else:
                    self.model_list.setCurrentRow(row)   # 触发 _on_model_selected
                self.ctx.info_display.append(f"[场景] 已选中：{os.path.basename(model['path'])}")
                return model
        return None

    def _update_selection_outline(self, row=-1):
        actor = getattr(self, "selection_actor", None)
        if actor is None:
            return
        if not (0 <= row < len(self.ctx.models)):
            actor.SetVisibility(0)
            return
        bounds = self.ctx.models[row]["actor"].GetBounds()
        self._outline_source.SetBounds(bounds)
        actor.SetVisibility(1)

    def set_grid_visible(self, checked):
        self.show_grid = bool(checked)
        self._update_ground()
        self.render()

    # ================================================================ 插件注册
    def register_plugin(self, plugin_cls):
        plugin = plugin_cls(self.ctx)
        self.plugin_instances.append(plugin)
        self.ctx.plugins.append(plugin)

        panel = plugin.create_ui()
        if not panel:
            return
        if panel.width() == 0 and panel.height() == 0:
            # 诸如工具栏插件返回 0x0 占位控件，不放进右侧面板
            return

        box = QGroupBox(self._plugin_title(plugin_cls))
        box.setCheckable(True)
        box.setChecked(True)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(panel)
        box.toggled.connect(lambda checked, b=box: self._set_group_collapsed(b, checked))
        self._insert_right_widget(box)

    @staticmethod
    def _plugin_title(plugin_cls):
        """分组标题优先用中文名，其次中文描述，最后才用英文标识"""
        for attr in ("plugin_name", "plugin_desc"):
            value = getattr(plugin_cls, attr, "") or ""
            if any("\u4e00" <= ch <= "\u9fff" for ch in value):
                return value
        return getattr(plugin_cls, "plugin_name", "") or getattr(plugin_cls, "plugin_cls_name", "插件")

    def _insert_right_widget(self, widget):
        layout = self.ctx.right_extension_layout
        layout.insertWidget(max(0, layout.count() - 1), widget)

    @staticmethod
    def _set_group_collapsed(box, checked):
        """折叠分组：只隐藏内容，保留标题栏，点标题即可展开"""
        for i in range(box.layout().count()):
            item = box.layout().itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setVisible(checked)

    # ================================================================ 模型管理
    def on_tree_item_clicked(self, index):
        file_path = self.file_model.filePath(index)
        if not (os.path.isfile(file_path) and file_path.endswith(".obj")):
            return
        # 已经加载过的模型：只切换选中并适配视图，避免重复叠加
        for row, model in enumerate(self.ctx.models):
            if os.path.normcase(model["path"]) == os.path.normcase(file_path):
                self.model_list.setCurrentRow(row)
                self.fit_camera()
                self.render()
                self.ctx.info_display.append(f"[场景] 该模型已在场景中：{os.path.basename(file_path)}")
                return
        self.add_model(file_path)

    def load_model(self, file_name):
        """兼容旧接口：加载即追加一个模型"""
        return self.add_model(file_name) is not None

    def add_model(self, file_name):
        """加载模型并**追加**到场景（不再清空已有模型）"""
        vtk.vtkObject.GlobalWarningDisplayOff()
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        polydata = reader.GetOutput()

        if not polydata or not polydata.GetPoints() or polydata.GetPoints().GetNumberOfPoints() == 0:
            self.ctx.info_display.append(f"[错误] 无法解析该模型或模型为空: {file_name}")
            return None

        # 许多 OBJ 没有法线，缺了就补一份，否则光照/高光都不正常
        if polydata.GetPointData().GetNormals() is None:
            normal = vtk.vtkPolyDataNormals()
            normal.SetInputData(polydata)
            normal.SplittingOff()
            normal.ConsistencyOn()
            normal.Update()
            polydata = normal.GetOutput()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarVisibility(False)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.7)
        actor.PickableOn()

        bounds = polydata.GetBounds()
        diag = polydata.GetLength() or 1.0
        self._model_seq += 1
        model = {
            "id": self._model_seq,
            "path": file_name,
            "actor": actor,
            "polydata": polydata,
            "diag": diag,
            "raw_bounds": bounds,
            "raw_center": ((bounds[0] + bounds[1]) / 2.0,
                           (bounds[2] + bounds[3]) / 2.0,
                           (bounds[4] + bounds[5]) / 2.0),
            "pos": [0.0, 0.0, 0.0],
        }
        self._recompute_transform(model)
        model["pos"] = self._auto_slot()
        self._apply_transform(model)

        self.ctx.models.append(model)
        self.ctx.vtk_renderer.AddActor(actor)

        self.ctx.current_obj_path = file_name
        self.ctx.current_actor = actor
        self.ctx.current_polydata = polydata

        self._refresh_model_list(select_last=True)
        self.update_info(file_name, polydata.GetNumberOfPoints(), polydata.GetNumberOfCells())

        for plugin in self.plugin_instances:
            try:
                plugin.on_model_loaded(file_name, actor, polydata)
            except Exception as e:
                print(f"[插件错误] {plugin}: {e}")

        if len(self.ctx.models) == 1:
            self.fit_camera(direction=(-0.5, -0.55, -0.67), up=(0.0, 1.0, 0.0))
        else:
            self.fit_camera()
        self._update_ground()
        self.render()
        return model

    def _recompute_transform(self, model):
        """按“统一尺寸 / 自动摆正”两个开关重算缩放与中心"""
        diag = model["diag"]
        cx, cy, cz = model["raw_center"]
        scale = (self.MODEL_TARGET_SIZE / diag) if self.normalize_size else 1.0
        model["scale"] = scale
        if self.auto_upright:
            # 这些资产是 Z 轴朝上，绕 X 轴 -90° 转到 VTK 的 Y 轴朝上
            model["upright"] = True
            model["center"] = (cx, cz, -cy)
            model["actor"].SetOrientation(-90.0, 0.0, 0.0)
        else:
            model["upright"] = False
            model["center"] = (cx, cy, cz)
            model["actor"].SetOrientation(0.0, 0.0, 0.0)

    def _apply_transform(self, model):
        x, y, z = model["pos"]
        scale = model["scale"]
        cx, cy, cz = model["center"]
        actor = model["actor"]
        actor.SetScale(scale, scale, scale)
        actor.SetPosition(x - cx * scale, y - cy * scale, z - cz * scale)

    def _auto_slot(self):
        n = len(self.ctx.models)
        spacing = self.LAYOUT_SPACING
        if not self.normalize_size and self.ctx.models:
            spacing = max(spacing, max(m["diag"] for m in self.ctx.models) * 1.3)
        col = n % 3
        row = n // 3
        return [(col - 1) * spacing, -row * spacing, 0.0]

    def arrange_models(self):
        """重新按网格排布所有模型"""
        for i, model in enumerate(self.ctx.models):
            spacing = self.LAYOUT_SPACING
            if not self.normalize_size:
                spacing = max(spacing, max(m["diag"] for m in self.ctx.models) * 1.3)
            col, row = i % 3, i // 3
            model["pos"] = [(col - 1) * spacing, -row * spacing, 0.0]
            self._apply_transform(model)
        self._refresh_model_list()
        self.fit_camera()
        self._update_ground()
        self._update_selection_outline(self.model_list.currentRow())

    def remove_model(self, model):
        if model is None or model not in self.ctx.models:
            return
        self.ctx.vtk_renderer.RemoveActor(model["actor"])
        self.ctx.models.remove(model)
        if self.ctx.current_actor is model["actor"]:
            last = self.ctx.models[-1] if self.ctx.models else None
            self.ctx.current_actor = last["actor"] if last else None
            self.ctx.current_polydata = last["polydata"] if last else None
            self.ctx.current_obj_path = last["path"] if last else None
        self._refresh_model_list()
        self.fit_camera()
        self._update_ground()
        self._update_selection_outline(self.model_list.currentRow())
        self.render()

    def delete_selected_model(self):
        row = self.model_list.currentRow()
        if 0 <= row < len(self.ctx.models):
            self.remove_model(self.ctx.models[row])

    def clear_scene(self):
        for model in list(self.ctx.models):
            self.ctx.vtk_renderer.RemoveActor(model["actor"])
        self.ctx.models.clear()
        self.ctx.current_actor = None
        self.ctx.current_polydata = None
        self.ctx.current_obj_path = None
        self._refresh_model_list()
        self.fit_camera()
        self._update_ground()
        self._update_selection_outline(-1)
        self.render()
        self.ctx.info_display.append("[场景] 已清空所有模型")

    def _refresh_model_list(self, select_last=False):
        if not hasattr(self, "model_list"):
            return
        self.model_list.blockSignals(True)
        self.model_list.clear()
        for model in self.ctx.models:
            self.model_list.addItem(f"{model['id']}. {os.path.basename(model['path'])}")
        self.model_list.blockSignals(False)
        if self.ctx.models:
            row = len(self.ctx.models) - 1 if select_last else min(
                self.model_list.currentRow(), len(self.ctx.models) - 1)
            if row < 0:
                row = 0
            self.model_list.setCurrentRow(row)
            self._on_model_selected(row)

    def _on_model_selected(self, row):
        if not (0 <= row < len(self.ctx.models)):
            return
        model = self.ctx.models[row]
        self.ctx.current_actor = model["actor"]
        self.ctx.current_polydata = model["polydata"]
        self.ctx.current_obj_path = model["path"]
        for spin, value in ((self.spin_x, model["pos"][0]),
                            (self.spin_y, model["pos"][1]),
                            (self.spin_z, model["pos"][2])):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._update_selection_outline(row)

    def _apply_selected_position(self, _value=0.0):
        row = self.model_list.currentRow()
        if not (0 <= row < len(self.ctx.models)):
            return
        model = self.ctx.models[row]
        model["pos"] = [self.spin_x.value(), self.spin_y.value(), self.spin_z.value()]
        self._apply_transform(model)
        self._update_selection_outline(row)
        self.render()

    # ================================================================ 相机
    def model_bounds(self):
        """只统计模型，忽略天空盒/对比平面等插件物体"""
        if not self.ctx.models:
            return None
        b = [1e30, -1e30, 1e30, -1e30, 1e30, -1e30]
        for model in self.ctx.models:
            mb = model["actor"].GetBounds()
            b[0] = min(b[0], mb[0])
            b[1] = max(b[1], mb[1])
            b[2] = min(b[2], mb[2])
            b[3] = max(b[3], mb[3])
            b[4] = min(b[4], mb[4])
            b[5] = max(b[5], mb[5])
        return b

    def fit_camera(self, direction=None, up=None):
        """按模型包围盒适配相机，方向可指定（默认保持当前朝向）"""
        renderer = self.ctx.vtk_renderer
        camera = renderer.GetActiveCamera()
        bounds = self.model_bounds()

        if bounds is None:
            camera.SetFocalPoint(0.0, 0.0, 0.0)
            camera.SetPosition(0.0, 0.0, 10.0)
            camera.SetViewUp(0.0, 1.0, 0.0)
            renderer.ResetCameraClippingRange()
            return

        center = ((bounds[0] + bounds[1]) / 2.0,
                  (bounds[2] + bounds[3]) / 2.0,
                  (bounds[4] + bounds[5]) / 2.0)
        radius = 0.5 * math.sqrt((bounds[1] - bounds[0]) ** 2 +
                                 (bounds[3] - bounds[2]) ** 2 +
                                 (bounds[5] - bounds[4]) ** 2)
        radius = max(radius, 0.2)

        if direction is None:
            d = camera.GetDirectionOfProjection()
            if abs(d[0]) + abs(d[1]) + abs(d[2]) < 1e-6:
                d = (0.0, 0.0, -1.0)
        else:
            d = direction
        norm = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) or 1.0
        d = (d[0] / norm, d[1] / norm, d[2] / norm)

        fov = math.radians(max(camera.GetViewAngle(), 5.0))
        distance = radius / math.sin(fov / 2.0) * 1.25

        camera.SetFocalPoint(*center)
        camera.SetPosition(center[0] - d[0] * distance,
                           center[1] - d[1] * distance,
                           center[2] - d[2] * distance)
        if up is not None:
            camera.SetViewUp(*up)
        else:
            current_up = camera.GetViewUp()
            if abs(current_up[0] * d[0] + current_up[1] * d[1] + current_up[2] * d[2]) > 0.99:
                camera.SetViewUp(0.0, 1.0, 0.0)
        camera.OrthogonalizeViewUp()
        renderer.ResetCameraClippingRange()

    def set_view(self, kind):
        if kind == "front":
            self.fit_camera(direction=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0))
        elif kind == "top":
            self.fit_camera(direction=(0.0, -1.0, 0.0), up=(0.0, 0.0, -1.0))
        else:   # 等轴测
            self.fit_camera(direction=(-0.57, -0.57, -0.59), up=(0.0, 1.0, 0.0))
        self.render()

    def set_auto_upright(self, checked):
        self.auto_upright = bool(checked)
        for model in self.ctx.models:
            self._recompute_transform(model)
            self._apply_transform(model)
        self.fit_camera()
        self._update_ground()
        self.render()

    def set_normalize_size(self, checked):
        self.normalize_size = bool(checked)
        for model in self.ctx.models:
            self._recompute_transform(model)
        self.arrange_models()
        self._update_ground()

    def set_axes_visible(self, checked):
        self.show_axes = bool(checked)
        if self.orientation_widget is not None:
            self.orientation_widget.SetEnabled(1 if self.show_axes else 0)
        self.render()

    # ================================================================ 杂项
    def render(self):
        if self.ctx.vtk_widget is not None:
            self.ctx.vtk_widget.GetRenderWindow().Render()

    def request_render(self):
        self.render()

    def update_info(self, obj_file, pts, faces):
        total = len(self.ctx.models)
        text = (f"OBJ File: {obj_file}\nVertices: {pts}\nFaces: {faces}\n"
                f"场景内模型数: {total}\n")
        self.ctx.info_display.setPlainText(text)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder", "")
        if folder:
            self.file_model.setRootPath(folder)
            self.tree_view.setRootIndex(self.file_model.index(folder))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for plugin in self.plugin_instances:
            plugin.on_window_resize()

    def closeEvent(self, event):
        for plugin in self.plugin_instances:
            plugin.on_destroy()
        event.accept()
