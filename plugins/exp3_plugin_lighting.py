"""
实验三 编程任务 2：多模型 + 光照与材质
=============================================================================
把实验三编程任务 1 的「3×4 球体矩阵」那 12 种光照/材质组合，套用到场景里
真实加载的 OBJ 模型上：

  * 场景里的每个模型都可以单独套一个预设（内核已支持多模型 + 位置调整）；
  * “循环应用到全部”会把 12 个预设依次分配给场景中的所有模型，一屏看全对比；
  * 灯光可调位置 (X/Y/Z)、颜色 (R/G/B)、强度，对应任务 1 的按键功能；
  * 高光指数可整体缩放，观察高光斑大小变化；
  * 光照开关：关闭后直接看模型的基础色。

12 个预设严格对应任务 1 的 3×4 矩阵：
    行 = 无环境光 / 有环境光 / 有颜色的环境光
    列 = 仅漫反射 / +低高光 / +很亮高光 / +自发光

说明：VTK 的 vtkProperty 没有独立的 emission 通道，但 PBR 插值模式提供
EmissiveFactor，因此“自发光”那一列用 PBR 材质 + EmissiveFactor 实现，
其余预设仍用经典 Phong 光照模型（逐像素高光）。

插件标识：exp3_lighting（在 plugin_config.yaml 中启用）
=============================================================================
"""
import vtk
from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
                             QComboBox, QPushButton, QCheckBox, QSlider)
from PyQt5.QtCore import Qt

from plugins.base_plugin import Base3DPlugin

ROW_DIFFUSE = [
    (0.10, 0.32, 0.72),
    (0.30, 0.52, 0.88),
    (0.20, 0.50, 0.52),
]
ROW_AMBIENT = [
    ((1.00, 1.00, 1.00), 0.00),   # 无环境光
    ((0.78, 0.86, 1.00), 0.35),   # 有环境光（偏冷的灰蓝）
    ((0.78, 0.70, 0.16), 0.55),   # 有颜色的环境光（橄榄色）
]
COL_SPEC = [
    (0.00, 1.0),
    (0.45, 18.0),
    (1.00, 110.0),
    (0.00, 1.0),
]
COL_EMISSION = (0.26, 0.09, 0.12)


def build_presets():
    row_title = ["无环境光", "有环境光", "有颜色的环境光"]
    col_title = ["仅漫反射", "+低高光", "+很亮高光", "+自发光"]
    presets = []
    for row in range(3):
        ambient_color, ambient_coef = ROW_AMBIENT[row]
        for col in range(4):
            specular, shininess = COL_SPEC[col]
            emission = COL_EMISSION if col == 3 else (0.0, 0.0, 0.0)
            presets.append({
                "name": f"{row + 1}-{col + 1} {row_title[row]} · {col_title[col]}",
                "ambient_color": ambient_color,
                "ambient_coef": ambient_coef,
                "diffuse": ROW_DIFFUSE[row],
                "specular": specular,
                "shininess": shininess,
                "emission": emission,
            })
    return presets


class Exp3LightingPlugin(Base3DPlugin):
    """实验三任务 2：多模型光照与材质对比"""

    plugin_cls_name = "exp3_lighting"
    plugin_name = "实验三·光照与材质"
    plugin_desc = "实验三任务2：12 种光照/材质预设，可逐模型应用"

    def __init__(self, viewer_ctx):
        super().__init__(viewer_ctx)
        self.presets = build_presets()
        self.scene_light = None
        self.highlight_scale = 1.0

        self.combo_preset = None
        self.sliders = {}
        self.chk_light = None
        self.lbl_status = None

    # ------------------------------------------------------------------ UI
    def create_ui(self):
        self.plugin_panel = QFrame()
        layout = QVBoxLayout(self.plugin_panel)
        layout.setSpacing(5)
        layout.setContentsMargins(8, 6, 8, 6)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("预设"))
        self.combo_preset = QComboBox()
        for preset in self.presets:
            self.combo_preset.addItem(preset["name"])
        row1.addWidget(self.combo_preset, 1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        btn_current = QPushButton("应用到当前模型")
        btn_current.clicked.connect(self.apply_to_current)
        row2.addWidget(btn_current)
        btn_all = QPushButton("循环全部")
        btn_all.setToolTip("把 12 个预设依次分配给场景中的所有模型")
        btn_all.clicked.connect(self.apply_to_all)
        row2.addWidget(btn_all)
        layout.addLayout(row2)

        # 灯光位置
        row3 = QGridLayout()
        row3.addWidget(QLabel("光位"), 0, 0)
        for i, key in enumerate(("x", "y", "z")):
            row3.addWidget(QLabel(key.upper()), 0, 1 + i * 2)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-80, 80)
            slider.setValue((-30, 40, 60)[i])
            slider.valueChanged.connect(self.on_light_changed)
            row3.addWidget(slider, 0, 2 + i * 2)
            self.sliders["light_" + key] = slider
        layout.addLayout(row3)

        # 灯光颜色
        row4 = QGridLayout()
        row4.addWidget(QLabel("光色"), 0, 0)
        for i, key in enumerate(("r", "g", "b")):
            row4.addWidget(QLabel(key.upper()), 0, 1 + i * 2)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(100)
            slider.valueChanged.connect(self.on_light_changed)
            row4.addWidget(slider, 0, 2 + i * 2)
            self.sliders["color_" + key] = slider
        layout.addLayout(row4)

        # 强度 + 高光
        row5 = QGridLayout()
        row5.addWidget(QLabel("强度"), 0, 0)
        s_int = QSlider(Qt.Horizontal)
        s_int.setRange(0, 150)
        s_int.setValue(110)
        s_int.valueChanged.connect(self.on_light_changed)
        row5.addWidget(s_int, 0, 1, 1, 2)
        self.sliders["intensity"] = s_int
        row5.addWidget(QLabel("高光×"), 0, 3)
        s_hi = QSlider(Qt.Horizontal)
        s_hi.setRange(20, 300)
        s_hi.setValue(100)
        s_hi.valueChanged.connect(self.on_highlight_changed)
        row5.addWidget(s_hi, 0, 4, 1, 2)
        self.sliders["highlight"] = s_hi
        layout.addLayout(row5)

        self.chk_auto_apply = QCheckBox("加载模型时自动套用预设")
        self.chk_auto_apply.setChecked(False)
        self.chk_auto_apply.setToolTip("默认关闭，避免和“真实材质”插件的 PBR 材质互相覆盖")
        layout.addWidget(self.chk_auto_apply)

        self.chk_light = QCheckBox("启用光照")
        self.chk_light.setChecked(True)
        self.chk_light.stateChanged.connect(self.on_light_toggle)
        layout.addWidget(self.chk_light)

        self.lbl_status = QLabel("加载模型后点“应用到当前模型”即可")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#555;")
        layout.addWidget(self.lbl_status)

        self.ensure_light()
        return self.plugin_panel

    # -------------------------------------------------------------- 灯光
    def ensure_light(self):
        renderer = self.ctx.vtk_renderer
        if renderer is None or self.scene_light is not None:
            return
        self.scene_light = vtk.vtkLight()
        self.scene_light.SetLightTypeToSceneLight()
        self.scene_light.SetFocalPoint(0.0, 0.0, 0.0)
        self.apply_light_settings()
        renderer.AddLight(self.scene_light)

    def apply_light_settings(self):
        if self.scene_light is None:
            return
        s = self.sliders
        self.scene_light.SetPosition(s["light_x"].value() / 10.0,
                                     s["light_y"].value() / 10.0,
                                     s["light_z"].value() / 10.0)
        self.scene_light.SetColor(s["color_r"].value() / 100.0,
                                  s["color_g"].value() / 100.0,
                                  s["color_b"].value() / 100.0)
        self.scene_light.SetIntensity(s["intensity"].value() / 100.0)
        self.scene_light.SetSwitch(1 if (self.chk_light and self.chk_light.isChecked()) else 0)

    def on_light_changed(self, _value=0):
        self.apply_light_settings()
        self._render()

    def on_light_toggle(self, _state=None):
        self.apply_light_settings()
        self._render()

    def on_highlight_changed(self, value):
        self.highlight_scale = value / 100.0
        for row, model in enumerate(self.ctx.models):
            self.apply_preset(model["actor"], model.get("preset_index", row))
        self._render()

    # -------------------------------------------------------------- 材质
    def apply_preset(self, actor, index):
        preset = self.presets[index % len(self.presets)]
        prop = actor.GetProperty()
        if preset["emission"] != (0.0, 0.0, 0.0):
            prop.SetInterpolationToPBR()
            prop.SetColor(*preset["diffuse"])
            prop.SetMetallic(0.0)
            prop.SetRoughness(0.55)
            prop.SetEmissiveFactor(*preset["emission"])
        else:
            prop.SetInterpolationToPhong()
            prop.SetColor(*preset["diffuse"])
            prop.SetAmbient(preset["ambient_coef"])
            prop.SetAmbientColor(*preset["ambient_color"])
            prop.SetDiffuse(0.90)
            prop.SetDiffuseColor(*preset["diffuse"])
            prop.SetSpecular(min(1.0, preset["specular"]))
            prop.SetSpecularColor(1.0, 1.0, 1.0)
            prop.SetSpecularPower(max(1.0, preset["shininess"] * self.highlight_scale))
            prop.SetEmissiveFactor(0.0, 0.0, 0.0)

    def _current_model(self):
        actor = self.ctx.current_actor
        for model in self.ctx.models:
            if model["actor"] is actor:
                return model
        return None

    def apply_to_current(self):
        model = self._current_model()
        if model is None:
            self._set_status("请先在左侧文件树点选一个模型")
            return
        index = self.combo_preset.currentIndex()
        self.apply_preset(model["actor"], index)
        model["preset_index"] = index
        self._set_status(f"已应用「{self.presets[index]['name']}」")
        self._render()

    def apply_to_all(self):
        if not self.ctx.models:
            self._set_status("场景里还没有模型")
            return
        for i, model in enumerate(self.ctx.models):
            self.apply_preset(model["actor"], i % len(self.presets))
            model["preset_index"] = i % len(self.presets)
        self._set_status(f"已把 12 个预设循环分配到 {len(self.ctx.models)} 个模型")
        self._render()

    # ------------------------------------------------------------------ 钩子
    def on_model_loaded(self, obj_path, actor, polydata):
        self.ensure_light()
        if not self.chk_auto_apply.isChecked():
            self._render()
            return
        index = self.combo_preset.currentIndex()
        self.apply_preset(actor, index)
        for model in self.ctx.models:
            if model["actor"] is actor:
                model["preset_index"] = index
                break
        self._render()

    def on_destroy(self):
        renderer = self.ctx.vtk_renderer
        if renderer is not None and self.scene_light is not None:
            renderer.RemoveLight(self.scene_light)
        self.scene_light = None

    # ------------------------------------------------------------------ 杂项
    def _render(self):
        if self.ctx.vtk_widget is not None:
            self.ctx.vtk_widget.GetRenderWindow().Render()

    def _set_status(self, text):
        if self.lbl_status is not None:
            self.lbl_status.setText(text)
        if self.ctx.info_display is not None:
            self.ctx.info_display.append(f"[光照材质] {text}")
