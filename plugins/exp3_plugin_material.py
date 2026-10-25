"""
实验三 进阶要求：不同模型表现出其真实的材质（金属、木质、陶瓷、玻璃……）
=============================================================================
在框架里提供一套基于 PBR（基于物理的渲染）的材质库：

  * 材质库：黄金、抛光钢、铜、木质、花岗岩、砖石、陶瓷、塑料、橡胶、水面/玻璃，
    每种材质预设 颜色 + 金属度 + 粗糙度 + 可选的底色贴图（data/texture 里现成的图）；
  * 自动匹配：加载模型时按文件名关键字推断材质（chair → 木质，vase → 陶瓷，
    bottle → 玻璃，car/airplane/lamp → 金属……）；
  * 示例场景：一键把 5 个模型加入场景（内核多模型），各自套用不同的真实材质；
  * 微调：金属度 / 粗糙度滑块改变当前材质的光泽。

金属要环境反射才像金属：插件会自动生成一张程序化天空作为 IBL 来源；
若“环境映射”插件已启用，则直接复用它放好的全景天空盒。

插件标识：exp3_material（在 plugin_config.yaml 中启用）
=============================================================================
"""
import os

import vtk
from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                             QPushButton, QCheckBox, QSlider)
from PyQt5.QtCore import Qt

from plugins.base_plugin import Base3DPlugin

MATERIALS = [
    dict(name="黄金", color=(1.00, 0.77, 0.34), metallic=1.0, roughness=0.12,
         texture="metal_gold.png"),
    dict(name="抛光钢", color=(0.73, 0.75, 0.79), metallic=1.0, roughness=0.06,
         texture=None),
    dict(name="铜", color=(0.95, 0.60, 0.40), metallic=1.0, roughness=0.20,
         texture=None),
    dict(name="木质", color=(0.62, 0.44, 0.26), metallic=0.0, roughness=0.62,
         texture="texture6.png"),
    dict(name="花岗岩", color=(0.66, 0.64, 0.62), metallic=0.0, roughness=0.78,
         texture="texture4.png"),
    dict(name="砖石", color=(0.76, 0.50, 0.42), metallic=0.0, roughness=0.80,
         texture="texture1.png"),
    dict(name="陶瓷", color=(0.93, 0.94, 0.96), metallic=0.0, roughness=0.12,
         texture=None),
    dict(name="塑料", color=(0.25, 0.45, 0.80), metallic=0.0, roughness=0.38,
         texture=None),
    dict(name="橡胶", color=(0.09, 0.09, 0.10), metallic=0.0, roughness=0.90,
         texture=None),
    dict(name="水面/玻璃", color=(0.55, 0.75, 0.85), metallic=0.0, roughness=0.04,
         texture="texture2.png", opacity=0.55),
]

AUTO_RULES = [
    (("chair", "bench", "bookshelf", "piano", "table", "desk", "shelf", "wood"), "木质"),
    (("vase", "cup", "flower_pot", "pot", "bowl", "teapot"), "陶瓷"),
    (("bottle", "glass", "jar", "window"), "水面/玻璃"),
    (("car", "airplane", "plane", "lamp", "range_hood", "bike", "knife"), "抛光钢"),
    (("person", "human", "man", "woman"), "塑料"),
]

DEMO_MODELS = [
    ("chair/chair_0977.obj", "木质"),
    ("vase/vase_0052.obj", "陶瓷"),
    ("bottle/bottle_0088.obj", "水面/玻璃"),
    ("lamp/lamp_0033.obj", "黄金"),
    ("flower_pot/flower_pot_0010.obj", "花岗岩"),
]


class Exp3MaterialPlugin(Base3DPlugin):
    """实验三进阶：PBR 真实材质库"""

    plugin_cls_name = "exp3_material"
    plugin_name = "实验三·真实材质"
    plugin_desc = "实验三进阶：真实材质（金属/木质/陶瓷…）"

    def __init__(self, viewer_ctx):
        super().__init__(viewer_ctx)
        base = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.normpath(os.path.join(base, ".."))
        self.tex_dir = os.path.join(self.base_dir, "data", "texture")
        self.obj_dir = os.path.join(self.base_dir, "data", "obj")

        self.texture_cache = {}
        self.fallback_env = None
        self.demo_models = []
        self.current_material = None
        self.last_actor = None

        self.combo_material = None
        self.chk_auto = None
        self.slider_metal = None
        self.slider_rough = None
        self.lbl_status = None

    # ------------------------------------------------------------------ UI
    def create_ui(self):
        self.plugin_panel = QFrame()
        layout = QVBoxLayout(self.plugin_panel)
        layout.setSpacing(5)
        layout.setContentsMargins(8, 6, 8, 6)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("材质"))
        self.combo_material = QComboBox()
        for mat in MATERIALS:
            self.combo_material.addItem(mat["name"])
        row1.addWidget(self.combo_material, 1)
        btn_apply = QPushButton("应用")
        btn_apply.setFixedWidth(52)
        btn_apply.clicked.connect(lambda _=0: self.apply_to_current(auto=False))
        row1.addWidget(btn_apply)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.chk_auto = QCheckBox("自动匹配")
        self.chk_auto.setChecked(False)
        self.chk_auto.setToolTip("按文件名关键字推断材质，例如 chair→木质、vase→陶瓷")
        row2.addWidget(self.chk_auto)
        row2.addStretch(1)
        btn_demo = QPushButton("示例场景")
        btn_demo.setToolTip("把 5 个模型加入场景，每个套一种真实材质")
        btn_demo.clicked.connect(self.build_demo_scene)
        row2.addWidget(btn_demo)
        btn_clear = QPushButton("清空")
        btn_clear.setFixedWidth(52)
        btn_clear.clicked.connect(self.clear_demo)
        row2.addWidget(btn_clear)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("金属度"))
        self.slider_metal = QSlider(Qt.Horizontal)
        self.slider_metal.setRange(0, 100)
        self.slider_metal.setValue(0)
        self.slider_metal.valueChanged.connect(self.on_tune)
        row3.addWidget(self.slider_metal, 1)
        layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("粗糙度"))
        self.slider_rough = QSlider(Qt.Horizontal)
        self.slider_rough.setRange(0, 100)
        self.slider_rough.setValue(50)
        self.slider_rough.valueChanged.connect(self.on_tune)
        row4.addWidget(self.slider_rough, 1)
        layout.addLayout(row4)

        self.lbl_status = QLabel("加载模型会自动套用推断出的材质；“示例场景”可一键对比")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#555;")
        layout.addWidget(self.lbl_status)

        self._ensure_pbr_environment()
        return self.plugin_panel

    # ------------------------------------------------------ 环境光照（IBL）
    def _ensure_pbr_environment(self):
        """金属需要环境反射。场景里还没有环境贴图时，生成一张程序化天空作为 IBL。"""
        renderer = self.ctx.vtk_renderer
        if renderer is None:
            return
        try:
            if renderer.GetEnvironmentTexture() is not None:
                return
        except AttributeError:
            return
        if self.fallback_env is None:
            self.fallback_env = self._make_procedural_sky()
        try:
            renderer.SetEnvironmentTexture(self.fallback_env)
            renderer.UseImageBasedLightingOn()
        except AttributeError:
            pass

    @staticmethod
    def _make_procedural_sky(width=512, height=256):
        """等距柱状渐变天空（天空渐变 + 太阳 + 地面）"""
        import numpy as np
        from vtk.util.numpy_support import numpy_to_vtk

        u = (np.arange(width) + 0.5) / width
        v = (np.arange(height) + 0.5) / height
        lon = (u * 2.0 - 1.0) * np.pi
        lat = (0.5 - v) * np.pi
        ct, st = np.cos(lat)[:, None], np.sin(lat)[:, None]
        d = np.stack([ct * np.sin(lon)[None, :],
                      np.broadcast_to(st, (height, width)),
                      ct * np.cos(lon)[None, :]], axis=-1).astype(np.float32)

        y = d[..., 1]
        zenith = np.array([0.32, 0.42, 0.56], np.float32)
        horizon = np.array([0.80, 0.82, 0.84], np.float32)
        ground = np.array([0.24, 0.24, 0.22], np.float32)
        t_sky = np.clip(y, 0, 1)[..., None]
        t_gnd = np.clip(-y, 0, 1)[..., None]
        img = zenith * (1 - t_sky) + horizon * t_sky
        img = img * (1 - t_gnd) + ground * t_gnd

        sun = np.array([0.35, 0.55, 0.76], np.float32)
        sun /= np.linalg.norm(sun)
        glow = np.clip(np.sum(d * sun, axis=-1), 0, 1)[..., None] ** 40
        img = np.clip(img + glow * 4.0, 0, 1) * 255.0

        arr = img.astype(np.uint8)[::-1]        # VTK 图像行序自下而上
        vtk_arr = numpy_to_vtk(arr.reshape(-1, 3), deep=1,
                               array_type=vtk.VTK_UNSIGNED_CHAR)
        image = vtk.vtkImageData()
        image.SetDimensions(width, height, 1)
        image.GetPointData().SetScalars(vtk_arr)
        texture = vtk.vtkTexture()
        texture.SetInputData(image)
        texture.SetInterpolate(1)
        texture.SetMipmap(1)
        return texture

    # -------------------------------------------------------------- 材质
    def material_by_name(self, name):
        for mat in MATERIALS:
            if mat["name"] == name:
                return mat
        return MATERIALS[-1]

    def guess_material(self, obj_path):
        name = (obj_path or "").replace("\\", "/").split("/")[-1].lower()
        for keys, mat_name in AUTO_RULES:
            if any(k in name for k in keys):
                return self.material_by_name(mat_name)
        return self.material_by_name("塑料")

    def _get_texture(self, file_name):
        if not file_name:
            return None
        if file_name in self.texture_cache:
            return self.texture_cache[file_name]
        path = os.path.join(self.tex_dir, file_name)
        if not os.path.isfile(path):
            print(f"[真实材质] 找不到贴图: {path}")
            return None
        reader = vtk.vtkJPEGReader() if path.lower().endswith((".jpg", ".jpeg")) \
            else vtk.vtkPNGReader()
        reader.SetFileName(path)
        reader.Update()
        texture = vtk.vtkTexture()
        texture.SetInputConnection(reader.GetOutputPort())
        texture.SetInterpolate(1)
        texture.SetRepeat(1)
        texture.SetMipmap(1)
        self.texture_cache[file_name] = texture
        return texture

    def _prepare_geometry(self, actor, need_uv=False):
        """补点法线、需要时补圆柱投影 UV（内核已补法线，这里做兜底）"""
        mapper = actor.GetMapper()
        if mapper is None:
            return
        polydata = mapper.GetInput()
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            return

        changed = False
        if polydata.GetPointData().GetNormals() is None:
            normal = vtk.vtkPolyDataNormals()
            normal.SetInputData(polydata)
            normal.SplittingOff()
            normal.ConsistencyOn()
            normal.Update()
            polydata = normal.GetOutput()
            changed = True

        if need_uv and polydata.GetPointData().GetTCoords() is None:
            gen = vtk.vtkTextureMapToCylinder()
            gen.SetInputData(polydata)
            gen.PreventSeamOn()
            gen.Update()
            polydata = gen.GetOutput()
            changed = True

        if changed:
            new_mapper = vtk.vtkPolyDataMapper()
            new_mapper.SetInputData(polydata)
            new_mapper.SetScalarVisibility(False)
            actor.SetMapper(new_mapper)

    def apply_material(self, actor, mat):
        prop = actor.GetProperty()
        prop.SetInterpolationToPBR()
        prop.SetColor(*mat["color"])
        prop.SetMetallic(mat["metallic"])
        prop.SetRoughness(mat["roughness"])
        prop.SetOpacity(mat.get("opacity", 1.0))

        texture = self._get_texture(mat.get("texture"))
        self._prepare_geometry(actor, need_uv=texture is not None)
        if texture is not None:
            prop.SetBaseColorTexture(texture)
        else:
            try:
                prop.SetBaseColorTexture(None)
            except Exception:
                pass

        self.current_material = mat
        self.last_actor = actor
        self.slider_metal.blockSignals(True)
        self.slider_rough.blockSignals(True)
        self.slider_metal.setValue(int(mat["metallic"] * 100))
        self.slider_rough.setValue(int(mat["roughness"] * 100))
        self.slider_metal.blockSignals(False)
        self.slider_rough.blockSignals(False)

    def apply_to_current(self, auto=False):
        actor = self.ctx.current_actor
        if actor is None:
            self._set_status("请先在左侧文件树点选一个模型")
            return
        if auto:
            mat = self.guess_material(self.ctx.current_obj_path)
            self.combo_material.setCurrentIndex(MATERIALS.index(mat))
        else:
            mat = MATERIALS[self.combo_material.currentIndex()]
        self.apply_material(actor, mat)
        self._set_status(f"已给 {os.path.basename(self.ctx.current_obj_path or '模型')} "
                         f"套用「{mat['name']}」材质")
        self._render()

    def on_tune(self, _value=0):
        if self.last_actor is None:
            return
        prop = self.last_actor.GetProperty()
        prop.SetMetallic(self.slider_metal.value() / 100.0)
        prop.SetRoughness(self.slider_rough.value() / 100.0)
        self._render()

    # ---------------------------------------------------------- 示例场景
    def build_demo_scene(self):
        """通过内核接口把 5 个模型加入场景，各自套一种材质"""
        window = getattr(self.ctx, "main_window", None)
        if window is None or not hasattr(window, "add_model"):
            self._set_status("当前内核不支持多模型接口")
            return
        self._ensure_pbr_environment()
        self.clear_demo()

        loaded = 0
        for rel_path, mat_name in DEMO_MODELS:
            path = os.path.join(self.obj_dir, rel_path.replace("/", os.sep))
            if not os.path.isfile(path):
                print(f"[真实材质] 示例模型缺失: {path}")
                continue
            model = window.add_model(path)
            if model is None:
                continue
            self.apply_material(model["actor"], self.material_by_name(mat_name))
            self.demo_models.append(model)
            loaded += 1

        if loaded:
            window.fit_camera()
            self._set_status(f"已把 {loaded} 个模型加入场景，各套一种真实材质")
        else:
            self._set_status("没有找到示例模型，请检查 data/obj 目录")
        self._render()

    def clear_demo(self):
        window = getattr(self.ctx, "main_window", None)
        if window is not None and hasattr(window, "remove_model"):
            for model in list(self.demo_models):
                window.remove_model(model)
        self.demo_models = []

    # ------------------------------------------------------------------ 钩子
    def on_model_loaded(self, obj_path, actor, polydata):
        if self.chk_auto.isChecked():
            self.apply_to_current(auto=True)
        self._render()

    def on_destroy(self):
        self.clear_demo()
        self.texture_cache.clear()

    # ------------------------------------------------------------------ 杂项
    def _render(self):
        if self.ctx.vtk_widget is not None:
            self.ctx.vtk_widget.GetRenderWindow().Render()

    def _set_status(self, text):
        if self.lbl_status is not None:
            self.lbl_status.setText(text)
        if self.ctx.info_display is not None:
            self.ctx.info_display.append(f"[真实材质] {text}")
