"""
实验二 进阶任务（二）：环境映射插件
==========================================================
在 3D 查看器框架里实现实验二编程任务 2 的环境映射：

  * 全景天空盒（cubemap + vtkSkybox）作为环境背景；
  * 金属反射球 / 反射立方体（PBR：metallic=1、roughness≈0），
    由渲染器的环境贴图（image based lighting）照亮并反射周边环境；
  * 面板可控制天空盒/球/立方体的显隐、金属度、粗糙度。

资源：data/texture/skybox/{px,nx,py,ny,pz,nz}.jpg|png，
      缺省时自动生成程序化天空（渐变 + 太阳 + 地面），保证插件随时可用。

插件标识：environment_mapping（在 plugin_config.yaml 中启用）
"""
import os

import vtk
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QCheckBox, QLabel,
                             QSlider, QPushButton, QFrame)
from PyQt5.QtCore import Qt

from plugins.base_plugin import Base3DPlugin

FACE_KEYS = ["px", "nx", "py", "ny", "pz", "nz"]
FACE_SIZE = 512          # cubemap 每个面的分辨率（6 面必须一致）


class EnvironmentMappingPlugin(Base3DPlugin):
    """天空盒 + PBR 环境映射（反射球 / 反射立方体）插件"""

    plugin_cls_name = "environment_mapping"
    plugin_name = "environment_mapping"
    plugin_desc = "环境映射插件（实验2 进阶二）"

    def __init__(self, viewer_ctx):
        super().__init__(viewer_ctx)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sky_dir = os.path.normpath(os.path.join(base_dir, "../data/texture/skybox"))

        self.env_texture = None
        self.skybox = None
        self.sphere_actor = None
        self.cube_actor = None
        self.ibl_enabled = False
        self.model_scale = 1.0

        self.chk_skybox = None
        self.chk_sphere = None
        self.chk_cube = None
        self.chk_model = None
        self.slider_metal = None
        self.slider_rough = None
        self.lbl_status = None
        self._model_saved = None      # 开启/关闭“模型反射”时用于还原材质

    # ------------------------------------------------------------------ UI
    def create_ui(self):
        self.plugin_panel = QFrame()
        self.plugin_panel.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self.plugin_panel)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("环境映射（实验2 进阶）")
        title.setStyleSheet("font-weight:bold;")
        layout.addWidget(title)

        self.chk_skybox = QCheckBox("显示全景天空盒")
        self.chk_skybox.setChecked(True)
        self.chk_skybox.stateChanged.connect(self.on_toggle)
        layout.addWidget(self.chk_skybox)

        self.chk_sphere = QCheckBox("显示反射球")
        self.chk_sphere.setChecked(False)
        self.chk_sphere.stateChanged.connect(self.on_toggle)
        layout.addWidget(self.chk_sphere)

        self.chk_cube = QCheckBox("显示反射立方体")
        self.chk_cube.setChecked(False)
        self.chk_cube.stateChanged.connect(self.on_toggle)
        layout.addWidget(self.chk_cube)

        # 金属度 / 粗糙度：PBR 材质参数，决定反射的“像不像镜子”
        row_metal = QHBoxLayout()
        row_metal.addWidget(QLabel("金属度"))
        self.slider_metal = QSlider(Qt.Horizontal)
        self.slider_metal.setRange(0, 100)
        self.slider_metal.setValue(100)
        self.slider_metal.valueChanged.connect(self.on_material_changed)
        row_metal.addWidget(self.slider_metal, 1)
        layout.addLayout(row_metal)

        row_rough = QHBoxLayout()
        row_rough.addWidget(QLabel("粗糙度"))
        self.slider_rough = QSlider(Qt.Horizontal)
        self.slider_rough.setRange(0, 100)
        self.slider_rough.setValue(3)
        self.slider_rough.valueChanged.connect(self.on_material_changed)
        row_rough.addWidget(self.slider_rough, 1)
        layout.addLayout(row_rough)

        # 让当前加载的 OBJ 模型本身也走 PBR 环境反射
        self.chk_model = QCheckBox("当前模型也反射环境（PBR）")
        self.chk_model.setChecked(False)
        self.chk_model.stateChanged.connect(self.on_model_reflect_toggled)
        layout.addWidget(self.chk_model)

        btn_reload = QPushButton("重新载入天空盒")
        btn_reload.clicked.connect(self.rebuild_environment)
        layout.addWidget(btn_reload)

        self.lbl_status = QLabel("环境贴图尚未载入")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#555;")
        layout.addWidget(self.lbl_status)

        # 面板建好后立刻准备环境（此时可能还没有模型，先放在原点）
        self._ensure_environment()
        return self.plugin_panel

    # ------------------------------------------------------------ 环境资源
    def _face_files(self):
        """返回 6 个面的图片路径；缺任意一面则返回 None"""
        if not os.path.isdir(self.sky_dir):
            return None
        paths = []
        for key in FACE_KEYS:
            found = None
            for name in sorted(os.listdir(self.sky_dir)):
                stem, ext = os.path.splitext(name)
                if stem.lower() == key and ext.lower() in (".jpg", ".jpeg", ".png"):
                    found = os.path.join(self.sky_dir, name)
                    break
            if found is None:
                return None
            paths.append(found)
        return paths

    def _read_face(self, path):
        """读取一个面：统一分辨率 + 垂直翻转（VTK 图像原点在左下）。"""
        if path.lower().endswith((".jpg", ".jpeg")):
            reader = vtk.vtkJPEGReader()
        else:
            reader = vtk.vtkPNGReader()
        reader.SetFileName(path)
        reader.Update()

        resize = vtk.vtkImageResize()
        resize.SetInputConnection(reader.GetOutputPort())
        resize.SetResizeMethodToOutputDimensions()
        resize.SetOutputDimensions(FACE_SIZE, FACE_SIZE, 1)
        flip = vtk.vtkImageFlip()
        flip.SetInputConnection(resize.GetOutputPort())
        flip.SetFilteredAxes(1)
        flip.Update()
        return flip.GetOutput()

    def _procedural_faces(self, size=128):
        """没有天空盒素材时，程序化生成 6 个面（天空渐变 + 太阳 + 地面）。"""
        import numpy as np
        from vtk.util.numpy_support import numpy_to_vtk

        s = (np.arange(size) + 0.5) / size * 2.0 - 1.0      # -1..1
        ss, tt = np.meshgrid(s, s)                          # ss 列, tt 行（行 0 = 上）
        dirs = {
            "px": (np.ones_like(ss), 1.0 - 2 * tt, 1.0 - 2 * ss),
            "nx": (-np.ones_like(ss), 1.0 - 2 * tt, 2 * ss - 1.0),
            "py": (2 * ss - 1.0, np.ones_like(ss), 2 * tt - 1.0),
            "ny": (2 * ss - 1.0, -np.ones_like(ss), -(2 * tt - 1.0)),
            "pz": (2 * ss - 1.0, 1.0 - 2 * tt, np.ones_like(ss)),
            "nz": (-(2 * ss - 1.0), 1.0 - 2 * tt, -np.ones_like(ss)),
        }
        sun = np.array([0.35, 0.55, 0.76])
        sun /= np.linalg.norm(sun)

        out = {}
        for key, d in dirs.items():
            v = np.stack(d, axis=-1).astype(np.float32)
            v /= np.linalg.norm(v, axis=-1, keepdims=True)
            y = v[..., 1]
            up = np.array([0.24, 0.45, 0.78], dtype=np.float32)
            horizon = np.array([0.72, 0.83, 0.93], dtype=np.float32)
            ground = np.array([0.33, 0.36, 0.30], dtype=np.float32)
            t_sky = np.clip(y, 0, 1)[..., None]
            t_gr = np.clip(-y, 0, 1)[..., None]
            img = up * (1 - t_sky) + horizon * t_sky
            img = img * (1 - t_gr) + ground * t_gr
            glow = np.clip(np.sum(v * sun, axis=-1), 0, 1)[..., None] ** 48
            img = img + glow * 3.0
            obj = np.clip(img, 0, 1) * 255.0
            arr = obj.astype(np.uint8)
            vtk_arr = numpy_to_vtk(arr.reshape(-1, 3), deep=1,
                                   array_type=vtk.VTK_UNSIGNED_CHAR)
            img_data = vtk.vtkImageData()
            img_data.SetDimensions(size, size, 1)
            img_data.GetPointData().SetScalars(vtk_arr)
            flip = vtk.vtkImageFlip()
            flip.SetInputData(img_data)
            flip.SetFilteredAxes(1)
            flip.Update()
            out[key] = flip.GetOutput()
        return out

    def _build_texture(self):
        """构造天空盒 cubemap 与 IBL 用的环境贴图。

        返回 (cube_texture, env_texture, 描述文字)。

        为什么 IBL 不直接用 cubemap：
        VTK 的球谐（SH）近似只能从 2D 图像计算，喂 cubemap 会打印
        “Cannot compute spherical harmonics of a cubemap, falling back to
        irradiance texture” 并退化成辐照度贴图。这里把 6 个面重采样成
        等距柱状（lat-long）图给 IBL 用，天空盒仍用 cubemap，
        于是既没有警告，漫反射环境光也更准确。"""
        cube_texture = vtk.vtkTexture()
        cube_texture.SetCubeMap(1)
        cube_texture.SetInterpolate(1)
        cube_texture.SetMipmap(1)
        cube_texture.SetMaximumAnisotropicFiltering(8)

        paths = self._face_files()
        if paths:
            faces = {}
            for key, path in zip(FACE_KEYS, paths):
                faces[key] = self._read_face(path)
            note = f"cubemap：{os.path.basename(self.sky_dir)}/*.jpg（{FACE_SIZE}²×6）"
        else:
            faces = self._procedural_faces()
            note = "cubemap：程序化生成（未找到 data/texture/skybox）"

        for i, key in enumerate(FACE_KEYS):
            cube_texture.SetInputDataObject(i, faces[key])

        env_texture = cube_texture
        try:
            equirect = self._build_equirect(faces)
            if equirect is not None:
                env_texture = vtk.vtkTexture()
                env_texture.SetInputData(equirect)
                env_texture.SetInterpolate(1)
                env_texture.SetMipmap(1)
                note += "；IBL 用等距柱状环境图"
        except Exception as e:
            print(f"[环境映射] 等距柱状环境图生成失败，IBL 退回 cubemap: {e}")
        return cube_texture, env_texture, note

    def _build_equirect(self, faces):
        """把 6 个 cubemap 面重采样成等距柱状（lat-long）环境图。

        方向 → 面内 (s,t) 的换算按 OpenGL cubemap 规范：
            +X: s=(1-rz)/2, t=(1-ry)/2      -X: s=(1+rz)/2, t=(1-ry)/2
            +Y: s=(1+rx)/2, t=(1+rz)/2      -Y: s=(1+rx)/2, t=(1-rz)/2
            +Z: s=(1+rx)/2, t=(1-ry)/2      -Z: s=(1-rx)/2, t=(1-ry)/2
        （其中 rx/ry/rz 要除以主轴的绝对值）"""
        import numpy as np
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

        size = faces[FACE_KEYS[0]].GetDimensions()[0]
        arrays = {}
        for key in FACE_KEYS:
            scalars = faces[key].GetPointData().GetScalars()
            arrays[key] = vtk_to_numpy(scalars).reshape(size, size, 3).astype(np.float32)

        w, h = 1024, 512
        u = (np.arange(w) + 0.5) / w
        v = (np.arange(h) + 0.5) / h
        lon = (u * 2.0 - 1.0) * np.pi
        lat = (0.5 - v) * np.pi
        ct, st = np.cos(lat)[:, None], np.sin(lat)[:, None]
        d = np.stack([ct * np.sin(lon)[None, :],
                      np.broadcast_to(st, (h, w)),
                      ct * np.cos(lon)[None, :]], axis=-1)
        ax = np.abs(d)
        major = np.argmax(ax, axis=-1)
        ma = np.take_along_axis(ax, major[..., None], axis=-1)[..., 0]
        comp = np.take_along_axis(d, major[..., None], axis=-1)[..., 0]

        out = np.zeros((h, w, 3), np.uint8)
        for idx, (kpos, kneg) in enumerate([("px", "nx"), ("py", "ny"), ("pz", "nz")]):
            for is_pos in (True, False):
                key = kpos if is_pos else kneg
                m = (major == idx) & ((comp > 0) == is_pos)
                if not m.any():
                    continue
                dx, dy, dz = d[..., 0][m], d[..., 1][m], d[..., 2][m]
                mm = ma[m]
                if key == "px":
                    s, t = (1 - dz / mm) / 2, (1 - dy / mm) / 2
                elif key == "nx":
                    s, t = (1 + dz / mm) / 2, (1 - dy / mm) / 2
                elif key == "py":
                    s, t = (1 + dx / mm) / 2, (1 + dz / mm) / 2
                elif key == "ny":
                    s, t = (1 + dx / mm) / 2, (1 - dz / mm) / 2
                elif key == "pz":
                    s, t = (1 + dx / mm) / 2, (1 - dy / mm) / 2
                else:
                    s, t = (1 - dx / mm) / 2, (1 - dy / mm) / 2
                col = np.clip((s * (size - 1)).astype(int), 0, size - 1)
                row = np.clip((t * (size - 1)).astype(int), 0, size - 1)
                out[m] = arrays[key][row, col]

        out = out[::-1]          # VTK 图像行序自下而上，与 6 个面保持一致
        vtk_arr = numpy_to_vtk(out.reshape(-1, 3), deep=1,
                               array_type=vtk.VTK_UNSIGNED_CHAR)
        img = vtk.vtkImageData()
        img.SetDimensions(w, h, 1)
        img.GetPointData().SetScalars(vtk_arr)
        return img

    # ------------------------------------------------------------ 场景搭建
    def _ensure_environment(self):
        if self.env_texture is not None and self.skybox is not None:
            return
        renderer = self.ctx.vtk_renderer
        if renderer is None:
            return

        cube_texture, env_texture, note = self._build_texture()
        self.env_texture = env_texture

        # 1) 无穷远天空盒（cubemap）
        self.skybox = vtk.vtkSkybox()
        self.skybox.SetTexture(cube_texture)
        try:
            self.skybox.SetProjectionToCube()
        except AttributeError:
            pass
        renderer.AddActor(self.skybox)

        # 2) 环境贴图 + 基于图像的光照（PBR 反射）
        try:
            renderer.SetEnvironmentTexture(env_texture)
            renderer.UseImageBasedLightingOn()
            self.ibl_enabled = True
        except AttributeError:
            self.ibl_enabled = False

        # 3) 反射球 / 反射立方体（单位尺寸，靠 actor 缩放适配模型）
        sphere_src = vtk.vtkSphereSource()
        sphere_src.SetRadius(0.5)
        sphere_src.SetThetaResolution(64)
        sphere_src.SetPhiResolution(64)
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere_src.GetOutputPort())
        self.sphere_actor = vtk.vtkActor()
        self.sphere_actor.SetMapper(sphere_mapper)

        cube_src = vtk.vtkCubeSource()
        cube_src.SetXLength(1.0)
        cube_src.SetYLength(1.0)
        cube_src.SetZLength(1.0)
        cube_mapper = vtk.vtkPolyDataMapper()
        cube_mapper.SetInputConnection(cube_src.GetOutputPort())
        self.cube_actor = vtk.vtkActor()
        self.cube_actor.SetMapper(cube_mapper)

        for actor in (self.sphere_actor, self.cube_actor):
            prop = actor.GetProperty()
            prop.SetInterpolationToPBR()
            prop.SetMetallic(1.0)
            prop.SetRoughness(0.03)
            prop.SetBaseIOR(1.5)

        renderer.AddActor(self.sphere_actor)
        renderer.AddActor(self.cube_actor)

        self._reposition()
        self._apply_visibility()
        self._render()
        self._set_status(note + ("，已开启 IBL 环境光照" if self.ibl_enabled else "，当前 VTK 不支持 IBL"))

    def _reposition(self):
        """按当前模型的包围盒把反射球 / 立方体摆到模型前方，大小随模型自适应"""
        pd = self.ctx.current_polydata
        if pd is None or pd.GetNumberOfPoints() == 0 or self.sphere_actor is None:
            return
        b = pd.GetBounds()
        diag = pd.GetLength() or 1.0
        r = max(diag * 0.12, 0.4)
        cx, cy = (b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0
        front = b[5] + r * 1.6          # VTK 默认相机看向 -Z，放在 +Z 一侧即“模型前方”

        self.sphere_actor.SetScale(r)
        self.sphere_actor.SetPosition(cx - r * 1.5, cy, front)

        self.cube_actor.SetScale(r * 1.25)
        self.cube_actor.SetPosition(cx + r * 1.5, cy, front)
        self.cube_actor.SetOrientation(18.0, 32.0, 0.0)   # 转个角度，让每个面反射不同方向

    def rebuild_environment(self):
        """素材换过之后重新载入"""
        renderer = self.ctx.vtk_renderer
        if renderer is None:
            return
        if self.skybox is not None:
            renderer.RemoveActor(self.skybox)
        cube_texture, env_texture, note = self._build_texture()
        self.env_texture = env_texture
        self.skybox = vtk.vtkSkybox()
        self.skybox.SetTexture(cube_texture)
        try:
            self.skybox.SetProjectionToCube()
        except AttributeError:
            pass
        renderer.AddActor(self.skybox)
        if self.ibl_enabled:
            renderer.SetEnvironmentTexture(env_texture)
        self._apply_visibility()
        self._render()
        self._set_status("已重新载入：" + note)

    # --------------------------------------------------------------- 事件
    def on_toggle(self, _state=None):
        self._apply_visibility()
        self._render()

    def _apply_visibility(self):
        if self.skybox is not None:
            self.skybox.SetVisibility(1 if self.chk_skybox.isChecked() else 0)
        if self.sphere_actor is not None:
            self.sphere_actor.SetVisibility(1 if self.chk_sphere.isChecked() else 0)
        if self.cube_actor is not None:
            self.cube_actor.SetVisibility(1 if self.chk_cube.isChecked() else 0)

    def on_material_changed(self, _value=0):
        metal = self.slider_metal.value() / 100.0
        rough = self.slider_rough.value() / 100.0
        for actor in (self.sphere_actor, self.cube_actor):
            if actor is None:
                continue
            prop = actor.GetProperty()
            prop.SetMetallic(metal)
            prop.SetRoughness(rough)
        if self.chk_model is not None and self.chk_model.isChecked():
            self._apply_model_reflect()
        self._render()

    # ------------------------------- 让当前模型也使用 PBR 环境反射
    def on_model_reflect_toggled(self, _state=None):
        self._apply_model_reflect()
        self._render()

    def _apply_model_reflect(self):
        actor = self.ctx.current_actor
        if actor is None:
            return
        prop = actor.GetProperty()
        if self.chk_model.isChecked():
            if self._model_saved is None:
                self._model_saved = (prop.GetInterpolation(),
                                     prop.GetMetallic(), prop.GetRoughness(),
                                     prop.GetColor())
            prop.SetInterpolationToPBR()
            prop.SetMetallic(self.slider_metal.value() / 100.0)
            prop.SetRoughness(max(self.slider_rough.value() / 100.0, 0.05))
        elif self._model_saved is not None:
            interp, metal, rough, color = self._model_saved
            prop.SetInterpolation(interp)
            prop.SetMetallic(metal)
            prop.SetRoughness(rough)
            prop.SetColor(color)
            self._model_saved = None

    # --------------------------------------------------------------- 钩子
    def on_model_loaded(self, obj_path, actor, polydata):
        # 内核已是多模型模式，加载时不会再清空场景，这里只需按当前模型摆一下球/立方体
        self._ensure_environment()
        self._reposition()
        self._apply_visibility()
        if self.chk_model is not None and self.chk_model.isChecked():
            self._model_saved = None
            self._apply_model_reflect()
        self._render()

    def on_window_resize(self):
        pass

    def on_destroy(self):
        renderer = self.ctx.vtk_renderer
        if self.chk_model is not None and self.chk_model.isChecked():
            self.chk_model.setChecked(False)
            self._apply_model_reflect()
        if renderer is not None:
            for a in (self.skybox, self.sphere_actor, self.cube_actor):
                if a is not None:
                    renderer.RemoveActor(a)
            try:
                renderer.UseImageBasedLightingOff()
            except AttributeError:
                pass
        self.env_texture = None

    # --------------------------------------------------------------- 工具
    def _render(self):
        if self.ctx.vtk_widget is not None:
            self.ctx.vtk_widget.GetRenderWindow().Render()

    def _set_status(self, text):
        if self.lbl_status is not None:
            self.lbl_status.setText(text)
        if self.ctx.info_display is not None:
            self.ctx.info_display.append(f"[环境映射] {text}")
