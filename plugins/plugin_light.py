from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QFrame
from PyQt5.QtCore import Qt
import vtk
from plugins.base_plugin import Base3DPlugin

class LightPlugin(Base3DPlugin):
    plugin_cls_name = "light_control"
    plugin_name = "light_control"
    plugin_desc = "光照材质调节插件"
    
    def __init__(self, viewer_ctx):
        super().__init__(viewer_ctx)
        self.light_global_enable = True
        self.light_ambient = True
        self.light_diffuse = True
        self.light_specular = True
        self.scene_light = None
        self.chk_global = None
        self.chk_ambient = None
        self.chk_diffuse = None
        self.chk_specular = None

    def create_ui(self):
        self.plugin_panel = QFrame()
        self.plugin_panel.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self.plugin_panel)
        layout.setSpacing(6)
        layout.setContentsMargins(10,10,10,10)

        # 全局光照开关
        self.chk_global = QCheckBox("启用光照")
        self.chk_global.setChecked(self.light_global_enable)
        self.chk_global.stateChanged.connect(self.on_global_toggle)
        layout.addWidget(self.chk_global)

        h_layout = QHBoxLayout()
        self.chk_ambient = QCheckBox("环境光")
        self.chk_ambient.setChecked(self.light_ambient)
        self.chk_ambient.stateChanged.connect(self.update_material)

        self.chk_diffuse = QCheckBox("漫反射")
        self.chk_diffuse.setChecked(self.light_diffuse)
        self.chk_diffuse.stateChanged.connect(self.update_material)

        self.chk_specular = QCheckBox("镜面高光")
        self.chk_specular.setChecked(self.light_specular)
        self.chk_specular.stateChanged.connect(self.update_material)

        h_layout.addWidget(self.chk_ambient)
        h_layout.addWidget(self.chk_diffuse)
        h_layout.addWidget(self.chk_specular)
        layout.addLayout(h_layout)

        # 创建场景光源
        self.create_light()
        self.set_sub_enable(True)
        return self.plugin_panel

    def create_light(self):
        self.scene_light = vtk.vtkLight()
        self.scene_light.SetPosition(-100,100,120)
        self.scene_light.SetFocalPoint(0,0,0)
        self.scene_light.SetIntensity(0.8)
        self.scene_light.SetColor(1,1,1)
        self.scene_light.SetLightTypeToSceneLight()
        self.ctx.vtk_renderer.AddLight(self.scene_light)

    def set_sub_enable(self, enable):
        self.chk_ambient.setEnabled(enable)
        self.chk_diffuse.setEnabled(enable)
        self.chk_specular.setEnabled(enable)

    def on_global_toggle(self, state):
        self.light_global_enable = (state == Qt.Checked)
        self.set_sub_enable(self.light_global_enable)
        self.scene_light.SetSwitch(self.light_global_enable)
        self.update_material()
        self.ctx.vtk_widget.GetRenderWindow().Render()

    def update_material(self):
        if not self.ctx.current_actor:
            return
        mat = self.ctx.current_actor.GetProperty()
        if not self.light_global_enable:
            mat.SetAmbient(0.0)
            mat.SetDiffuse(0.0)
            mat.SetSpecular(0.0)
        else:
            mat.SetAmbient(0.2 if self.chk_ambient.isChecked() else 0.0)
            mat.SetDiffuse(0.8 if self.chk_diffuse.isChecked() else 0.0)
            mat.SetSpecular(0.4 if self.chk_specular.isChecked() else 0.0)
        self.ctx.vtk_widget.GetRenderWindow().Render()

    def on_model_loaded(self, obj_path, actor, polydata):
        # 新模型加载自动同步光照参数
        self.update_material()