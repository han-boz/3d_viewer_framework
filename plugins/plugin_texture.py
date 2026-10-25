import os
from PyQt5.QtWidgets import QWidget, QScrollArea, QGridLayout, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPixmap, QIcon
import vtk
from plugins.base_plugin import Base3DPlugin

class TexturePlugin(Base3DPlugin):
    # yaml配置里写这个字符串
    plugin_cls_name = "texture_mapping"
    plugin_name = "texture_mapping"
    plugin_desc = "纹理贴图功能插件"

    def __init__(self, viewer_ctx):
        super().__init__(viewer_ctx)
        self.selected_texture_path = None
        self.selected_texture_btn = None
        self.texture_grid = None
        self.texture_scroll = None
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.load_texture_images)
        # 使用项目BASE_DIR构建绝对路径，不要用../
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.texture_dir = os.path.normpath(os.path.join(self.BASE_DIR, "../data/texture"))

    def create_ui(self):
        """构建纹理缩略图面板，挂载到右侧扩展区"""
        self.plugin_panel = QWidget()
        layout = QVBoxLayout(self.plugin_panel)
        layout.setContentsMargins(0,0,0,0)

        self.texture_scroll = QScrollArea()
        self.texture_scroll.setWidgetResizable(True)
        self.texture_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.texture_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        self.texture_grid = QGridLayout(container)
        self.texture_grid.setSpacing(6)
        self.texture_grid.setContentsMargins(15,15,15,15)
        for c in range(4):
            self.texture_grid.setColumnStretch(c, 1)
        self.texture_scroll.setWidget(container)
        layout.addWidget(self.texture_scroll)
        # 等待窗口布局完成再加载图片，延迟200ms
        QTimer.singleShot(200, self.load_texture_images)
        return self.plugin_panel

    def on_window_resize(self):
        # 窗口缩放延迟刷新纹理缩略图
        self.timer.start(150)

    def on_model_loaded(self, obj_path, actor, polydata):
        # 加载模型后自动应用已选纹理
        if self.selected_texture_path:
            self.apply_texture(self.selected_texture_path)

    def on_texture_btn_click(self, btn, tex_path):
        if self.selected_texture_btn:
            self.selected_texture_btn.setStyleSheet("")
        btn.setStyleSheet("border:3px solid #ff4400;")
        self.selected_texture_btn = btn
        self.selected_texture_path = tex_path
        self.ctx.info_display.append(f"\n选中纹理：{tex_path}")
        if self.ctx.current_actor:
            self.apply_texture(tex_path)

    def apply_texture(self, tex_path):
        # 纹理映射逻辑，完全隔离在插件
        if tex_path.lower().endswith((".jpg", ".jpeg")):
            reader = vtk.vtkJPEGReader()
        else:
            reader = vtk.vtkPNGReader()
        reader.SetFileName(tex_path)
        reader.Update()
        texture = vtk.vtkTexture()
        texture.SetInputData(reader.GetOutput())
        texture.RepeatOn()
        texture.InterpolateOn()

        tex_gen = vtk.vtkTextureMapToCylinder()
        tex_gen.SetInputData(self.ctx.current_actor.GetMapper().GetInput())
        tex_gen.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(tex_gen.GetOutput())
        self.ctx.current_actor.SetMapper(mapper)
        self.ctx.current_actor.SetTexture(texture)
        self.ctx.vtk_widget.GetRenderWindow().Render()
        self.ctx.info_display.append(f"纹理已绑定模型：{os.path.basename(self.ctx.current_obj_path)}")

    def load_texture_images(self):
        max_c = 4
        # 清空旧控件
        while self.texture_grid.count():
            item = self.texture_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        view_w = self.texture_scroll.viewport().width()
        if view_w <= 20:
            view_w = 300
        spacing = self.texture_grid.spacing()
        mg = self.texture_grid.contentsMargins()
        avail = view_w - mg.left() - mg.right() - (max_c-1)*spacing
        thumb_size = max(64, int(avail / max_c))

        img_list = []
        if os.path.exists(self.texture_dir):
            for f in sorted(os.listdir(self.texture_dir)):
                if f.lower().endswith((".png",".jpg",".jpeg")):
                    img_list.append(os.path.join(self.texture_dir, f))
        else:
            print(f"[TexturePlugin] 纹理目录不存在: {self.texture_dir}")
            return

        # 只渲染真实图片，不再填充空占位按钮
        for slot, path in enumerate(img_list):
            r = slot // max_c
            c = slot % max_c
            pix = QPixmap(path).scaled(thumb_size, thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            btn = QPushButton()
            btn.setFixedSize(thumb_size, thumb_size)
            btn.setIcon(QIcon(pix))
            btn.setIconSize(QSize(thumb_size-8, thumb_size-8))
            btn.clicked.connect(lambda ch, b=btn, p=path: self.on_texture_btn_click(b,p))
            self.texture_grid.addWidget(btn, r, c)
