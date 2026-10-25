import os
from PyQt5.QtWidgets import (QWidget, QToolBar, QAction, QActionGroup, QSizePolicy)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
import vtk
from plugins.base_plugin import Base3DPlugin

class ToolbarViewPlugin(Base3DPlugin):
    # YAML配置唯一标识
    plugin_cls_name = "view_toolbar"
    plugin_name = "view_toolbar"
    plugin_desc = "顶部工具栏插件：线框/面片切换、退出按钮"

    def __init__(self, viewer_ctx):
        super().__init__(viewer_ctx)
        self.toolbar = None
        # 显示模式标记：0=面片，1=线框
        self.display_mode = 0
        # 动作组，实现按钮互斥
        self.mode_action_group = None
        self.act_wire = None
        self.act_surface = None
        self.act_exit = None

    def create_ui(self):
        # 1. 创建工具栏，挂载到主窗口顶部菜单栏下方
        self.toolbar = QToolBar("Model View Toolbar", self.ctx.main_window)
        self.toolbar.setIconSize(QSize(32, 32))
        self.toolbar.setMovable(False)
        self.toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # 图标目录
        icon_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/icon")

        # ========== 第一组：线框/面片互斥按钮 ==========
        self.mode_action_group = QActionGroup(self.toolbar)
        self.mode_action_group.setExclusive(True)

        # 面片按钮（默认选中）
        surf_icon_path = os.path.join(icon_root, "surface.png")
        self.act_surface = QAction(QIcon(surf_icon_path), "面片显示", self.toolbar)
        self.act_surface.setCheckable(True)
        self.act_surface.setChecked(True)
        self.act_surface.triggered.connect(lambda: self.set_display_mode(0))
        self.mode_action_group.addAction(self.act_surface)
        self.toolbar.addAction(self.act_surface)

        # 线框按钮
        wire_icon_path = os.path.join(icon_root, "wire.png")
        self.act_wire = QAction(QIcon(wire_icon_path), "线框显示", self.toolbar)
        self.act_wire.setCheckable(True)
        self.act_wire.triggered.connect(lambda: self.set_display_mode(1))
        self.mode_action_group.addAction(self.act_wire)
        self.toolbar.addAction(self.act_wire)

        # 分组分隔符
        self.toolbar.addSeparator()

        # ========== 第二组：退出按钮 ==========
        exit_icon_path = os.path.join(icon_root, "exit.png")
        self.act_exit = QAction(QIcon(exit_icon_path), "退出程序", self.toolbar)
        self.act_exit.triggered.connect(self.ctx.main_window.close)
        self.toolbar.addAction(self.act_exit)

        # 将工具栏添加到主窗口顶部
        self.ctx.main_window.addToolBar(Qt.TopToolBarArea, self.toolbar)

        # 工具栏无右侧面板，返回空Widget不占用右侧区域
        dummy_panel = QWidget()
        dummy_panel.setFixedSize(0,0)
        return dummy_panel

    def set_display_mode(self, mode):
        """切换渲染模式：0面片，1线框"""
        self.display_mode = mode
        actor = self.ctx.current_actor
        if not actor:
            self.ctx.info_display.append("提示：未加载任何模型，无法切换显示模式")
            return
        prop = actor.GetProperty()
        if mode == 0:
            # 面片模式：填充实体，无线框
            prop.SetRepresentationToSurface()
        else:
            # 线框模式：仅网格线
            prop.SetRepresentationToWireframe()
        self.ctx.vtk_widget.GetRenderWindow().Render()
        mode_text = "面片模式" if mode == 0 else "线框模式"
        self.ctx.info_display.append(f"切换模型显示：{mode_text}")

    def on_model_loaded(self, obj_path, actor, polydata):
        """加载新模型，同步当前显示模式"""
        self.set_display_mode(self.display_mode)

    def on_window_resize(self):
        """工具栏无需缩放逻辑，空实现"""
        pass

    def on_destroy(self):
        """销毁工具栏"""
        if self.toolbar:
            self.toolbar.deleteLater()