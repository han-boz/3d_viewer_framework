from PyQt5.QtWidgets import QWidget

class Base3DPlugin:
    """3D查看器插件基类，所有功能插件实现该接口"""
    # 约定：子类必须重写该值，作为json配置里的唯一标识
    plugin_cls_name = ""
    plugin_name = "base_plugin"
    plugin_desc = "基础插件模板"

    def __init__(self, viewer_ctx):
        # viewer_ctx：框架上下文，包含主窗口、renderer、vtk_widget、info_display等全局对象
        self.ctx = viewer_ctx
        self.plugin_panel = None  # 插件专属UI面板，会挂载到右侧扩展区

    def create_ui(self):
        """【钩子】创建插件专属UI面板，返回QWidget"""
        raise NotImplementedError("插件必须实现create_ui")

    def on_model_loaded(self, obj_path, actor, polydata):
        """【钩子】模型加载完成后触发，传入模型路径、actor、多边形数据"""
        pass

    def on_render_update(self):
        """【钩子】渲染刷新时触发"""
        pass

    def on_window_resize(self):
        """【钩子】窗口缩放触发（纹理缩略图重绘放这里）"""
        pass

    def on_destroy(self):
        """【钩子】程序关闭释放资源"""
        pass