import sys
import os
import importlib.util
from PyQt5.QtWidgets import QApplication
import yaml

# 项目根目录加入搜索路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from core.viewer_3d import VTKViewerCore
from plugins.base_plugin import Base3DPlugin

def auto_scan_plugins(plugin_dir: str = "plugins") -> dict:
    """自动扫描插件目录，返回 {plugin_cls_name: 插件类}"""
    plugin_map = {}
    plugins_path = os.path.join(BASE_DIR, plugin_dir)

    for file in os.listdir(plugins_path):
        if not file.endswith(".py") or file.startswith("base_"):
            continue
        module_name = file[:-3]
        file_path = os.path.join(plugins_path, file)

        # 动态导入插件模块
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # 筛选合法插件子类
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Base3DPlugin)
                and attr is not Base3DPlugin
                and getattr(attr, "plugin_cls_name", "")
            ):
                key = attr.plugin_cls_name
                plugin_map[key] = attr
    return plugin_map

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = VTKViewerCore()

    # 自动扫描全部插件
    plugin_mapping = auto_scan_plugins()

    # 读取YAML配置文件
    config_path = os.path.join(BASE_DIR, "plugin_config.yaml")
    active_list = []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        active_list = cfg.get("active_plugins", [])
        # 如果获取到的是None，转换为空列表
        if active_list is None:
            active_list = []
    except FileNotFoundError:
        print("提示：未找到 plugin_config.yaml，自动加载所有可用插件")
        active_list = list(plugin_mapping.keys())
    except Exception as e:
        print(f"YAML配置读取异常: {e}，自动加载所有插件")
        active_list = list(plugin_mapping.keys())

    # 按配置注册插件
    for plug_key in active_list:
        if plug_key in plugin_mapping:
            viewer.register_plugin(plugin_mapping[plug_key])
            print(f"已加载插件: {plug_key}")
        else:
            print(f"警告：不存在标识为 [{plug_key}] 的插件，跳过")

    viewer.show()
    sys.exit(app.exec_())