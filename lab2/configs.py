import os

# 模型保存路径
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'models')
# 数据存放路径
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
# 特征图保存路路径
IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'images')


def generate_save_dir(model_name: str, show_path: bool = True):
    """模型保存路径

    Args:
        model_name (str): 模型名称.
        show_path (bool, optional): 是否打印路径名. 默认为 True.

    Returns:
        str: 模型保存的路径.
    """
    model_path = os.path.join(SAVE_DIR, model_name)
    if show_path:
        print("models will be saved at: ", model_path)
    return model_path


cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(cifar10_classes)
