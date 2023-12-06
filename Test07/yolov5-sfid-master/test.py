import torch
from models.common import yolov5s  # 导入你的模型
model = yolov5s()  # 创建模型实例
model.load_state_dict(torch.load("pretrained/yolov5s.pt"))  # 加载权重
traced_model = torch.jit.trace(model.eval(), torch.rand(1, 3, 416, 416))  # 转换为 TorchScript
traced_model.save("yolov5s.torchscript.pt")  # 保存 TorchScript 模型
