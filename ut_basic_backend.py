import torch
import torch_tcu  # 自定义TCU后端实现

# 重命名并注册设备模块
torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)

# 生成标准张量方法
torch.utils.generate_methods_for_privateuse1_backend()

# 使用TCU设备创建张量
x = torch.ones(2, 2, device='tcu')
print(x.device)  # 应该输出 tcu:0