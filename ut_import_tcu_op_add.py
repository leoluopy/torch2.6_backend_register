import torch
import torch_tcu  # 自定义TCU后端实现

# 重命名并注册设备模块
torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)

# 生成标准张量方法
torch.utils.generate_methods_for_privateuse1_backend()

x1 = torch.ones(4, 4, device='tcu')
x1_cpu = x1.to('cpu')
print(" Go execute To TCU")
x1 = x1_cpu.to('tcu')
y1 = torch.ones(4, 4)
y1 = y1.to('tcu')

add_val = torch.add(x1, y1)

print("   END ")
