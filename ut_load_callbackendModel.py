import torch
import torch_tcu  # 自定义TCU后端实现
import torchvision.models

# 重命名并注册设备模块
torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)

# 生成标准张量方法
torch.utils.generate_methods_for_privateuse1_backend()

res18 = torchvision.models.resnet18()
res18.eval()
res18 = res18.to('tcu')

x = torch.ones([1, 3, 224, 224]).to('tcu')
out = res18(x)
print(out.shape)

print("   END ")
