import torch
import torch_tcu  # 自定义TCU后端实现

# 重命名并注册设备模块
torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)

# 生成标准张量方法
torch.utils.generate_methods_for_privateuse1_backend()

def test(x, y):
    print()
    print("Test START")
    # Check that our device is correct.
    print(f'x.device={x.device}, x.is_cpu={x.is_cpu}')
    print(f'y.device={y.device}, y.is_cpu={y.is_cpu}')

    # calls out custom add kernel, registered to the dispatcher
    print('Calling z = x + y')
    z = x + y
    print(f'z.device={z.device}, z.is_cpu={z.is_cpu}')

    print('Calling z = z.to(device="cpu")')
    z_cpu = z.to(device='cpu')

    # Check that our cross-device copy correctly copied the data to cpu
    print(f'z_cpu.device={z_cpu.device}, z_cpu.is_cpu={z_cpu.is_cpu}')

    # Confirm that calling the add kernel no longer invokes our custom kernel,
    # since we're using CPU t4ensors.
    print('Calling z2 = z_cpu + z_cpu')
    z2 = z_cpu + z_cpu
    print("Test END")
    print()

# Option 1: Use torch.register_privateuse1_backend("tcu"), which will allow
# "tcu" as a device string to work seamlessly with pytorch's API's.
# You may need a more recent nightly of PyTorch for this.
torch.utils.rename_privateuse1_backend("tcu")

# Show that in general, passing in a custom device string will fail.
try:
    x = torch.ones(4, 4, device='bar')
    exit("Error: you should not be able to make a tensor on an arbitrary 'bar' device.")
except RuntimeError as e:
    print("(Correctly) unable to create tensor on device='bar'")

# Show that in general, passing in a custom device string will fail.
try:
    x = torch.ones(4, 4, device='tcu:2')
    exit("Error: the tcu device only has two valid indices: tcu:0 and tcu:1")
except RuntimeError as e:
    print("(Correctly) unable to create tensor on device='tcu:2'")

print("Creating x on device 'tcu:0'")
x1 = torch.ones(4, 4, device='tcu:0')
print("Creating y on device 'tcu:0'")
y1 = torch.ones(4, 4, device='tcu:0')

test(x1, y1)


# Option 2: Directly expose a custom device object
# You can pass an optional index arg, specifying which device index to use.
tcu_device1 = torch_tcu.tcu(1)

print("Creating x on device 'tcu:1'")
x2 = torch.ones(4, 4, device=tcu_device1)
print("Creating y on device 'tcu:1'")
y2 = torch.ones(4, 4, device=tcu_device1)

# Option 3: Enable a TorchFunctionMode object in user land,
# that will convert `device="tcu"` calls into our custom device objects automatically.
# Option 1 is strictly better here (in particular, printing a.device() will still
# print "privateuseone" instead of your custom device name). Mostly showing this option because:
# (a) Torch Function Modes have been around for longer, and the API in Option 1
#     is only available on a more recent nightly.
# (b) This is a cool example of how powerful torch_function and torch_dispatch modes can be!
# holder = enable_tcu_device()
# del _holder

test(x2, y2)
