# torch2.6_backend_register
torch2.6_backend_register

# reference
https://pytorch.org/tutorials/advanced/privateuseone.html
https://pytorch.org/tutorials/advanced/extend_dispatcher.html
https://pytorch.org/tutorials/advanced/torch_script_custom_ops
https://pytorch.org/tutorials/advanced/dispatcher


# HOWTO
use docker to build 
https://www.codewithgpu.com/i/vllm-project/vllm/vllm0.8.x_torch2.6_ai_chip_backend

```
# use command to remove package 
pip uninstall torch_tcu
rm -rf /usr/local/lib/python3.12/dist-packages/torch_tcu*

python setup.py install

python ut_basic_backend.py
python ut_custom_rms_norm_tcu.py  
python ut_import_tcu_op_add.py  
python ut_load_call_backendop.py  
python ut_load_callbackendModel.py

```

for all steps you can use
```
sh run_build_and_test.sh
```

