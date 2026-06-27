from ultralytics.utils.torch_utils import select_device
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

try:
    dev_auto = select_device("auto")
    print("select_device('auto') success:", dev_auto)
except Exception as e:
    print("select_device('auto') failed:", str(e))

try:
    dev_0 = select_device("0")
    print("select_device('0') success:", dev_0)
except Exception as e:
    print("select_device('0') failed:", str(e))
