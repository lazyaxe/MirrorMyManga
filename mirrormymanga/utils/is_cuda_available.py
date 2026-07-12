import paddle

def is_cuda_available():
    device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    return device