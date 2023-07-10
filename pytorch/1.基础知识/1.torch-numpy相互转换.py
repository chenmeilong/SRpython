import  torch
import  numpy as np


# 创建一个 numpy ndarray
numpy_tensor = np.random.randn(1, 2)

# 我们可以使用下面两种方式将numpy的ndarray转换到tensor上
pytorch_tensor1 = torch.Tensor(numpy_tensor)
pytorch_tensor2 = torch.from_numpy(numpy_tensor)

# 使用以上两种方法进行转换的时候，会直接将 NumPy ndarray 的数据类型转换为对应的 PyTorch Tensor 数据类型

# 同时我们也可以使用下面的方法将 pytorch tensor 转换为 numpy ndarray

# 如果 pytorch tensor 在 cpu 上
numpy_array = pytorch_tensor1.numpy()
# 如果 pytorch tensor 在 gpu 上
numpy_array = pytorch_tensor1.cpu().numpy() # GPU 上的 Tensor 不能直接转换为 NumPy ndarray，需要使用`.cpu()`先将 GPU 上的 Tensor 转到 CPU 上

print ("numpy",numpy_tensor)
print ("torch",pytorch_tensor1)
print ("tensor2arry",numpy_array)

