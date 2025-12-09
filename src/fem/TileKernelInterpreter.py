# 储存 tile 的种类 及相应的kernel
# 在程序开始运行时按照tileHandler注册的tile构建type->kernel的字典
# sigma来自json或其他RVE得到的文件
from typing import List,Dict
import jax
import jax.numpy as np
from jax import vmap
import os
import json
from functools import partial


class TileKernelInterpreter:
    def __init__(self, typeList:List,folderPath:str=None,*args,** kwargs) -> None:
        self.typeList = typeList
        self.folderPath = folderPath
        self.kernels:np.ndarray=None # array [[kernel1],[kernel2],[kernel3],...]
        self.p= np.array(kwargs.pop('p',np.array([3.]*len(self.typeList))))
        self._buildKernels()  # 构建包含void的刚度矩阵列表
    

    # @partial(jax.jit, static_argnames=())
    def __call__(self, weights, nx, ny, *args, **kwargs):
        def upsample(arr:np.ndarray, kernel:np.ndarray):
            # 步骤1：扩展原始数组维度 → (Nx, 1, Ny, 1)，为广播做准备
            arr_expanded = arr[:, np.newaxis, :, np.newaxis]  # 插入两个长度为1的轴
            block_array = arr_expanded * kernel[None,:, None, :]  # 广播乘法，结果形状为 (Nx, 3, Ny, 3)
            upsampled = block_array.reshape(nx * 3, ny * 3)
            return upsampled
        batch_upsampled = jax.vmap(upsample, in_axes=(2, 0),out_axes=2)(weights, self.kernels)
        return batch_upsampled

    def __repr__(self) -> str:
        if not self.debug:
            header = f"{'idx':>3}  {'type':>5}"
            bar = "-" * len(header)
            lines = [header, bar]
            for ext_idx, typ in enumerate(self.typeList):
                # 安全地把“外部索引”映射到“内部排序序号”
                lines.append(f"{ext_idx:>3}  {typ:<12}")
            return "\n".join(lines)
        if self.debug:
            return "debug mode"
    
    def _buildKernels(self):
        kernels = []
        for mat_type in self.typeList:
            file_path = os.path.join(self.folderPath, f"{mat_type}.json")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                    kernels.append(data)
                    print(f"Loaded Kernel for * {mat_type} * from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        self.kernels = np.clip(np.array(kernels),1e-12,1)  # 形状：(tileNum+1, 6, 6)（含void）