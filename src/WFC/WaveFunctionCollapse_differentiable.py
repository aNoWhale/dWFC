# import torch
# import torch.nn.functional as F
#
#
# class DifferentiableWFC(torch.nn.Module):  # 继承Module以简化梯度管理
#     def __init__(self, grid_size, num_tiles, compatibility_rules,
#                  entropy_coef=1.0, constraint_coef=0.5, temp_start=1.0, temp_decay=0.95):
#         super().__init__()
#         self.H, self.W = grid_size
#         self.num_tiles = num_tiles
#         self.register_buffer('compat_rules', compatibility_rules)  # 固定约束规则
#         self.entropy_coef = entropy_coef
#         self.constraint_coef = constraint_coef
#         self.temp = temp_start
#         self.temp_decay = temp_decay
#
#         # 关键修复：添加梯度追踪
#         self.grid_logits = torch.nn.Parameter(  # 使用可学习的logits张量
#             torch.zeros(self.H, self.W, num_tiles), requires_grad=True
#         )
#
#         # 正确初始化
#         self._initialize_probs()
#
#     def _initialize_probs(self):
#         """初始化等概率分布"""
#         with torch.no_grad():
#             self.grid_logits.data = torch.ones_like(self.grid_logits) / self.num_tiles
#             self.grid_logits.data = torch.log(self.grid_logits.data)  # 转换为logits
#
#     @property
#     def grid_probs(self):
#         """获取概率分布"""
#         return F.softmax(self.grid_logits, dim=-1)
#
#     def shannon_entropy(self, probs):
#         """计算香农熵的可微近似"""
#         safe_probs = torch.clamp(probs, min=1e-8, max=1 - 1e-8)
#         entropy = -torch.sum(safe_probs * torch.log(safe_probs), dim=-1)
#         return entropy
#
#     def neighborhood_loss(self):
#         """计算邻域约束损失 (可微版本)"""
#         # 优化：使用卷积风格的邻域检查
#         total_loss = 0.0
#         probs = self.grid_probs
#
#         # 定义四个方向 (上, 右, 下, 左)
#         dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
#
#         # 填充边界 (使用零填充)
#         padded_probs = F.pad(probs, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
#
#         for i, (dh, dw) in enumerate(dirs):
#             # 提取偏移邻域概率
#             shifted_probs = padded_probs[1 + dh: 1 + self.H + dh, 1 + dw: 1 + self.W + dw]
#
#             # 获取当前方向的约束规则
#             compat_rule = self.compat_rules[i]  # [num_tiles, num_tiles]
#
#             # 高效计算兼容性损失
#             # 计算: 邻居状态s2在位置(x,y)出现的概率 * 约束值(s1,s2) * 当前位置状态s1出现的概率
#             # 维度: [H, W, num_tiles]
#             compat_loss = torch.einsum('hwn,nm,hwm->hw', shifted_probs, compat_rule, probs)
#
#             # 添加负对数似然损失 (鼓励高兼容性)
#             total_loss += -torch.mean(torch.log(compat_loss + 1e-8))
#
#         return total_loss
#
#     def sharpen(self, temperature):
#         """锐化概率分布"""
#         return self.grid_logits / temperature
#
#     def forward(self):
#         """组合损失计算，用于反向传播"""
#         # 计算熵损失 (鼓励低熵/确定性)
#         entropy_loss = torch.mean(self.shannon_entropy(self.grid_probs)) * self.entropy_coef
#
#         # 计算约束损失
#         constraint_loss = self.neighborhood_loss() * self.constraint_coef
#
#         # 组合总损失
#         return entropy_loss + constraint_loss
#
#     def step(self, optimizer):
#         """执行单步优化并更新温度"""
#         optimizer.zero_grad()
#         total_loss = self.forward()
#         total_loss.backward()
#         optimizer.step()
#
#         # 应用锐化并更新温度
#         with torch.no_grad():
#             self.grid_logits.data = self.sharpen(self.temp)
#             self.temp *= self.temp_decay
#
#         return total_loss.item()
#
#     def collapse(self):
#         """执行最终坍缩 (仅在训练后使用)"""
#         return torch.argmax(self.grid_logits, dim=-1)
#
#
# # 示例用法
# if __name__ == "__main__":
#     # 1. 参数配置
#     grid_size = (10, 10)
#     num_tiles = 4
#
#     # 2. 创建可微约束规则
#     compatibility_rules = torch.zeros(4, num_tiles, num_tiles) #（direction,N_tiletype,N_tiletype）
#
#     # 模块0可以与所有模块相邻
#     compatibility_rules[:, 0, :] = 1.0
#
#     # 模块1只能与自身相邻
#     compatibility_rules[0, 1, 1] = 1.0  # 上
#     compatibility_rules[1, 1, 1] = 1.0  # 右
#     compatibility_rules[2, 1, 1] = 1.0  # 下
#     compatibility_rules[3, 1, 1] = 1.0  # 左
#
#     # 模块2和3可以互相相邻
#     compatibility_rules[0, 2, 3] = 1.0  # 上: 模块2上方是模块3
#     compatibility_rules[0, 3, 2] = 1.0  # 上: 模块3上方是模块2
#     compatibility_rules[1, 2, 3] = 1.0  # 右: 模块2右侧是模块3
#     compatibility_rules[1, 3, 2] = 1.0  # 右: 模块3右侧是模块2
#     compatibility_rules[2, 2, 3] = 1.0  # 下: 模块2下方是模块3
#     compatibility_rules[2, 3, 2] = 1.0  # 下: 模块3下方是模块2
#     compatibility_rules[3, 2, 3] = 1.0  # 左: 模块2左侧是模块3
#     compatibility_rules[3, 3, 2] = 1.0  # 左: 模块3左侧是模块2
#
#     # 3. 初始化可微WFC
#     wfc = DifferentiableWFC(
#         grid_size=grid_size,
#         num_tiles=num_tiles,
#         compatibility_rules=compatibility_rules,
#         entropy_coef=1.0,
#         constraint_coef=0.5,
#         temp_start=1.0,
#         temp_decay=0.97
#     )
#
#     # 4. 优化器设置 - 现在可以正确跟踪梯度
#     optimizer = torch.optim.Adam(wfc.parameters(), lr=0.01)  # 注意使用wfc.parameters()
#
#     # 5. 训练循环
#     for step in range(100):
#         total_loss = wfc.step(optimizer)
#
#         if step % 10 == 0:
#             # 计算平均熵(用于监控)
#             avg_entropy = torch.mean(wfc.shannon_entropy(wfc.grid_probs)).item()
#             print(f"Step {step}: loss={total_loss:.4f}, avg_entropy={avg_entropy:.4f}, temp={wfc.temp:.3f}")
#
#     # 6. 获取最终坍缩结果
#     collapsed_grid = wfc.collapse()
#     print("Collapsed grid:")
#     print(collapsed_grid.cpu().numpy())