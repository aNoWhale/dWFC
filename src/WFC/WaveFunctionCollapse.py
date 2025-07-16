import numpy as np
import random
from typing import List, Dict, Tuple, Set, Optional


class WaveFunctionCollapse:
    def __init__(self, grid_size: int, tile_types: List[str], adjacency_rules: Dict[str, Set[str]]):
        """
        初始化波函数坍缩算法

        参数:
        grid_size: 网格大小
        tile_types: 可用的图块类型列表
        adjacency_rules: 图块相邻规则，格式为 {图块类型: 可以相邻的图块类型集合}
        """
        self.grid_size = grid_size
        self.tile_types = tile_types
        self.adjacency_rules = adjacency_rules
        self.grid = self._initialize_grid()
        self.entropy_grid = self._initialize_entropy_grid()

    def _initialize_grid(self) -> List[List[Set[str]]]:
        """初始化网格，每个单元格包含所有可能的图块类型"""
        return [[set(self.tile_types) for _ in range(self.grid_size)] for _ in range(self.grid_size)]

    def _initialize_entropy_grid(self) -> List[List[int]]:
        """初始化熵网格，每个单元格的熵值为可能的图块类型数量"""
        return [[len(self.tile_types) for _ in range(self.grid_size)] for _ in range(self.grid_size)]

    def _get_lowest_entropy_cell(self) -> Optional[Tuple[int, int]]:
        """获取熵值最低且未坍缩的单元格坐标"""
        min_entropy = float('inf')
        candidates = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                entropy = self.entropy_grid[i][j]
                # 跳过已坍缩的单元格（熵值为1）
                if entropy == 1:
                    continue

                if entropy < min_entropy:
                    min_entropy = entropy
                    candidates = [(i, j)]
                elif entropy == min_entropy:
                    candidates.append((i, j))

        # 如果没有候选者，说明所有单元格都已坍缩
        if not candidates:
            return None

        # 随机选择一个最低熵的单元格
        return random.choice(candidates)

    def _collapse_cell(self, i: int, j: int) -> None:
        """坍缩指定单元格为一个随机选择的可能状态"""
        possible_states = list(self.grid[i][j])
        chosen_state = random.choice(possible_states)
        self.grid[i][j] = {chosen_state}
        self.entropy_grid[i][j] = 1

    def _propagate(self, i: int, j: int) -> None:
        """传播坍缩，更新相邻单元格的可能状态"""
        stack = [(i, j)]

        while stack:
            x, y = stack.pop()
            current_possibilities = self.grid[x][y]

            # 检查四个方向的邻居
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy

                # 检查邻居是否在网格内
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbor_possibilities = self.grid[nx][ny]
                    new_possibilities = set()

                    # 应用相邻规则
                    for tile in neighbor_possibilities:
                        for current_tile in current_possibilities:
                            # 检查当前图块是否可以与邻居图块相邻
                            if tile in self.adjacency_rules.get(current_tile, set()):
                                new_possibilities.add(tile)
                                break

                    # 如果邻居的可能状态发生了变化
                    if new_possibilities != neighbor_possibilities:
                        self.grid[nx][ny] = new_possibilities
                        self.entropy_grid[nx][ny] = len(new_possibilities)
                        stack.append((nx, ny))

    def run(self) -> List[List[str]]:
        """运行波函数坍缩算法直到所有单元格都被坍缩"""
        while True:
            # 获取最低熵的单元格
            cell = self._get_lowest_entropy_cell()

            # 如果没有找到可坍缩的单元格，说明算法完成
            if cell is None:
                break

            i, j = cell

            # 坍缩单元格
            self._collapse_cell(i, j)

            # 传播坍缩
            self._propagate(i, j)

        # 将结果转换为字符串网格
        result = [[next(iter(cell)) for cell in row] for row in self.grid]
        return result


# 使用示例
if __name__ == "__main__":
    # 定义图块类型
    tile_types = ["🌿", "🌊", "🏠", "🌲"]

    # 定义相邻规则
    adjacency_rules = {
        "🌿": {"🌿", "🌊", "🏠", "🌲"},
        "🌊": {"🌊", "🌿"},
        "🏠": {"🏠", "🌿"},
        "🌲": {"🌲", "🌿"}
    }

    # 创建并运行算法
    wfc = WaveFunctionCollapse(grid_size=8, tile_types=tile_types, adjacency_rules=adjacency_rules)
    result = wfc.run()

    # 打印结果
    for row in result:
        print(' '.join(row))