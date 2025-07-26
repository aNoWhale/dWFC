from src.dynamicGenerator.TileImplement.CubeTile import CubeTile



class CF(CubeTile):
    """
    Cylindrical frame(圆柱框架)

    """

    @classmethod
    def build(cls, points, *args, **kwargs):
        """生成一个立方体框架结构并保存为STP格式

        Args:
            points: 立方体8个角的坐标点列表，格式为[[x1,y1,z1], [x2,y2,z2], ...]
                   按照以下顺序排列：
                   [0]: 底面左前角 (x_min, y_min, z_min)
                   [1]: 底面右前角 (x_max, y_min, z_min)
                   [2]: 底面右后角 (x_max, y_max, z_min)
                   [3]: 底面左后角 (x_min, y_max, z_min)
                   [4]: 顶面左前角 (x_min, y_min, z_max)
                   [5]: 顶面右前角 (x_max, y_min, z_max)
                   [6]: 顶面右后角 (x_max, y_max, z_max)
                   [7]: 顶面左后角 (x_min, y_max, z_max)

        Returns:
            str: 生成的STP文件路径
        """
        filename = "cylindrical_frame.stp"
        # 12 条边（顶点索引）
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)  # 立棱
        ]
        # 3. 统一圆柱半径
        result_shape = CubeTile.build_cylinder(points, edges, CubeTile.RADIUS)
        CubeTile.write_stp(filename, result_shape)
        return result_shape





if __name__ == "__main__":
    #cube=Cube()
    cube_points = [
        [0, 0, 0],  # 底面左前角
        [1, 0, 0],  # 底面右前角
        [1, 1, 0],  # 底面右后角
        [0, 1, 0],  # 底面左后角
        [0, 0, 1],  # 顶面左前角
        [1, 0, 1],  # 顶面右前角
        [1, 1, 1],  # 顶面右后角
        [0, 1, 1]  # 顶面左后角
    ]
    # 生成立方体框架结构并保存为STP文件
    result = CF.build(cube_points)
    if result:
        print(f"立方体框架结构已成功生成并保存到: {result}")
    else:
        print("生成立方体框架结构失败")