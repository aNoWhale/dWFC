from abc import ABC, abstractmethod



class Tile(ABC):
    """
    所有tile的基类，所有tile都必须继承此类并实现此类的功能
    """
    @abstractmethod
    def build(self,points,*args,**kwargs):
        """
        call this method to build the tile structure to stp.
        :param points:
        :param args:
        :param kwargs:
        :return:
        input:
                points: 立方体8个角的坐标点列表，格式为[[x1,y1,z1], [x2,y2,z2], ...]
                   按照以下顺序排列：
                   [0]: 底面左前角 (x_min, y_min, z_min)
                   [1]: 底面右前角 (x_max, y_min, z_min)
                   [2]: 顶面右前角 (x_max, y_min, z_max)
                   [3]: 顶面左前角 (x_min, y_min, z_max)
                   [4]: 底面左后角 (x_min, y_max, z_min)
                   [5]: 底面右后角 (x_max, y_max, z_min)
                   [6]: 顶面右后角 (x_max, y_max, z_max)
                   [7]: 顶面左后角 (x_min, y_max, z_max)
        return:
             result_shape
        """

        pass

    @property
    @abstractmethod
    def properties(self):
        """
        获取tile的属性。已弃用
        :return: tile属性的dict
        """
        pass



if __name__ == '__main__':
    """一个Tile使用方法的示例"""
    class TestTile(Tile):
        @property
        def property(self):
            return {"strength": 100,
                    "youngs":19899}

        def build(self, points, *args, **kwargs):
            #save as stp
            pass

