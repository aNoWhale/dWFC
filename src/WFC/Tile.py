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
        """
        pass

    @property
    @abstractmethod
    def property(self):
        """
        获取tile的属性。
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

