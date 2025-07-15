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