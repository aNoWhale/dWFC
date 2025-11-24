from src.WFC.Tile import Tile
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopLoc import TopLoc_Location 
import numpy as np

class Voidtile(Tile):
    """实例化以后调用实例方法即可，减少内存开支

    """
    def __init__(self, ):
        pass
    def build(self,points,*args,**kwargs):
        pass