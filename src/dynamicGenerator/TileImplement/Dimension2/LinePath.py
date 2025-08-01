
from src.WFC.Tile import Tile
import numpy as np
from typing import List

class LinePath(Tile):
    def __init__(self, lines: List[str],color='blue'):
        """type you want

        Args:
            lines (List[str]): \na---ab---b
                               \n|        | 
                               \nda  cen   bc 
                               \n|        | 
                               \nd---cd---c 
                               \nuse a-b to define a line.
        """
        self.lines = lines
        self._properties={"color":color}
    def __repr__(self):
        return f"LinePath<{str(self.lines)}>"
    @property
    def properties(self):
        return self._properties

    def build(self, points:np.ndarray, *args, **kwargs):
        lines=self.lines
        p={"a": points[0,:],
            "b": points[1,:],
            "c": points[2,:],
            "d": points[3,:],
            "cen": np.mean(points, axis=0, keepdims=False),}
        
        p=p|{"ab": np.mean([p["a"],p["b"]],axis=0),
            "bc": np.mean([p["b"],p["c"]],axis=0),
            "cd": np.mean([p["c"],p["d"]],axis=0),
            "da": np.mean([p["a"],p["d"]],axis=0),
        }
        line=[[p["cen"],p["cen"]]]
        for direction in lines:
            result = direction.split('-')
            line.append([p[result[0]],p[result[1]]])
        return line


        