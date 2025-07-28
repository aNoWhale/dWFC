
from src.WFC.Tile import Tile
import numpy as np
from typing import List

class LinePath(Tile):
    def __init__(self, lines: List[str]):
        """type you want

        Args:
            lines (List[str]): a---ab---b
                               |        |
                              da  cen   bc
                               |        |
                               d---cd---c
                               use ()-() to define a line.
        """
        self.lines = lines

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
            "d": np.mean([p["a"],p["d"]],axis=0),
        }
        line=[p["cen"],p["cen"]],
        for direction in lines:
            result = direction.split('-')
            line.append([p[result[0]],p[result[1]]])
        return line
            # if direction=="ab,bc":
            #     line.append([ab,bc])
            # if direction=="ab-cd":
            #     line.append([ab,cd])
            # if direction=="ab-da":
            #     line.append([ab,da])
            
            # if direction=="bc-cd":
            #     line.append([bc,cd])
            # if direction == "bc-da":
            #     line.append([bc,da])

            # if direction=="cd-da":
            #     line.append([cd,da])
            
            # if direction=="a-b":
            #     line.append([a,b])
            # if direction=="a-c":
            #     line.append([a,c])
            # if direction=="a-d":
            #     line.append([a,d])
            
            # if direction=="b-c":
            #     line.append([b,c])
            # if direction=="b-d":
            #     line.append([b,d])

            # if direction=="c-d":
            #     line.append([c,d])

        