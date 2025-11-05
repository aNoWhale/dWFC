
import numpy as np  
from src.WFC.FigureManager import FigureManager
from src.WFC.TileHandler_JAX import TileHandler
import tqdm

def visualizer_2D(tileHandler:TileHandler,probs:np.array,points:np.array,figureManager:FigureManager=None,epoch=0,prefix:str="",*args,**kwargs):
    assert figureManager is not None
    fig,ax=figureManager.get_figure_ax()
    # print(f"{__name__}")
    probs = np.array(probs)
    probs = probs.reshape(-1,tileHandler.typeNum)
    points = np.array(points)

    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            typeName=tileHandler._index_to_name[j]
            lines=tileHandler.typeMethod[typeName].build(points=points[i,:],)
            color=tileHandler.typeMethod[typeName].properties["color"]
            for line in lines:
                ax.plot([line[0][0],line[1][0]],
                        [line[0][1],line[1][1]],
                        linewidth=10,color=color,alpha=probs[i,j])
    ax.set_title(f"epoch:{epoch}, at:{kwargs.pop('at','unknow')}")
    figureManager.save(f'data/img/{prefix}{epoch}.jpg')
    ax.cla()

class Visualizer:
    def __init__(self,tileHandler:TileHandler,mode='2d',*args,**kwargs):
        self.tileHandler = tileHandler
        self.points=kwargs.pop("points",None)
        self.figureManager=kwargs.pop("figureManager",None)
        self.probs=[]
        self.collapse_list=[]
    
    def add_frame(self,probs):
        self.probs.append(probs)

    def draw(self,prefix:str=""):
        for i,prob in tqdm.tqdm(enumerate(self.probs),desc="ploting",total=len(self.probs)):
            visualizer_2D(tileHandler=self.tileHandler,probs=prob,points=self.points,figureManager=self.figureManager,epoch=i,prefix=prefix,at=self.collapse_list[i])
