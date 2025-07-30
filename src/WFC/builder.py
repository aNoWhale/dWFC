
import numpy as np  
from src.WFC.FigureManager import FigureManager
from src.WFC.TileHandler import TileHandler

def visualizer_2D(tileHandler:TileHandler,probs:np.array,points:np.array,figureManager:FigureManager=None,epoch=0):
    assert figureManager is not None
    fig,ax=figureManager.get_figure_ax()
    print(f"{__name__}")
    probs = np.array(probs)
    probs = probs.reshape(-1,tileHandler.typeNum)
    points = np.array(points)
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            typeName=tileHandler._index_to_name[j]
            lines=tileHandler.typeMethod[typeName].build(points=points[i,:],)
            for line in lines:
                ax.plot([line[0][0],line[1][0]],
                        [line[0][1],line[1][1]],
                        linewidth=20,color='blue',alpha=probs[i,j])
                
    figureManager.save(f'data/img/{epoch}.jpg')
    ax.cla()
