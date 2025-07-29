import matplotlib.pyplot as plt
import numpy as np  

from src.WFC.TileHandler import TileHandler
ax=plt.subplot()
def visualizer_2D(tileHandler:TileHandler,probs:np.array,points:np.array,ax:plt.axes=ax):
    probs = np.array(probs)
    probs = probs.reshape(-1,tileHandler.typeNum)
    points = np.array(points).reshape(-1,2)
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            typeName=tileHandler._index_to_name[j]
            lines=tileHandler.typeMethod[typeName].build(points=points[i,:],)
            for line in lines:
                ax.plot([line[0][0],line[1][0]],
                        [line[0][1],line[1][1]],
                        linewidth=2,color='blue')
