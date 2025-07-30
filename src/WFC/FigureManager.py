import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FigureManager:
    _instance = None
    fig, ax = None, None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls.fig, cls.ax = plt.subplots()
            cls.ax.set_title("Managed Shared Figure")
        return cls._instance
    
    def get_figure_ax(self):
        return self.fig, self.ax
    
    def save(self, filename):
        self.fig.savefig(filename)