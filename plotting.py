import matplotlib.pyplot as plot
from AppKit import NSScreen

class ScreenInfo:
    pass

def getScreensInfo():
    screens = []

    for i, s in enumerate(NSScreen.screens()):
        screen = ScreenInfo
        frame = s.frame()
        screen.x = frame.origin.x
        screen.y = frame.origin.y
        screen.w = frame.size.width
        screen.h = frame.size.height
        screens.append(screen)
    return screens

class PlottingHelper:
    def __init__(self, width = 800, height = 900, for_print = False):
        plot.ion()
        screens = getScreensInfo()
        target_screen = screens[-1]
        wm = plot.get_current_fig_manager()
        if for_print:
            wm.window.wm_geometry("1200x650+0+0")
        else:
            wm.window.wm_geometry("%dx%d+%d+%d" % (target_screen.w, target_screen.h, target_screen.x, target_screen.y))

    def getPlotter(self):
        return plot

    def quiver(self, x, y, u, v, color="black", label=None):
        try:
            x+y+u+v
        except ValueError:
            raise Exception("The vector positions and directions should have the same dimensions")

        #self.get_plotter().quiver(x, y, u, v, scale=0.0, color=color, label=label)
        self.getPlotter().quiver(x, y, u, v, color=color, label=label)

    def wireframe(self, x, y, z):
        from mpl_toolkits.mplot3d import axes3d
        plot = self.getPlotter()
        ax = plot.subplot(111, projection='3d')
        ax.plot_wireframe(x, y, z)



