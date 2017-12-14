import numpy as np
import matplotlib.pyplot as plt

def fig2img(fig, dpi=None, closefig=True):
    if dpi is not None:
        fig.set_dpi(dpi)
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if closefig:
        plt.close(fig)
    return data
