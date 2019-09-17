import pickle
with open("saveData/20_img", "rb") as data:
    fig, ims = pickle.load(data)

import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt



def animate(fig, ims):
    ani = animation.ArtistAnimation(fig, ims, interval = 100, blit=True,
                                    repeat_delay = 8)

    writer = PillowWriter(fps = 20)
    ani.save("/Users/momolee/Downloads/demo.gif", writer = 'imagemagick')

animate(fig, ims)