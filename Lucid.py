import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

import matplotlib.pyplot as plt

def printNeurons(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            plt.imshow(data[i][j])
            plt.show()

model = models.InceptionV1()
model.load_graphdef()

data = render.render_vis(model, "mixed4a_pre_relu:476")
printNeurons(data)

for i in range(450,460):
    data = render.render_vis(model, "mixed4a_pre_relu:%s" % i)
    printNeurons(data)

print("end.")