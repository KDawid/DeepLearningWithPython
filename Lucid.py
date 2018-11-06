import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform


model = models.InceptionV1()
model.load_graphdef()

from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D

model.show_graph()

data = render.render_vis(model, "mixed4a_pre_relu:476")

print("end.")