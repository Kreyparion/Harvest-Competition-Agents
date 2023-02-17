from env.environnement import Env
import numpy as np
import time
env = Env(42)
t = time.time()
env.reset()
# load predictions.npy
npy_file = np.load("predictions47223.npy")
for row in npy_file:
    env.step(int(row))
print(f"score:{env.score}")
print(time.time()-t)
env.final_render()