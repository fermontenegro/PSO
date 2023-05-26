import numpy as np
import math
import time

x = 2/3
y = x
z = x

M = 2*(-2 + math.exp(-z) + math.exp(-x) + math.exp(-y) + math.exp(-2*y-x-2*z) + math.exp(-2*x) + math.exp(-2*y) + math.exp(-2*z) - math.exp(-2*x-2*y-2*z) - math.exp(-x -2*z) + math.exp(-2*x -2*y -z) - math.exp(-z -2*x) - math.exp(-y -2*z) -math.exp(-2*x -y) - math.exp(-2*y -z) + math.exp(-2*z -2*x -y) - math.exp(-2*y -x))/(-1 + math.exp(-2*z) + math.exp(-2*y)- math.exp(-2*y -2*z) + math.exp(-2*x) - math.exp(-2*x -2*z) - math.exp(-2*x -2*y) + math.exp(-2*x -2*y -2*z))

print("valor M: ",M)