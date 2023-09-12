import math

x = 1/2
y = x
z = x
v = x
M = (-5 - 2*math.exp(-x-2*y-2*z-2*v) - 2*math.exp(-2*x-2*y-z-2*v) +
    2*math.exp(-z-2*v-2*x) - 2*math.exp(-2*x-2*z-2*v-y) -
    2*math.exp(-z-2*v) + 2*math.exp(-2*y-z-2*v) +
    2*math.exp(-2*v-y-2*x) + 2*math.exp(-2*y-2*z-v) -
    2*math.exp(-2*x-2*y-2*z-v) + 2*math.exp(-2*y-x-2*v) +
    2*math.exp(-2*z-v-2*x ) + 3*math.exp(-2*x) +
    2*math.exp(-x) - 2*math.exp(-v-2*y) +
    2*math.exp(-2*y-2*x-v) + 2*math.exp(-2*z-2*v-y) -
    2*math.exp(-v-2*x) - 2*math.exp(-y-2*v) -
    math.exp(-2*z-2*v) - 2*math.exp(-2*z-v) -
    math.exp(-2*y-2*z-2*v) + 3*math.exp(-2*v) -
    math.exp(-2*v-2*y) + 2*math.exp(-2*y-x-2*z) +
    2*math.exp(-z) + 2*math.exp(-y) -
    2*math.exp(-2*y-x) - 2*math.exp(-x-2*z) +
    2*math.exp(-2*z-2*x-y) - 2*math.exp(-y-2*z) -
    2*math.exp(-2*x-y) + 3*math.exp(-2*z) +
    3*math.exp(-2*y) - 2*math.exp(-2*y-z) -
    2*math.exp(-2*v-x) -math.exp(-2*y-2*z) +
    2*math.exp(-2*x-2*y-z) + 2*math.exp(-x-2*z-2*v) +
    2*math.exp(-v) - 2*math.exp(-z-2*x) -
    math.exp(-2*x-2*z) - math.exp(-2*x-2*y ) -
    math.exp(-2*x-2*y-2*z) + 3*math.exp(-2*x-2*y-2*z-2*v) - 
    math.exp(-2*z-2*v-2*x) - math.exp(-2*y-2*x-2*v) - math.exp(-2*v-2*x))/(-1 + 
    math.exp(-2*x) - math.exp(-2*z-2*v) + math.exp(-2*y-2*z-2*v) +
    math.exp(-2*v) - math.exp(-2*v-2*y) + math.exp(-2*z) + math.exp(-2*y) -
    math.exp(-2*y-2*z) - math.exp(-2*x-2*z) - math.exp(-2*x-2*y) + math.exp(-2*x-2*y-2*z) -
    math.exp(-2*x-2*y-2*z-2*v) + math.exp(-2*z-2*v-2*x) + math.exp(-2*y-2*x-2*v) -
    math.exp(-2*v-2*x))


print(M)