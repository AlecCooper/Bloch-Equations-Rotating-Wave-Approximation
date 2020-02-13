import numpy as np
import scipy.integrate as integrate
from math import cos
from matplotlib import pyplot as plt
from matplotlib import rc

# Timespan to integrate over
t_i = 0
t_f = 100

# Magnetization vector IV
m = np.array([0.0,0.0,1.0])

# Define our value of w_0 = w = w/w_r
w = 10.0

# LHS of our DE
def func(t,y):

    y_n = np.empty(3)

    y_n[0] = w*y[1]
    y_n[1] = 2*(1/w)*y[2]*cos(w*t)-w*y[0]
    y_n[2] = -2*(1/w)*y[1]*cos(w*t)

    return y_n

# Solve our DE
sol = integrate.solve_ivp(func,(t_i,t_f),m)

# Plot first M_z
plt.plot(sol["t"],sol["y"][2],"o")
plt.xlabel("time")
plt.ylabel(r"$M_z$")
plt.title(r"$\omega/\omega_0=10$")
plt.savefig("m_z_1")
plt.clf()

# Increase our ratio of w/w_r
w = 100.0
# Increase time period
t_f = 1000
# Recalculate
sol = integrate.solve_ivp(func,(t_i,t_f),m)
plt.plot(sol["t"],sol["y"][2],"o")
plt.xlabel("time")
plt.ylabel(r"$M_z$")
plt.title(r"$\omega/\omega_0=100$")
plt.savefig("m_z_2")
plt.clf()
# Increase our ratio of w/w_r
w = 500
# Recalculate
sol = integrate.solve_ivp(func,(t_i,t_f),m)
plt.plot(sol["t"],sol["y"][2],"o")
plt.xlabel("time")
plt.ylabel(r"$M_z$")
plt.title(r"$\omega/\omega_0=500$")
plt.savefig("m_z_3")

print("done!")