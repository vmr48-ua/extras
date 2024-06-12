import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint

# Desarrollo y despejo la ecuación de Euler-Lagrange
def triple_pendulum(z, t, par):
    z1, z2, z3, z4, z5, z6 = z  
    L1, L2, L3, m1, m2, m3, m123, g = par
    sin12 = np.sin(z1 - z2)
    cos12 = np.cos(z1 - z2)
    sin13 = np.sin(z1 - z3)
    cos13 = np.cos(z1 - z3)
    sin23 = np.sin(z2 - z3)
    cos23 = np.cos(z2 - z3)
    sinz1 = np.sin(z1)
    sinz2 = np.sin(z2)
    sinz3 = np.sin(z3)
    z42 = z4 * z4
    z52 = z5 * z5
    z62 = z6 * z6
    coszsq = cos12 * cos12

    dzdt = [
        z4,
        z5,
        z6,
        (-m2 * L1 * z42 * sin12 * cos12 + g * m2 * sinz2 * cos12 - m2 * L2 * z52 * sin12 - m123 * g * sinz1) / (L1 * m123 - m2 * L1 * coszsq),
        (m3 * L2 * z52 * sin23 * cos23 + g * sinz1 * cos12 * m123 + L1 * z42 * sin12 * m123 - g * sinz2 * m123) / (L2 * m123 - m3 * L2 * cos23 * cos23),
        (m3 * L3 * z62 * sin23 * cos23 + g * sinz1 * cos12 * m123 + L1 * z42 * sin12 * m123 - g * sinz3 * m123) / (L3 * m123 - m3 * L3 * cos23 * cos23)
    ]
    return dzdt

# Parámetros
L1 = 0.1  # longitud del péndulo 1
L2 = 0.1  # longitud del péndulo 2
L3 = 0.1  # longitud del péndulo 3
m1 = 1.0  # masa del péndulo 1
m2 = 1.0  # masa del péndulo 2
m3 = 1.0  # masa del péndulo 3
g = 9.81  # aceleración de la gravedad
tf = 10.0  # tiempo de simulación
m123 = m1 + m2 + m3
par = [L1, L2, L3, m1, m2, m3, m123, g]  # Resto de variables que necesitamos
nt = 10000  # número de intervalos de tiempo
dt = tf / nt
abserr = 1.0e-8
relerr = 1.0e-6

##################################################################################################################################
# ANÁLISIS DEL CAOS
##################################################################################################################################

# Ángulos iniciales y finales
th0 = np.linspace(0.0, np.pi, 200)
thF = np.zeros_like(th0)

# Integración para cada ángulo inicial
for i in range(200):
    theta1_0 = th0[i]
    theta2_0 = th0[i]
    theta3_0 = th0[i]
    
    z0 = [theta1_0, theta2_0, theta3_0, 0.0, 0.0, 0.0]  # Valores iniciales (velocidades iniciales nulas)
    t = np.linspace(0, tf, nt)
    z = odeint(triple_pendulum, z0, t, args=(par,), atol=abserr, rtol=relerr)
    thF[i] = z[-1, 0]  # Tomamos el ángulo final del primer péndulo

# Gráfico de evolución del sistema para distintos ángulos iniciales
plt.figure()
plt.plot(th0, thF, 'x', c='k')
plt.xlabel('$\\theta_{\\text{inicial}}$ (rad)')
plt.ylabel('$\\theta_{\\text{final}} $ (rad)')
plt.title('Evolución del sistema para distintos ángulos iniciales')

##################################################################################################################################

# Ángulos iniciales en grados
theta1_0_deg = 120
theta2_0_deg = -10
theta3_0_deg = 30
# Ángulos iniciales en radianes
theta1_0 = theta1_0_deg * np.pi / 180.0
theta2_0 = theta2_0_deg * np.pi / 180.0
theta3_0 = theta3_0_deg * np.pi / 180.0

z0 = [theta1_0, theta2_0, theta3_0, 0.0, 0.0, 0.0]  # Valores iniciales (velocidades iniciales nulas)
t = np.linspace(0, tf, nt)

z = odeint(triple_pendulum, z0, t, args=(par,), atol=abserr, rtol=relerr)

Llong = (L1 + L2 + L3) * 1.1

fig, ax3 = plt.subplots()
ax3.set_xlim(-Llong, Llong)
ax3.set_ylim(-Llong, Llong)
ax3.set_xlabel('x (m)')
ax3.set_ylabel('y (m)')
ax3.set_aspect('equal')

line1, = ax3.plot([], [], lw=2)
line2, = ax3.plot([], [], lw=2)
line3, = ax3.plot([], [], lw=2)
line4, = ax3.plot([], [], lw=1)
bob1 = plt.Circle((1, 1), Llong * 0.02, fc='b')
bob2 = plt.Circle((1, 1), Llong * 0.02, fc='r')
bob3 = plt.Circle((1, 1), Llong * 0.02, fc='g')
time_template = 'time = %.1fs'
time_text = ax3.text(0.05, 0.9, '', transform=ax3.transAxes)

def init():
    bob1.center = (1, 1)
    ax3.add_artist(bob1)
    bob2.center = (0, 0)
    ax3.add_artist(bob2)
    bob3.center = (0, 0)
    ax3.add_artist(bob3)
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    time_text.set_text('')
    return bob1, bob2, bob3, line1, line2, line3, line4, time_text

def animate(i):
    x1 = L1 * np.sin(z[i, 0])
    y1 = -L1 * np.cos(z[i, 0])
    x2 = x1 + L2 * np.sin(z[i, 1])
    y2 = y1 - L2 * np.cos(z[i, 1])
    x3 = x2 + L3 * np.sin(z[i, 2])
    y3 = y2 - L3 * np.cos(z[i, 2])
    
    bob1.center = (x1, y1)
    bob2.center = (x2, y2)
    bob3.center = (x3, y3)
    
    line1.set_data([0, x1], [0, y1])
    line2.set_data([x1, x2], [y1, y2])
    line3.set_data([x2, x3], [y2, y3])
    line4.set_data(L1 * np.sin(z[:i, 0]) + L2 * np.sin(z[:i, 1]) + L3 * np.sin(z[:i, 2]), 
                   -L1 * np.cos(z[:i, 0]) - L2 * np.cos(z[:i, 1]) - L3 * np.cos(z[:i, 2]))

    time_text.set_text(time_template % (i * dt))
    return bob1, bob2, bob3, line1, line2, line3, line4, time_text

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nt, interval=20, blit=True)

plt.show()