import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Time
T = np.linspace(0, 150, 1500)

# Parameter Values
V0 = -65  # milliVolt
Cm = 1.0  # microFarad/cm2
ENa = 50  # milliVolt
Ek = -77  # milliVolt
El = -54.4  # milliVolt
gNa = 120  # milliSiemens/cm2
gK = 36  # milliSiemens/cm2
gL = 0.3  # milliSiemens/cm2


# Rate coefficient for K
def alpha_n(v):
    return 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10))


def beta_n(v):
    return 0.125 * np.exp(-(v + 65) / 80)


# Rate coefficient for Na
def alpha_m(v):
    return 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10))


def beta_m(v):
    return 4 * np.exp(-(v + 65) / 18)


def alpha_h(v):
    return 0.07 * np.exp(-(v + 65) / 20)


def beta_h(v):
    return 1 / (np.exp(-(v + 35) / 10) + 1)


# Steady state values.
steady_n = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
steady_m = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
steady_h = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))

x0 = [V0, steady_n, steady_m, steady_h]


# Differential equations
def hhm_derivatives(x, t):
    I = 0
    if 100 > t > 50:
        I = 10

    # Gating particles
    # Na activation gating variable
    dmdt = alpha_m(x[0]) * (1 - x[2]) - (beta_m(x[0]) * x[2])

    # Na inactivation gating variable
    dhdt = alpha_h(x[0]) * (1 - x[3]) - (beta_h(x[0]) * x[3])

    # K activation gating variable
    dndt = alpha_n(x[0]) * (1 - x[1]) - (beta_n(x[0]) * x[1])

    IL = -gL * (x[0] - El)
    INa = -gNa * x[2] ** 3 * x[3] * (x[0] - ENa)
    IK = -gK * x[1] ** 4 * (x[0] - Ek)

    dvdt = (I + IL + INa + IK) / Cm

    return [dvdt, dndt, dmdt, dhdt]


x = odeint(hhm_derivatives, x0, T)

volt = x[:, 0]
n_part = x[:, 1]
m_part = x[:, 2]
h_part = x[:, 3]

fig, ax = plt.subplots(4, 1)
ax[0].plot(T, volt, 'r')
ax[0].set_xlabel("Time")
ax[0].set_ylabel("V")

ax[1].plot(T, n_part, 'g')
ax[1].set_xlabel("Time")
ax[1].set_ylabel("n")

ax[2].plot(T, m_part, 'm')
ax[2].set_xlabel("Time")
ax[2].set_ylabel("m")

ax[3].plot(T, h_part, 'k')
ax[3].set_xlabel("Time")
ax[3].set_ylabel("h")

plt.tight_layout()
plt.show()
