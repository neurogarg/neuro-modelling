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
EA = -80  # milliVolt
El = -22  # milliVolt
gNa = 120  # milliSiemens/cm2
gK = 20  # milliSiemens/cm2
gA = 47.7  # milliSiemens/cm2
gL = 0.3  # milliSiemens/cm2


# Rate coefficient for K
def alpha_n(v):
    return (3.8 / 2) * -0.01 * (v + 50.7) / ((np.exp(-(v + 50.7) / 10)) - 1)


def beta_n(v):
    return (3.8 / 2) * 0.125 * np.exp(-(v + 60.7) / 80)


# Rate coefficient for Na
def alpha_m(v):
    return 3.8 * -0.1 * (v + 34.7) / ((np.exp(-(v + 34.7) / 10)) - 1)


def beta_m(v):
    return 3.8 * 4 * np.exp(-(v + 59.7) / 18)


def alpha_h(v):
    return 3.8 * 0.07 * np.exp(-(v + 53) / 20)


def beta_h(v):
    return 3.8 / (np.exp(-(v + 23) / 10) + 1)


# time constants for IA current
def tau_a(v):
    return 0.3632 + 1.158 / (1 + np.exp((v + 60.96) / 20.12))


def tau_b(v):
    return 1.24 + 2.678 / (1 + np.exp((v + 55) / 16.027))


def steady_a(v):
    return ((0.0761 * np.exp((v + 99.22) / 31.84)) / (1 + np.exp((v + 6.17) / 28.93))) ** (1 / 3)


def steady_b(v):
    return 1 / ((1 + np.exp((v + 58.3) / 14.54)) ** 4)


# Steady state values.
steady_n = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
steady_m = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
steady_h = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
steady_con_a = steady_a(V0)
steady_con_b = steady_b(V0)

x0 = [V0, steady_n, steady_m, steady_h, steady_con_a, steady_con_b]


# Differential equations
def csm_derivatives(x, t):
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

    # IA currents
    dadt = (steady_a(x[0]) - x[4]) / tau_a(x[0])

    dbdt = (steady_b(x[0]) - x[5]) / tau_b(x[0])

    IL = -gL * (x[0] - El)
    INa = -gNa * x[2] ** 3 * x[3] * (x[0] - ENa)
    IK = -gK * x[1] ** 4 * (x[0] - Ek)
    IA = -gA * (x[4] ** 3) * x[5] * (x[0] - EA)

    dvdt = (I + IL + INa + IK + IA) / Cm

    return [dvdt, dndt, dmdt, dhdt, dadt, dbdt]


x = odeint(csm_derivatives, x0, T)

volt = x[:, 0]
n_part = x[:, 1]
m_part = x[:, 2]
h_part = x[:, 3]
a = x[:, 4]
b = x[:, 5]

fig, ax = plt.subplots(3, 2)
ax[0, 0].plot(T, volt, 'r')
ax[0, 0].set_xlabel("Time")
ax[0, 0].set_ylabel("V")

ax[1, 0].plot(T, n_part, 'g')
ax[1, 0].set_xlabel("Time")
ax[1, 0].set_ylabel("n")

ax[2, 0].plot(T, m_part, 'm')
ax[2, 0].set_xlabel("Time")
ax[2, 0].set_ylabel("m")

ax[0, 1].plot(T, h_part, 'k')
ax[0, 1].set_xlabel("Time")
ax[0, 1].set_ylabel("h")

ax[1, 1].plot(T, a, 'y')
ax[1, 1].set_xlabel("Time")
ax[1, 1].set_ylabel("a")

ax[2, 1].plot(T, b, 'c')
ax[2, 1].set_xlabel("Time")
ax[2, 1].set_ylabel("b")

plt.tight_layout()
plt.show()
