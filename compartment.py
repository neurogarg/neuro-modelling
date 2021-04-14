# TODO: Might not be implemented accurately.

# implementation of a simple compartment equation
import numpy as np
from numpy import ndarray
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# parameter values
V0 = -65  # mV resting potential
Cm = 1.0  # microFarad Capacitance
l = 50  # micro meter length
x = 5  # segment
lx = l / x
G_plasma = 10.0  # milli Siemens Conductivity
G_membrane = 0.3  # milli Siemens Conductivity
r = 10  # miro meter radius

v_each_segment = np.ones(x) * V0


# voltage differences
def diff(v):
    v_diff_left = np.ediff1d(v, to_begin=0)  # sealed end
    v_diff_right = - np.ediff1d(v, to_end=0)  # sealed end
    return v_diff_left, v_diff_right


def derivatives(v_seg, t):
    v_diff_left, v_diff_right = diff(v_seg)
    length = len(v_seg)
    v_seg_new = np.zeros(length)
    for i in range(length):
        # membrane + axial
        I = (V0 - v_seg[i]) * G_membrane / Cm + ((r * G_plasma / (2 * lx * lx)) * (v_diff_left[i] - v_seg[i] +
                                                                                   v_diff_right[i] - v_seg[i])) / Cm
        v_seg_new[i] = I
    v_seg_new[0] = +50 / (Cm * np.pi * 2 * r * lx)  # Injected current
    return v_seg_new


t = np.linspace(0, 150, 1500)
v = odeint(derivatives, v_each_segment, t)

# voltage in diff segments
segment1 = v[:, 0]
segment2 = v[:, 1]
segment3 = v[:, 2]
segment4 = v[:, 3]
segment5 = v[:, 4]

# plots
fig, ax = plt.subplots(5, 1)
ax[0].plot(t, segment1, 'r')
ax[1].plot(t, segment2, 'g')
ax[2].plot(t, segment3, 'm')
ax[3].plot(t, segment4, 'k')
ax[4].plot(t, segment5)

plt.tight_layout()
plt.show()
