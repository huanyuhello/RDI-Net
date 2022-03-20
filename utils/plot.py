import matplotlib.pyplot as plt

x1 = [0.98, 1.01, 1.4, 2.1, 3.8]
k1 = [91.4, 91.85, 92.4, 92.93, 93.5]
x2 = [0.8, 1.1, 1.25, 3.15]
k2 = [91.6, 91.8, 92.45, 93.4]
x3 = [0.58, 0.92]
k3 = [90.83, 92.38]

x6 = [4.1, 5.0]
k6 = [100-5.76, 100-5.14]

x4 = [0.9, 1.2, 1.55, 1.9]
k4 = [91.2, 92.3, 93.1, 93.6]

x5 = [1.02, 0.63 * 2.2, 0.67 * 2.2, 0.96 * 2.2, 1.07 * 2.2, 1.55 * 2.2, 1.81 * 2.2]
k5 = [92.4, 93.06, 93.21, 93.62, 93.72, 93.95, 94.10]

plt.plot(x1, k1,'s-.',color = 'orange',label="ACT", linewidth=3)
plt.plot(x2, k2,'+-.',color = 'purple',label="SACT", linewidth=3)
plt.plot(x3, k3,'x-.',color = 'green',label="SkipNet", linewidth=3)
plt.plot(x4, k4,'d-.',color = 'blue',label="DropBlock", linewidth=3)
plt.plot(x6, k6,'--.',color = 'brown',label="AIG", linewidth=3)
plt.plot(x5, k5,'o-',color = 'red',label="Ours", linewidth=3)

plt.xlabel("FLOPs(1e8)")
plt.ylabel("Accuracy(%)")
plt.legend(loc='best')
plt.grid(True, linestyle='-.')
plt.show()
