#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, random, math
import matplotlib.pyplot as plt

C1_X1 = []
C1_X2 = []
C2_X1 = []
C2_X2 = []

# generate
with open('data.csv', 'wb') as f:
    w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    w.writerow(['X1', 'X2', 'Y'])
    for i in range(10000):
        x1 = random.random()
        x2 = random.random()

        y = x1 > x2
        #y = (x1-.5)**2 + (x2-.1)**2 > 0.25
        #y = ((x1-.5)**2 + (x2-.5)**2 > 0.01) and ((x1-.25)**2 + (x2-.25)**2 > 0.005) and ((x1-.75)**2 + (x2-.25)**2 > 0.005) and ((x1-.5)**2 + (x2-.75)**2 > 0.03)

        w.writerow([x1, x2, int(y)])

        if y == 0:
            C1_X1.append(x1)
            C1_X2.append(x2)
        else:
            C2_X1.append(x1)
            C2_X2.append(x2)

#display
plt.figure(figsize=(10,8))
plt.plot(C1_X1, C1_X2, lw=2, color='#FAA43A', marker='o', ls='')
plt.plot(C2_X1, C2_X2, lw=2, color='#5DA5DA', marker='o', ls='')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('X2')
plt.ylabel('X1')
plt.show()
