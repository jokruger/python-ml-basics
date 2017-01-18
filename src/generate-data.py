#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, random
import matplotlib.pyplot as plt

# generate
C1_X1 = []
C1_X2 = []
C2_X1 = []
C2_X2 = []
with open('data.csv', 'wb') as f:
    w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    w.writerow(['X1', 'X2', 'Y'])
    for i in range(1000):
        x1 = random.random()
        x2 = random.random()
        y = int(x1 > x2)
        w.writerow([x1, x2, y])
        if y == 0:
            C1_X1.append(x1)
            C1_X2.append(x2)
        else:
            C2_X1.append(x1)
            C2_X2.append(x2)

#display
plt.figure(figsize=(10,8))
plt.plot(C1_X1, C1_X2, lw=2, color='#5DA5DA', marker='o', ls='', label='Class 1')
plt.plot(C2_X1, C2_X2, lw=2, color='#FAA43A', marker='o', ls='', label='Class 2')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('X2')
plt.ylabel('X1')
plt.legend(loc="lower left", prop={'size': 8})
plt.show()
