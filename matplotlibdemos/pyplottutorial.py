import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

# plot x and y, last parameter is type of line 'b-' blue line, 'ro' red dots
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')

# takes a list of [xmin, xmax, ymin, ymax] and specifies the viewport of the axes
plt.axis([0, 6, 0, 20])
plt.ylabel('numbers')
plt.show()

# plot 3 data series on one graph
data = np.arange(0., 5., 0.2)
plt.plot(data, data, 'r--', data, data**2, 'bs', data, data**3, 'g^')
plt.show()

# set up dummy data
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

# access dictionary data via strings
plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()


# plotting categorical values
labels = ['group a', 'group b', 'group c']
values = [1, 10, 100]

plt.figure(figsize=(9, 3))

# plt subplot takes 3 params as a 3 digit int or separate ints: nrows, ncols and index
plt.subplot(131)
plt.bar(labels, values)
plt.subplot(132)
plt.scatter(labels, values)
plt.subplot(133)
plt.plot(labels, values)
plt.suptitle('Categorical Plotting')
plt.show()

# multiple figures and axes-


def do_math(num):
    return np.exp(-num) * np.cos(2 * np.pi * num)


num1 = np.arange(0.0, 5.0, 0.1)
num2 = np.arange(0.0, 5.0, 0.02)

plt.figure()  # not needed if only one figure
plt.subplot(211)  # 2 rows, 1 column, index 1
plt.plot(num1, do_math(num1), 'bx', num2, do_math(num2), 'k')

plt.subplot(212)
plt.plot(num2, np.cos(2 * np.pi * num2), 'r-')
plt.suptitle('wiggly lines')
plt.show()

# making plots with multiple figures
plt.figure(1)
plt.subplot(211)
plt.plot([1, 2, 3])
plt.subplot(212)
plt.plot([1, 3, 5])

plt.figure(2)
plt.plot([2, 4, 6])
plt.show()

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)

# text
plt.xlabel('Smarts', fontsize=14, color='pink')
plt.ylabel('Probability', fontsize=14, color='purple')
plt.title('Histogram of IQ', color='green')
plt.text(60, .022, r'$\mu=100,\ \sigma=15$')
plt.annotate('maximum', xy=(100, 0.027), xytext=(120, 0.027),
             arrowprops=dict(facecolor='pink', shrink=0.05))
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()


# logarithmic and non-linear axes

# dummy data from 0-1
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

plt.figure()

plt.subplot(221)
plt.plot(x, y, 'b-')
plt.yscale('linear')
plt.title('Linear', color='blue')
plt.grid(True)

plt.subplot(222)
plt.plot(x, y, 'r-')
plt.yscale('symlog', linthreshy=0.01)
plt.title('Symmetric Log', color='red')
plt.grid(True)

plt.subplot(223)
plt.plot(x, y, 'm-')
plt.yscale('log')
plt.title('Log', color='purple')
plt.grid(True)

plt.subplot(224)
plt.plot(x, y, 'g-')
plt.yscale('logit')
plt.title('Logit', color='green')
plt.grid(True)

# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()


# https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py
