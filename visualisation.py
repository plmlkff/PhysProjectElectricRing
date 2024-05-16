import sys
from PyQt5.Qt import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
from scipy.spatial import distance
from enum import Enum

from numba import jit

X_PROP = 0
Y_PROP = 1
CHARGE_PROP = 2
ENABLE_PARTICLE_PROP = 3


class GraphType(Enum):
    TENSION = 'Напряженность'
    POWER_LINES = 'Силовые линии'
    PHI = 'Фи'


class PyQtGraph(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle('PyQt Graph')

        # Инициализировать макет
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setSpacing(15)

        # Виджет со слайдерами и лейблами
        self.slidersVBOX = QVBoxLayout()
        self.layout.addLayout(self.slidersVBOX)

        # Создать слайдеры
        self.slider_radius = QSlider(Qt.Horizontal)
        self.slider_radius.setRange(1, 100)
        self.slider_radius.setValue(50)
        self.slider_radius.valueChanged.connect(self.update)

        self.slider_charge = QSlider(Qt.Horizontal)
        self.slider_charge.setRange(-100, 100)
        self.slider_charge.setValue(1)
        self.slider_charge.valueChanged.connect(self.update)

        self.slider_lines_count = QSlider(Qt.Horizontal)
        self.slider_lines_count.setRange(1, 70)
        self.slider_lines_count.setValue(40)
        self.slider_lines_count.valueChanged.connect(self.update)

        # Создать лейблы
        self.lable_radius = QLabel()
        self.lable_radius.setText(str(self.slider_radius.value()))
        self.lable_radius.setAlignment(Qt.AlignCenter)

        self.lable_charge = QLabel()
        self.lable_charge.setText(str(self.slider_charge.value()))
        self.lable_charge.setAlignment(Qt.AlignCenter)

        self.lable_lines_count = QLabel()
        self.lable_lines_count.setText(str(self.slider_lines_count.value()))
        self.lable_lines_count.setAlignment(Qt.AlignCenter)

        # Создать комбо-бокс
        self.comboBox = QComboBox()
        self.comboBox.setPlaceholderText("Выберите график")
        self.comboBox.addItems([GraphType.PHI.value, GraphType.TENSION.value, GraphType.POWER_LINES.value])
        self.comboBox.activated.connect(self.update)

        # Создать область для графика
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # Добавить элементы управления в макет
        self.slidersVBOX.addWidget(self.lable_radius)
        self.slidersVBOX.addWidget(self.slider_radius)
        self.slidersVBOX.addWidget(self.lable_charge)
        self.slidersVBOX.addWidget(self.slider_charge)
        self.layout.addWidget(self.comboBox)
        self.layout.addWidget(self.canvas)

        # Инициализировать график
        self.update()

    def _show_lines_count(self):
        self.slidersVBOX.addWidget(self.lable_lines_count)
        self.slidersVBOX.addWidget(self.slider_lines_count)
        self.lable_lines_count.show()
        self.slider_lines_count.show()

    def _hide_lines_count(self):
        self.slidersVBOX.removeWidget(self.lable_lines_count)
        self.slidersVBOX.removeWidget(self.slider_lines_count)
        self.lable_lines_count.hide()
        self.slider_lines_count.hide()

    def update(self):
        self._update_lables()
        if self.comboBox.currentText() == GraphType.PHI.value:
            self._hide_lines_count()
            self._draw_phi()
        if self.comboBox.currentText() == GraphType.TENSION.value:
            self._hide_lines_count()
            self._draw_e()
        if self.comboBox.currentText() == GraphType.POWER_LINES.value:
            self._show_lines_count()
            self._draw_lines()
        # Добавить легенду и показать график
        if self.comboBox.currentText() != '':
            plt.legend()
            self.canvas.draw()

    def _update_lables(self):
        self.lable_radius.setText(str(self.slider_radius.value()))
        self.lable_charge.setText(str(self.slider_charge.value()))
        self.lable_lines_count.setText(str(self.slider_lines_count.value()))

    def _draw_phi(self):
        self.figure.clear()
        x = np.linspace(-50, 50, 100)
        y = phi(2e-9 * self.slider_charge.value(), self.slider_radius.value(), x)
        plt.plot(x, y, label="Фи")

    def _draw_e(self):
        self.figure.clear()
        x = np.linspace(-50, 50, 100)
        y = e(2e-9 * self.slider_charge.value(), self.slider_radius.value(), x)
        plt.plot(x, y, label="Напряженность")

    def _draw_lines(self):
        self.figure.clear()
        size = 100  # size of plot
        particles = np.array([[0, -self.slider_radius.value(), self.slider_charge.value(), 1],
                              [0, self.slider_radius.value(), self.slider_charge.value(), 1]])
        lines, particle_radius = compute_lines(particles, size, self.slider_lines_count.value())
        plt.xlim(-(size + 1), (size + 1))
        plt.ylim(-(size + 1), (size + 1))

        plt.gca().set_aspect('equal', adjustable='box')
        for i in lines:
            for j in i:
                plt.plot(j[0], j[1], color="r", linewidth=0.8)
        plt.scatter(particles[:, 0], particles[:, 1], s=particle_radius * 30)
        for particle in particles:
            if particle[2] > 0:
                plt.text(particle[0], particle[1], "+", fontsize=20)
            else:
                plt.text(particle[0], particle[1], "-", fontsize=20)


def e(q, r, x):
    y = []
    for x_i in x:
        y.append(
            q * x_i / (4 * math.pi * 8.8541878128e-12 * math.sqrt(r * r + x_i * x_i) * (r * r + x_i * x_i)))
    return y


def phi(q, r, x):
    y = []
    for x_i in x:
        y.append(q / (4 * math.pi * 8.8541878128e-12 * math.sqrt(r * r + x_i * x_i)))
    return y


def invert_charge(particle):
    particle[CHARGE_PROP] *= -1
    return particle


def compute_lines(particles, size, lines_amount):
    # [x, y, charge, enable lines for the particle]
    # charge is expressed in multiples of base charge
    particles = np.array(particles)

    particle_radius = np.sqrt(15)

    # draw line around charge in a radial pattern
    # theta denotes angle between neighbor lines
    thetas = np.linspace(0, 2 * np.pi, lines_amount)
    # by default draw only for positive charges
    positive_charges = particles[particles[:, CHARGE_PROP] > 0]

    if len(positive_charges) < 1:
        particles = np.array(list(map(invert_charge, particles)))
        positive_charges = particles

    all_lines = []
    for particle in positive_charges:
        # points on surface of charge
        x_surface = particle_radius * np.cos(thetas) + particle[X_PROP]
        y_surface = particle_radius * np.sin(thetas) + particle[Y_PROP]
        particle_lines = []
        for xs, ys in tqdm(zip(x_surface, y_surface)):
            straight_lines = [[xs], [ys]]

            count = 0
            while abs(xs) < size and abs(ys) < size and (
                    np.all((particle_radius - 0.001) < distance.cdist([[xs, ys]], particles[:, 0:2]))):
                dx, dy = E1(particles, xs, ys)
                xs += dx
                ys += dy
                straight_lines[0].append(xs)
                straight_lines[1].append(ys)
                count += 1

            particle_lines.append(straight_lines)

        all_lines.append(particle_lines)
    return all_lines, particle_radius


@jit
def E1(particles, xs: float, ys: float):  # для n-ого числа зарядов
    dx = 0
    dy = 0

    # computes how much every particle contributes to
    # strength of electric field at given point (xs, ys)
    for index in range(len(particles)):
        particle = particles[index]
        # math reveals it is absolutely necessary to calculate distance
        # it's impossible to simply formula by discarding 2d distance away:
        # strength by x and by y are interdependent on each other
        distance = ((xs - particle[0]) ** 2.0 + (ys - particle[1]) ** 2.0) ** 0.5
        vector = (9 * particle[CHARGE_PROP]) / distance ** 2

        # divide by distance to obtain direction (unit) vector.
        # The vector points away/into current particle
        x_component = (xs - particle[X_PROP]) / distance * vector
        y_component = (ys - particle[Y_PROP]) / distance * vector

        dx += x_component
        dy += y_component

    # compute unit vector again to make even steps
    length = (dx ** 2 + dy ** 2) ** 0.5
    dx /= length
    dy /= length
    return dx, dy


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pg = PyQtGraph()
    pg.show()
    sys.exit(app.exec_())
