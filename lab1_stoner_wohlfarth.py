# -*- coding: utf-8 -*-

# Лабораторная работа №1
# Модель Стонера–Вольфарта

import sys
import subprocess

# Автоматическая установка библиотек, если их нет
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Устанавливаю необходимые библиотеки...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "numpy",
        "matplotlib"
    ])
    import numpy as np
    import matplotlib.pyplot as plt

# ОСНОВНЫЕ ФОРМУЛЫ

def energy(phi, h, theta):
    return 0.5 * np.sin(phi) ** 2 - h * np.cos(theta - phi)


def d2_energy(phi, h, theta):
    return np.cos(2 * phi) + h * np.cos(theta - phi)


def angular_distance(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))


def stable_minima(h, theta):
    phi_grid = np.linspace(-np.pi, np.pi, 4000, endpoint=False)

    e = energy(phi_grid, h, theta)

    e_left = np.roll(e, 1)
    e_right = np.roll(e, -1)

    mask = (
        (e <= e_left)
        & (e <= e_right)
        & ((e < e_left) | (e < e_right))
    )

    minima = phi_grid[mask]

    if len(minima) == 0:
        minima = np.array([phi_grid[np.argmin(e)]])

    stable = minima[d2_energy(minima, h, theta) > -1e-6]

    if len(stable) == 0:
        stable = minima

    return stable

# ПЕТЛИ ГИСТЕРЕЗИСА

def trace_branch(h_values, theta):

    if h_values[0] >= 0:
        previous_phi = theta
    else:
        previous_phi = theta + np.pi

    m_values = []

    for h in h_values:

        minima = stable_minima(h, theta)

        distances = angular_distance(
            minima,
            previous_phi
        )

        current_phi = minima[np.argmin(distances)]

        m = np.cos(theta - current_phi)

        m_values.append(m)

        previous_phi = current_phi

    return np.array(m_values)


def plot_hysteresis():

    h_down = np.linspace(2, -2, 601)
    h_up = np.linspace(-2, 2, 601)

    angles = [0, 30, 60, 90]

    plt.figure(figsize=(9, 7))

    for angle in angles:

        theta = np.deg2rad(angle)

        m_down = trace_branch(h_down, theta)
        m_up = trace_branch(h_up, theta)

        h_all = np.concatenate([
            h_down,
            [np.nan],
            h_up
        ])

        m_all = np.concatenate([
            m_down,
            [np.nan],
            m_up
        ])

        plt.plot(
            h_all,
            m_all,
            linewidth=2,
            label=f"{angle}°"
        )

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)

    plt.xlabel("Безразмерное поле h")
    plt.ylabel("Намагниченность m")

    plt.title(
        "Петли гистерезиса модели Стонера–Вольфарта"
    )

    plt.grid()
    plt.legend(title="Угол θ")
    plt.tight_layout()
    plt.show()
# АСТРОИДА

def plot_astroid():

    phi = np.linspace(0, 2 * np.pi, 1000)

    hx = -(np.sin(phi) ** 3)
    hz = np.cos(phi) ** 3

    plt.figure(figsize=(7, 7))

    plt.plot(hx, hz, linewidth=2)

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)

    plt.xlabel("hx")
    plt.ylabel("hz")

    plt.title(
        "Астроида Стонера–Вольфарта"
    )

    plt.axis("equal")
    plt.grid()
    plt.tight_layout()
    plt.show()


# =========================
# КРИТИЧЕСКОЕ ПОЛЕ
# =========================

def plot_critical_field():

    theta_deg = np.linspace(0, 90, 500)

    theta = np.deg2rad(theta_deg)

    hcrit = 1 / (
        np.sin(theta) ** (2 / 3)
        +
        np.cos(theta) ** (2 / 3)
    ) ** (3 / 2)

    plt.figure(figsize=(9, 6))

    plt.plot(
        theta_deg,
        hcrit,
        linewidth=2
    )

    plt.xlabel("Угол θ, градусы")
    plt.ylabel("hcrit")

    plt.title(
        "Критическое поле Стонера–Вольфарта"
    )

    plt.grid()
    plt.tight_layout()
    plt.show()


# =========================
# ГЛАВНАЯ ПРОГРАММА
# =========================

def main():

    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №1")
    print("Модель Стонера–Вольфарта")
    print("=" * 60)

    print()
    print("Контрольное значение:")
    print("hcrit(45°) = 0.5")
    print()

    print("1. Строим петли гистерезиса...")
    plot_hysteresis()

    print("2. Строим астроиду...")
    plot_astroid()

    print("3. Строим критическое поле...")
    plot_critical_field()

    print()
    print("Готово.")


if __name__ == "__main__":
    main()
