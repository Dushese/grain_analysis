import numpy as np
from matplotlib import pyplot as plt

from analysis_functional import plot_line, count_dist


def show_distr(array, normalize: bool, round_to: int):
    """
    По данному массиву сроит распределение его элементов
    Сначала нормализует значения максимальным элементом,
    затем считает кол-во элементов, принимающих соответсвующее значение,
    нормализует кол-вом элементов
    """
    if normalize:
        array = [round(el / np.max(array), round_to) for el in array]

    dist_norm = {}

    count_dist(array, dist_norm)

    plot_line(dist_norm, len(array))



# distr_angle = np.load('30_density_colored_distributions/all_areas.npy', allow_pickle=True).all()
all_angles_30 = np.load('30_density_distributions/all_angles.npy', allow_pickle=True)
all_angles_30_colored = np.load('30_density_colored_distributions/all_angles.npy', allow_pickle=True)
all_angles_50 = np.load('50_density_distributions/all_angles.npy', allow_pickle=True)
all_angles = np.concatenate([all_angles_30, all_angles_50, all_angles_30_colored])
print(len(all_angles), len(all_angles_30), len(all_angles_50))
# show_distr(all_angles, False, 2)
show_distr(all_angles, False, 0)
