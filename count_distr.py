import numpy as np
from analysis_functional import plot_line, count_dist
# distr_angle = np.load('30_density_colored_distributions/all_areas.npy', allow_pickle=True).all()
distr_angle = np.load('30_density_colored_distributions/all_areas.npy', allow_pickle=True)


for i in range(1, 100, 10):
    print(i)
    # normalized_areas = [round(el * i) for el in distr_angle / np.max(distr_angle)]
    normalized_areas = [el // i for el in distr_angle]
    dist = {}

    count_dist(normalized_areas, dist)
    print(sorted(dist.items(), key=lambda x: x[0]))
    plot_line(dist, np.shape(normalized_areas)[0], 'areas_true_dist', '30_density_colored_distributions')


sum = 0
for k, i in dist.items():
    sum += i
print(sum)