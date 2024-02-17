from numba import njit, prange
import numpy as np
from shapely.geometry import Polygon
from matplotlib import pyplot as plt

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)#np.clip(vector / np.linalg.norm(vector), [10**(-10), 10**(-10)], [10, 10])


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

             angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            angle_between((1, 0, 0), (1, 0, 0))
            0.0
            angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    return angle


def drow_three_points(img, one, two, three):
    size = 5
    img[one[1] - size:one[1] + size, one[0] - size:one[0] + size] = np.array([255])
    img[two[1] - size:two[1] + size, two[0] - size:two[0] + size] = np.array([255, 0, 0])
    img[three[1] - size:three[1] + size, three[0] - size:three[0] + size] = np.array([255])
    return img


def count_angles(x_shape, y_shape, approxed_arr):
    """
    Считает углы всех многоугольников по координатам их вершин

    :param approxed_arr: массив координат вершин для каждого многоугольника вида [ [ [[x1, y1]], [[x2, y2]], ...], ...], где x1, y1 коорднаты вершины
    :return: массив всех встречавшихся углов на изображении
    """
    angles = []
    counter = 0
    for cnt in approxed_arr:
        cnt_new = []
        prev_point = cnt[0]
        ang_amount = len(cnt)
        for i in range(len(cnt)): # Если две вершины очень близко - берем среднее этих вершин
            next_point = cnt[(i + 1) % len(cnt)]
            amount_merged = 0

            if np.linalg.norm(prev_point[0] - next_point[0]) < 3 and ang_amount > 3:
                amount_merged += 1
                prev_point = (prev_point + next_point) / 2
                prev_point = [[np.round(prev_point[0][0]), np.round(prev_point[0][1])]]
                ang_amount -= amount_merged
            else:
                cnt_new.append(prev_point)
                prev_point = cnt[(i + 1) % len(cnt)]
        cnt = np.array(cnt_new, dtype=np.intc)
        if np.shape(cnt)[0] > 10:  # не учитываю многоугольники с большим количеством точек, так как эти точки не являются вершинами многоугольника
            continue
        # cv.drawContours(img, [cnt], -1, (255, 0, 0), 1, cv.LINE_AA)

        points = [point[0] for point in cnt]
        points = np.array(points)
        # print(points)
        flag = 0
        for i in range(len(points)):
            # print(points[i])
            if ((points[i][0] >= 0 and points[i][0] <= 0 + 5) or (points[i][0] >= x_shape - 5 and points[i][0] <= x_shape)) \
                    or ((points[i][1] >= 0 and points[i][1] <= 0 + 5) or (points[i][1] >= y_shape - 5 and points[i][1] <= y_shape)):
                # print(points[i], np.shape(img)[0], np.shape(img)[1])
                flag = 1
                # img_test = np.copy(img)
                # one = points[i]
                # two = points[(i + 1) % len(points)]
                # three = points[(i + 2) % len(points)]
                # img_test = drow_three_points(img_test, one, two, three)
                # fig, ax = plt.subplots(figsize=(12, 12))
                # plt.imshow(img_test)
                # plt.show()
                break
        if flag == 0:
            for i in range(len(points)):
                # img_test = np.copy(img)
                first_vec = np.array(points[i] - points[(i + 1) % len(points)])
                second_vec = np.array(points[(i + 2) % len(points)] - points[(i + 1) % len(points)])
                angle = round(angle_between(first_vec, second_vec))

                angles.append(angle)
                # print(angle)
                # one = points[i]
                # two = points[(i + 1) % len(points)]
                # three = points[(i + 2) % len(points)]
                # img_test = drow_three_points(img_test, one, two, three)
                # fig, ax = plt.subplots(figsize=(12, 12))
                # plt.imshow(img_test)
                # plt.show()

                # if angle == 135:
                #     counter += 1
                #     img = cv.imread(img_name)
                #     one = points[i]
                #     two = points[(i + 1) % len(points)]
                #     three = points[(i + 2) % len(points)]
                #     img = drow_three_points(img, one, two, three)
                #     cv.imwrite(f'angle_id_{counter}_135.png', img)
            # cv.imshow('135', img)
            # cv.waitKey()
            # cv.destroyAllWindows()

    # cv.imshow('cvcv', img)
    # cv.waitKey()
    # cv.destroyAllWindows()
    return angles


@njit()
def show_cluster(clusters_matrix, x_shape, y_shape, cluster):
    """
    С помощью k-means была получена матрица кластеров,
    которая каждому пискелю изображения сопоставляла номер кластера, кластеры получаются очень шумные.
    Идея фильтрануть каждый кластер по отдельности (вместе их не получится фильтрануть).
    Функция выделяет конкретный cluster
    :param clusters_matrix: матрица кластеров
    :param x_shape: размер изображения по х
    :param y_shape: размер изображения по у
    :param cluster: номер кластера - номером кластера маркируются пиксели, принадлежащие кластеру
    :return:
    """
    marker_matrix = np.zeros((x_shape, y_shape))
    cluster_image = np.zeros((x_shape, y_shape))
    amount_marked_pix = 0
    for i in prange(x_shape):
        for j in range(y_shape):
            if clusters_matrix[i, j] == cluster:
                amount_marked_pix += 1
                marker_matrix[i, j] = cluster
                cluster_image[i, j] = 255
    return amount_marked_pix, marker_matrix, cluster_image


def count_area(shape_x, shape_y, approxed_arr):
    """
    :param approxed_arr: координаты вершин многоугольников
    :return: новый массив координат, плотность зерен, средняя площадь зерна, массив площадей
    """
    all_area = 0
    not_real_grain = 0 # для подсчета ненастоящих зерен
    areas = []
    new_approxed_arr = []
    for cnt in approxed_arr:
        area = Polygon([point[0] for point in cnt]).area
        if area < 1: # удаляю случайно выделенные области - очень маленькие
            not_real_grain += 1
            continue
        new_approxed_arr.append(cnt)
        areas.append(round(area/100.0)) # нужно добиться какого нужного округления
        all_area += area
    # print(areas)
    print(f'Средняя площадь зерна в пикселях = {all_area / (len(approxed_arr) - not_real_grain)}')
    print(
        f' площадь всех зерен = {all_area} \n '
        f'площадь изображения = {shape_x * shape_y} \n '
        f'плотность зерен = {all_area / (shape_x * shape_y)} \n '
        f'максимальная площадь зерна = {np.max(areas)} \n ' # обратите внимание на areas, может быть сильно округлен
        f'кол-во зерен = {len(areas)}')
    return new_approxed_arr, all_area / (shape_x * shape_y), all_area / (len(approxed_arr) - not_real_grain), areas


def resize_img(img):
    scale_percent = 30  # calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100.0)
    height = int(img.shape[0] * scale_percent / 100.0)
    dsize = (width, height)
    return cv.resize(img, dsize)

def count_dist(arr: np.array, distribution):
    '''
    Считает распределение величины
    :param arr: массив величин
    :return: массив, где каждому значению arr сопоставлено количество встретившихся значений в arr
    '''
    # print(arr, np.shape(arr)[0])
    for ind in range(np.shape(arr)[0]):
        if arr[ind] in distribution:
            distribution[arr[ind]] += 1
        else:
            distribution[arr[ind]] = 1


def plot_hist(distribution: dict, objects_amount: int, name_distribution: str, density_amount: int):
    '''
    Нормирует распределение, строит график, сохраняет распределение

    :param distribution: распредедение
    :param objects_amount:  кол-во объектов чтоды отнормировать
    :param name_distribution: название величины
    :density_amount: плотность зерен на изображениях, чтобы файл распределений сохранить в нужную папку
    '''
    # print(name_distribution, distribution)
    index = sorted(distribution.keys())
    values = np.array([distribution[ang] for ang in index]) / objects_amount
    print('\n\n')
    for key in index:
        distribution[key] = distribution[key] / objects_amount
        if distribution[key] > 0.06:
            print(f'угол = {key}, кол-во углов = {distribution[key] / objects_amount}')
    np.save(f'{density_amount}_density_distributions/{name_distribution}.npy', distribution)
    fig, ax = plt.subplots()
    ax.bar(index, values)
    ax.set_ylabel('Доля углов')
    ax.set_xlabel('значение угла, градусы')
    plt.show()
