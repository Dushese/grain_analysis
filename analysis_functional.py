import math
import cv2 as cv
from numba import njit, prange
import numpy as np
from shapely.geometry import Polygon, Point
from shapely import contains
from matplotlib import pyplot as plt

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)#np.clip(vector / np.linalg.norm(vector), [10**(-10), 10**(-10)], [10, 10])


def drow_vertices(contour, img):
    img = np.copy(img)
    starting_color = [0, 0, 100]
    point_size = 3
    for cnt_arr in contour:
        cv.drawContours(img, np.array([cnt_arr]), -1, (255, 255, 0), 1, cv.LINE_AA)
        for cn in cnt_arr:
            img[cn[0][1] - point_size:cn[0][1] + point_size, cn[0][0] - point_size:cn[0][0] + point_size] = starting_color
        starting_color = ((starting_color[0] + 50) % 255, (starting_color[1] + 5) % 255, (starting_color[2] + 30) % 255)

    return img

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


def show_three_points(points, index, img, trigger='какая-то ошибка'):
    print(trigger)
    img_test = np.copy(img)
    one = points[index]
    two = points[(index + 1) % len(points)]
    three = points[(index + 2) % len(points)]
    size = 3
    img_test[one[1] - size:one[1] + size, one[0] - size:one[0] + size] = np.array([255])
    img_test[two[1] - size:two[1] + size, two[0] - size:two[0] + size] = np.array([255, 0, 0])
    img_test[three[1] - size:three[1] + size, three[0] - size:three[0] + size] = np.array([255])
    cv.drawContours(img_test, [points], -1, (255, 0, 0), 1, cv.LINE_AA)
    plt.imshow(img_test)
    plt.show()

def show_grain_contour(points, img, trigger='какая-то ошибка'):
    print(trigger)
    img_test = np.copy(img)
    points = np.array([[el] for el in points], dtype=np.intc)
    img_test = drow_vertices([points], img_test)
    plt.imshow(img_test)
    plt.show()


def approximation(contour, eps=0.0075):
    approxed_arr = []
    # distr = {}
    # prev = 0
    # intervals = map(int, np.linspace(5000, 120000, 12))
    # for point in intervals:
    #     distr[f'{prev}_{point}'] = 0
    #     prev = point
    for cnt_arr in contour:
        if len(cnt_arr) < 3:
            continue
        poly = Polygon([point[0] for point in cnt_arr])
        area = poly.area

        prev = 0
        # for point in intervals:
        #     if prev <= area and area <= point:
        #         distr[f'{prev}_{point}'] += 1
        #     prev = point
        cnt_arr = np.array(cnt_arr)
        if area >= 20000:
            epsilon = 0.007 * cv.arcLength(cnt_arr, True)
        elif area >= 15000:
            epsilon = 0.007 * cv.arcLength(cnt_arr, True)
        elif area >= 10000:
            epsilon = 0.007 * cv.arcLength(cnt_arr, True)
        elif area >= 5000:
            epsilon = 0.007 * cv.arcLength(cnt_arr, True)
        else:
            epsilon = 0.007 * cv.arcLength(cnt_arr, True)
        approx = cv.approxPolyDP(cnt_arr, epsilon, True)
        approxed_arr.append(approx)
    # print(distr)
    return approxed_arr


def merge_points(prev_point, next_point):
    # prev_point = (prev_point + next_point) / 2
    prev_point = prev_point # TODO attention
    prev_point = [[np.round(prev_point[0][0]), np.round(prev_point[0][1])]]
    return prev_point


def grain_cleaner(cnt, area, img=None, a=175, b=3):
    # считаем углы
    points = np.array([point[0] for point in cnt])
    cnt_new = []
    prev_point = cnt[0]
    ang_amount = len(cnt)
    index = -1
    for i in range(len(cnt)): # Если две вершины очень близко - берем среднее этих вершин
        next_point = cnt[(i + 1) % len(cnt)]
        if np.linalg.norm(prev_point[0] - next_point[0]) < 15 and ang_amount > 3 and area > 20000:
            prev_point = merge_points(prev_point, next_point)
            ang_amount -= 1
            index = i + 1
        # elif np.linalg.norm(prev_point[0] - next_point[0]) < 1 and ang_amount > 3 and area > 15000:
        #     prev_point = merge_points(prev_point, next_point)
        #     ang_amount -= 1
        #     index = i + 1
        # elif np.linalg.norm(prev_point[0] - next_point[0]) < 1 and ang_amount > 3 and area > 10000:
        #     prev_point = merge_points(prev_point, next_point)
        #     ang_amount -= 1
        #     index = i + 1
        # elif np.linalg.norm(prev_point[0] - next_point[0]) < 1 and ang_amount > 3 and area > 5000:
        #     prev_point = merge_points(prev_point, next_point)
        #     ang_amount -= 1
        #     index = i + 1
        elif np.linalg.norm(prev_point[0] - next_point[0]) < 5 and ang_amount > 3:
            prev_point = merge_points(prev_point, next_point)
            ang_amount -= 1
            index = i + 1
        else:
            first_vec = np.array(prev_point - next_point)[0]
            # print((i + 2) % len(cnt), index)
            if (i + 2) % len(cnt) == index:
                second_vec = np.array(cnt[(i + 3) % len(cnt)] - next_point)[0]
            else:
                second_vec = np.array(cnt[(i + 2) % len(cnt)] - next_point)[0]

            angle = math.floor(angle_between(first_vec, second_vec))
            if (angle < 180 and area >= 20000) or ang_amount == 3:
                cnt_new.append(prev_point)
                prev_point = next_point
                # if isinstance(img, np.ndarray):
                #     show_three_points(points, i, img, f'{area, angle}')
            elif (angle < 180 and area >= 15000) or ang_amount == 3:
                cnt_new.append(prev_point)
                prev_point = next_point
                # if isinstance(img, np.ndarray):
                #     show_three_points(points, i, img, f'{area, angle}')
            elif (angle < 180 and area >= 10000) or ang_amount == 3:
                cnt_new.append(prev_point)
                prev_point = next_point
                # if isinstance(img, np.ndarray):
                #     show_three_points(points, i, img, f'{area, angle}')
            elif (angle < 155 and area >= 5000) or ang_amount == 3:
                cnt_new.append(prev_point)
                prev_point = next_point
                # if isinstance(img, np.ndarray):
                #     show_three_points(points, i, img, f'{area, angle}')
            elif (angle < 150 and area < 5000) or ang_amount == 3:
                cnt_new.append(prev_point)
                prev_point = next_point

            elif i == len(cnt) - 1:
                # if isinstance(img, np.ndarray) and area == 1512:
                #     show_three_points(points, i, img, f'{area, angle} [pltvfd')
                cnt_new.pop(0)
                cnt_new.append(prev_point)
            else:
                ang_amount -= 1
                index = i + 1
    return cnt_new

def count_characteristics(x_shape, y_shape, approxed_arr, amount_images, img = None, a=3, b=3):
    """
    Считает характеристики многоугольников по координатам их вершин

    :param approxed_arr: массив координат вершин для каждого многоугольника вида [ [ [[x1, y1]], [[x2, y2]], ...], ...], где x1, y1 коорднаты вершины
    :return: массив всех встречавшихся углов на изображении
    """

    all_area = 0
    areas = []
    angles = []
    perimeters = []
    new_approxed_arr = []

    distr = {}
    prev = 0
    intervals = list(map(int, np.linspace(0, 120000, 10)))
    for point in intervals:
        distr[f'{prev}_{point}'] = 0
        prev = point


    cracked_grains = {}
    cracked_grains['меньше, чем 3 вершины'] = 0
    cracked_grains['большое колво точек'] = 0
    cracked_grains['у стенок'] = 0
    cracked_grains['маленькая площадь'] = 0
    cracked_grains['Не выпуклый'] = 0
    for cnt in approxed_arr:
        if len(cnt) < 3:
            # if isinstance(img, np.ndarray):
            #     show_grain_contour(points, img, area)

            cracked_grains['меньше, чем 3 вершины'] += 1
            continue

        points = np.array([point[0] for point in cnt])
        poly = Polygon(points)
        area = poly.area
        prev = 0
        for point in intervals:
            if prev <= area and area <= point:
                distr[f'{prev}_{point}'] += 1
            prev = point
        # if area > 110000 and isinstance(img, np.ndarray):
        #     show_grain_contour(points, img,area )

        if area < 30:  # TODO не учитываю случайно выделенные области - очень маленькие
            cracked_grains['маленькая площадь'] += 1
            continue

        flag = 0
        for i in range(len(points)):
            padding = 2
            if ((points[i][0] >= 0 and points[i][0] <= 0 + padding) or (
                    points[i][0] >= x_shape - padding and points[i][0] <= x_shape)) \
                    or ((points[i][1] >= 0 and points[i][1] <= 0 + padding) or (
                    points[i][1] >= y_shape - padding and points[i][1] <= y_shape)):
                cracked_grains['у стенок'] += 1
                flag = 1
                break
        if flag == 1:
            continue

        new_cnt = grain_cleaner(cnt, area, img, a=a, b=b)
        points = np.array([point[0] for point in new_cnt])
        cnt = np.array(new_cnt, dtype=np.intc)
        # if len(cnt) < 3:
        #     if isinstance(img, np.ndarray):
        #         show_grain_contour(points, img, area)
        #
        #     cracked_grains['меньше, чем 3 вершины'] += 1
        #     continue

        # if area < 30:  # TODO не учитываю случайно выделенные области - очень маленькие
        #     cracked_grains['маленькая площадь'] += 1
        #     continue

        # else:
            # if isinstance(img, np.ndarray):
            #     show_grain_contour(points, img, f'area = {area}')

        # if np.shape(cnt)[0] > 15:  #TODO не учитываю многоугольники с большим количеством точек, так как эти точки не являются вершинами многоугольника
        #     # if isinstance(img, np.ndarray):
        #     #     show_grain_contour(points, img, f'большое колво точек {area}')
        #
        #     cracked_grains['большое колво точек'] += 1
        #     continue


        # if len(cnt) < 3:
        #
        #     cracked_grains['меньше, чем 3 вершины'] += 1
        #     continue

        flag = 0
        for i in range(len(points)):

            first_vec_ = np.array(points[(i + 1) % len(points)] - points[i])
            second_vec_ = np.array(points[(i + 2) % len(points)] - points[i])
            first_vec = np.array(points[i] - points[(i + 1) % len(points)])
            second_vec = np.array(points[(i + 2) % len(points)] - points[(i + 1) % len(points)])
            angle = math.floor(angle_between(first_vec, second_vec))
            if np.cross(first_vec_, second_vec_) < 0:
                flag = 1
                cracked_grains['Не выпуклый'] += 1
                # print(first_vec_, second_vec_)
                # points = [point[0] for point in cnt]
                # points = np.array(points)
                # show_three_points(points, i, img, f'Не выпуклый {angle} {np.cross(first_vec_, second_vec_)}')
                # break
        if flag == 0:
            for i in range(len(points)):
                first_vec = np.array(points[i] - points[(i + 1) % len(points)])
                second_vec = np.array(points[(i + 2) % len(points)] - points[(i + 1) % len(points)])
                angle = math.floor(angle_between(first_vec, second_vec))
                angles.append(angle)
                # if angle in (118, 119, 120, 121, 122) and isinstance(img, np.ndarray):
                #     show_three_points(points, i, img, f'{angle, area}')
            area = polygonArea(points)
            areas.append(area)
            all_area += area

            perimeters.append(count_perimetr(points))

            new_approxed_arr.append(cnt)

    print(cracked_grains)
    print(f'Средняя площадь зерна в пикселях = {all_area / (len(new_approxed_arr))}')
    print(
        f' площадь всех зерен = {all_area} \n '
        f'площадь изображения = {x_shape * y_shape} \n '
        f'плотность зерен = {all_area / (x_shape * y_shape * amount_images)} \n '
        f'максимальная площадь зерна = {np.max(areas)} \n '
        f'минимальная площадь зерна = {np.min(areas)} \n ' 
        f'кол-во зерен = {len(new_approxed_arr)}')
    return angles, new_approxed_arr, all_area / (x_shape * y_shape), all_area / len(new_approxed_arr), areas, perimeters, distr


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


def count_area(shape_x, shape_y, approxed_arr, img):
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
        if area < 5: # удаляю случайно выделенные области - очень маленькие
            not_real_grain += 1
            continue
        # elif area == 97766:
        #     show_three_points(img, cnt[0][0], cnt[1][0], cnt[2][0])
        #     imgplot = plt.imshow(img)
        #     plt.show()

        new_approxed_arr.append(cnt)
        areas.append(round(area)) # нужно добиться какого-то нужного округления
        all_area += area
    # print(areas)
    print(f'Средняя площадь зерна в пикселях = {all_area / (len(approxed_arr) - not_real_grain)}')
    print(
        f' площадь всех зерен = {all_area} \n '
        f'площадь изображения = {shape_x * shape_y} \n '
        f'плотность зерен = {all_area / (shape_x * shape_y)} \n '
        f'максимальная площадь зерна = {np.max(areas)} \n '
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
    for interval in np.arange(0, 10, 0.1):
        if interval not in distribution:
            distribution[interval] = 0


def polygonArea(points):
    # Initialize area
    area = 0.0

    # Calculate value of shoelace formula
    j = len(points) - 1
    for i in range(len(points)):
        area += (points[i, 0] + points[j, 0]) * (points[j, 1] - points[i, 1])
        j = i  # j is previous vertex to i

    # Return absolute value
    return abs(area / 2.0)

def count_perimetr(points):
    points = np.array(points)
    perimetr = 0
    j = len(points) - 1
    for i in range(len(points)):
        perimetr += np.linalg.norm(points[i] - points[j])
        j = i  # j is previous vertex to i
    return perimetr



def plot_hist(distribution: dict, objects_amount: int, name_distribution: str, folder_name: str):
    '''
    Нормирует распределение, строит график

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
    np.save(f'{folder_name}/{name_distribution}.npy', distribution)
    fig, ax = plt.subplots()
    ax.bar(index, values)
    ax.set_ylabel('Доля углов')
    ax.set_xlabel('значение угла, градусы')
    plt.show()

def plot_line(distribution_arr, target, x_name, y_name):
    '''
    Нормирует распределение, строит график, сохраняет распределение

    :param distribution: распредедение
    :param objects_amount:  кол-во объектов чтоды отнормировать
    :param name_distribution: название величины
    :density_amount: плотность зерен на изображениях, чтобы файл распределений сохранить в нужную папку
    '''
    # print(name_distribution, distribution)
    fig, ax = plt.subplots()
    ax.plot(target[:, 0], target[:, 1], label='Целевое распределение')
    ax.plot(distribution_arr[:, 0], distribution_arr[:, 1], label='Распределение, посчитанное на смоделированных микроснимках')
    # ax.legend()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, ncol=1)
    ax.set_ylabel(y_name)
    ax.set_xlabel(x_name)
    plt.show()
    # plt.savefig(f'./fitting_plots/{str(x_name).replace(".", ",")}_{y_name}')

