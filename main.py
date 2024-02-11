import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numba import njit, prange
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
import time

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


def count_angles(approxed_arr):
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
            if ((points[i][0] >= 0 and points[i][0] <= 0 + 5) or (points[i][0] >= np.shape(img)[0] - 5 and points[i][0] <= np.shape(img)[0])) \
                    or ((points[i][1] >= 0 and points[i][1] <= 0 + 5) or (points[i][1] >= np.shape(img)[1] - 5 and points[i][1] <= np.shape(img)[1])):
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
    marker_matrix = np.zeros((image_base.shape[0], image_base.shape[1]))
    cluster_image = np.zeros((image_base.shape[0], image_base.shape[1]))
    amount_marked_pix = 0
    for i in prange(x_shape):
        for j in range(y_shape):
            if clusters_matrix[i, j] == cluster:
                amount_marked_pix += 1
                marker_matrix[i, j] = cluster
                cluster_image[i, j] = 255
    return amount_marked_pix, marker_matrix, cluster_image


def count_area(approxed_arr):
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
    print(areas)
    print(f'Средняя площадь зерна в пикселях = {all_area / (len(approxed_arr) - not_real_grain)}')
    print(
        f' площадь всех зерен = {all_area} \n '
        f'площадь изображения = {np.shape(img)[0] * np.shape(img)[1]} \n '
        f'плотность зерен = {all_area / (np.shape(img)[0] * np.shape(img)[1])} \n '
        f'максимальная площадь зерна = {np.max(areas)} \n ' # обратите внимание на areas, может быть сильно округлен
        f'кол-во зерен = {len(areas)}')
    return new_approxed_arr, all_area / (np.shape(img)[0] * np.shape(img)[1]), all_area / (len(approxed_arr) - not_real_grain), areas


def resize_img(img):
    scale_percent = 30  # calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100.0)
    height = int(img.shape[0] * scale_percent / 100.0)
    dsize = (width, height)
    return cv.resize(img, dsize)


def name_files_50_density():
    folder_name = "C:/Users/Dushese/files_for_jupiter/3400/img-"
    for folder_1 in range(10, 80, 10):
        for folder_2 in range(51, 55):
            for img_ind_1 in [4, 5]:
                if img_ind_1 == 4:
                    indexes = [x for x in range(2, 10)]
                else:
                    indexes = [x for x in range(0, 9)]
                for img_ind_2 in indexes:
                    yield folder_name + str(folder_1) + '-' + str(folder_2) + '-3400-0.08-0.03/z' + str(
                        img_ind_1) + '.' + str(img_ind_2) + '.png'

def count_dist(arr: np.array, distribution):
    '''
    Считает распределение величины
    :param arr: массив величин
    :return: массив, где каждому значению arr сопоставлено количетсво встретившихся значений в arr
    '''
    print(arr)
    for ind in range(np.shape(arr)[0]):
        if arr[ind] in distribution:
            distribution[arr[ind]] += 1
        else:
            distribution[arr[ind]] = 1


def plot_hist(distribution: dict, name_distribution: str):
    # print(name_distribution, distribution)
    index = sorted(distribution.keys())
    values = np.array([distribution[ang] for ang in index]) / all_angle_amount
    print('\n\n')
    for key in index:
        distribution[key] = distribution[key] / all_angle_amount
        if distribution[key] > 0.06:
            print(f'угол = {key}, кол-во углов = {distribution[key] / all_angle_amount}')
    np.save(f'{name_distribution}.npy', distribution)
    fig, ax = plt.subplots()
    ax.bar(index, values)
    ax.set_ylabel('Доля углов')
    ax.set_xlabel('значение угла, градусы')
    plt.show()

t = time.time()
# folder_name = "C:/Users/Dushese/files_for_jupiter/grains_density_30/img-"
density_sum = 0
average_grain_area_sum = 0
all_grains_amount = 0
all_angle_amount = 0
amount_images = 0
angle_dist = {}

area_dist = {}

# цикл по изображениям
for img_name in name_files_50_density():
    amount_images += 1 # подсчитать кол-во снимков
    print(img_name)
    image_base = cv.imread(img_name)
    img = image_base.reshape((image_base.shape[0] * image_base.shape[1], 3))
    common_params = {
        "init": 'k-means++',
        "n_init": "auto",
        "random_state": 4
    }
    y_pred = KMeans(n_clusters=110, **common_params).fit_predict(img)#cv.medianBlur(image_base,5))

    clusters_matrix = y_pred.reshape(image_base.shape[0], image_base.shape[1])


    result_markers = np.zeros((image_base.shape[0], image_base.shape[1]))
    result_image = np.zeros((image_base.shape[0], image_base.shape[1]))
    kernel_erode = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((3, 3), np.uint8)
    flag = 0
    for x in range(clstrs):
      amount_markered_in_pix, marker_matrix, cluster_image = show_cluster(clusters_matrix, image_base.shape[0], image_base.shape[1], x)
      if amount_markered_in_pix > image_base.shape[0] * image_base.shape[1] / 4: # если кол-во точек, принадлежащих одному кластеру равно четверти всей картинке и больше, то это фон, он нам не нужен
        continue
      marker_matrix = cv.erode(marker_matrix, kernel_erode, iterations=2)
      result_markers += marker_matrix

      cluster_image = cv.erode(cluster_image, kernel_erode, iterations=2)
      result_image += cluster_image
    # cv.imshow('cvcv', result_image)
    # cv.waitKey()
    # cv.destroyAllWindows()

    # Подготовка к watershed

    img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    ret, thresh = cv.threshold(img, 69, 255, cv.THRESH_BINARY_INV)
    # opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 1)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv.dilate(thresh, kernel, iterations=3)
    sure_fg = np.uint8(result_image)
    unknown = cv.subtract(sure_bg, sure_fg)

    # watershed
    img_real = cv.imread(img_name)
    result_markers = np.array(result_markers, dtype=np.int32)
    result_markers += 1
    result_markers[unknown > 50] = 0
    markers = cv.watershed(img_real, result_markers)

    img_real[markers == -1] = [255, 0, 0]


    # approximation

    img = cv.imread(img_name)
    img[::] = [0, 0, 0]
    img[markers == -1] = [255, 0, 0]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    # fig, ax = plt.subplots(figsize=(12, 12))
    # plt.imshow(img_test)
    # plt.show()
    img[:] = 0
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    new_cntrs = [contours[ind] for h, ind in zip(hierarchy[0], range(len(hierarchy[0]))) if h[2] == -1]
    if len(new_cntrs) < 5:
        amount_images -= 1
        print(f'кол-во найденных зерен {len(new_cntrs)} => не учитывается в статистике')
        continue
    # cv.drawContours(img, contours, -1, (255, 255, 0), 1, cv.LINE_AA)
    cv.drawContours(img, new_cntrs, -1, (255, 0, 0), 1, cv.LINE_AA)

    starting_color = [0, 0, 100]
    img_new = cv.imread(img_name)

    img = img_new.copy()
    eps = 0.01
    approxed_arr = []
    for cnt_arr in new_cntrs:
        epsilon = eps * cv.arcLength(cnt_arr, True)
        approx = cv.approxPolyDP(cnt_arr, epsilon, True)
        if len(approx) <= 2:
            continue
        approxed_arr.append(approx)
        cv.drawContours(img, [approx], -1, (255, 0, 0), 1, cv.LINE_AA)
        for cn in approx:
            img[cn[0][1] - 1:cn[0][1] + 1, cn[0][0]-1:cn[0][0]+1] = starting_color
        starting_color = ((starting_color[0] + 50) % 255, (starting_color[1] + 5) % 255, (starting_color[2] + 30) % 255)


    # for eps in np.linspace(0.0001, 0.02, 2):
    #     img = img_new.copy()
    #     # print('eps=', eps)
    #     approxed_arr = []
    #     for cnt_arr in new_cntrs:
    #         epsilon = eps * cv.arcLength(cnt_arr, True)
    #         approx = cv.approxPolyDP(cnt_arr, epsilon, True)
    #         approxed_arr.append(approx)
    #         cv.drawContours(img, [approx], -1, (255, 0, 0), 1, cv.LINE_AA)
    #         for cn in approx:
    #             img[cn[0][1] - 2:cn[0][1] + 2, cn[0][0]-2:cn[0][0]+2] = starting_color
    #         starting_color = ((starting_color[0] + 50) % 255, (starting_color[1] + 5) % 255, (starting_color[2] + 30) % 255)

    # Плотность
    approxed_arr, density, average_grain_area, areas = count_area(approxed_arr)

    density_sum += density
    average_grain_area_sum += average_grain_area
    # Распределение углов
    angles = count_angles(approxed_arr)
    all_angle_amount += len(angles)

    count_dist(angles, angle_dist)
    count_dist(areas, area_dist)


    # for ang in angles:
    #     if ang in angle_dist:
    #         angle_dist[ang] += 1
    #     else:
    #         angle_dist[ang] = 1

    if img_ind_2 == 2:
        name = str(folder_1) + '-' + '49' + '-3400-0.08-0.03-z' + str(
    img_ind_1) + '.' + str(img_ind_2)
        cv.imwrite(f'{name}_segmented.png', img)


print(f'Общие данные \n\tСредняя плотность зерен = {density_sum / amount_images}\n'
      f'\tСредний размер зерна = {average_grain_area_sum / amount_images}')
print(f'\tКол-во снимков = {amount_images}')
print(f'\n Epsilon = {eps}')
print(f'Общее время выполнения: {time.time() - t}')

plot_hist(angle_dist, 'angle_distribution')
plot_hist(area_dist, 'area_distribution')
# print(len(area_dist))

# index = sorted(angle_dist.keys())
# values = np.array([angle_dist[ang] for ang in index]) / all_angle_amount
#
# print('\n\n')
# for key in index:
#     angle_dist[key] = angle_dist[key] / all_angle_amount
#     if angle_dist[key] > 0.06:
#         print(f'угол = {key}, кол-во углов = {angle_dist[key] / all_angle_amount}')
# np.save('angle_distribution.npy', angle_dist)
# fig, ax = plt.subplots()
# ax.bar(index, values)
# ax.set_ylabel('Доля углов')
# ax.set_xlabel('значение угла, градусы')
# plt.show()
