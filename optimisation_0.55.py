import numpy as np
from scipy.optimize import dual_annealing  # отжиг
import matplotlib.pyplot as plt
from visualize import visualize_3d_arrays

# Физические константы
MU0 = 4e-7 * np.pi  # Магнитная постоянная [Тл·м/А]

# Параметры системы (в метрах)
GRID_SIZE = 20  # 20x20x20 точек (2 мм шаг)
GRID_LIMIT = 0.02  # Область -20..20 мм (0.02 м)
COIL_TURNS = 800  # Число витков
MAX_CURRENT = 20.0  # Максимальный ток ±5 А
COIL_RADIUS = 0.015  # Радиус катушки 10 мм

#Создает независимое поле с выскоими значениями в средних слоях, и низкими в первых и последних
'''def generate_field():
    """Генерация поля с выраженным горбом по Z"""
    x = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
    y = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
    z = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Основное поле 1 Тл + горб по Z
    Bz = 1.0 + 0.5 * np.exp(-(Z ** 2) / (2 * (0.01 ** 2)))  # Гауссов горб в центре
    Bx = 0.1 * np.sin(5 * X / GRID_LIMIT * np.pi)  # Лёгкие искажения по X
    By = 0.0 * Y  # Нет искажений по Y

    return np.stack((Bx, By, Bz), axis=-1)'''


def create_field_from_coils(coil_currents, grid):
    """
    Создаёт суммарное поле от всех катушек с заданными токами
    :param coil_currents: массив токов для каждой катушки (A)
    :param grid: 3D сетка точек (shape: NxNxNx3)
    :return: суммарное поле (shape: NxNxNx3)
    """
    B_total = np.zeros_like(grid)  # Инициализируем нулевое поле

    for i, coil in enumerate(coils_config):
        # Получаем параметры катушки
        pos = np.array(coil["pos"])
        axis = np.array(coil["axis"])
        current = coil_currents[i]

        # Добавляем поле текущей катушки
        B_total += bio_savart(
            coil_pos=pos,
            coil_normal=axis,
            current=current,
            points=grid.reshape(-1, 3),  # Преобразуем в 2D массив
            n_points=20
        ).reshape(grid.shape)  # Возвращаем исходную форму

    return B_total

def bio_savart(coil_pos, coil_normal, current, points, n_points=20):
    """Оптимизированный расчет по закону Био-Савара"""
    theta = np.linspace(0, 2*np.pi, n_points)
    dl = COIL_RADIUS * np.stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)], axis=1)
    coil_points = coil_pos + COIL_RADIUS * np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    
    B = np.zeros_like(points)
    for i in range(n_points):
        r = points - coil_points[i]
        r_norm = np.linalg.norm(r, axis=-1, keepdims=True)**3 + 1e-12
        B += np.cross(dl[i], r) / r_norm
    
    return MU0 * current * COIL_TURNS / (4*np.pi) * B

# Конфигурация 12 катушек (все вне куба 40×40×40 мм)
coils_config = [
    {"pos": [0.05, 0, 0], "axis": [1, 0, 0]},    # X+ (50 мм)
    {"pos": [-0.05, 0, 0], "axis": [-1, 0, 0]},  # X-
    {"pos": [0, 0.05, 0], "axis": [0, 1, 0]},    # Y+
    {"pos": [0, -0.05, 0], "axis": [0, -1, 0]},  # Y-
    {"pos": [0, 0.04, 0.05], "axis": [0, 0, 1]},    # Z+
    {"pos": [0, 0, -0.05], "axis": [0, 0, -1]},  # Z-
    {"pos": [0.04, 0.04, 0], "axis": [1, 1, 0]}, # XY
    {"pos": [-0.04, 0.04, 0], "axis": [-1, 1, 0]},
    {"pos": [0, 0.04, 0.04], "axis": [0, 1, 1]}, # YZ
    {"pos": [0, -0.04, 0.04], "axis": [0, -1, 1]},
    {"pos": [0.04, 0, 0.04], "axis": [1, 0, 1]}, # XZ
    {"pos": [-0.04, 0, 0.04], "axis": [-1, 0, 1]}
]

def optimize_currents(B_initial):
    # Создаем сетку 
    x = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
    y = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
    z = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = np.stack((X, Y, Z), axis=-1)
    
    flat_grid = grid.reshape(-1, 3)
    flat_B = B_initial.reshape(-1, 3)
    
    def objective(currents):
        B_total = flat_B.copy()
        for i, coil in enumerate(coils_config):
            B_total += bio_savart(
                np.array(coil["pos"]),
                np.array(coil["axis"]),
                currents[i],
                flat_grid,
                n_points=15
            )
        B_norm = np.linalg.norm(B_total, axis=1)
        return np.std(B_norm) / np.mean(B_norm) * 1e6  # Целевая функция (минимизация неоднородности)
    
    # Ограничения на токи: -MAX_CURRENT ≤ I ≤ MAX_CURRENT
    bounds = [(-MAX_CURRENT, MAX_CURRENT)] * len(coils_config)
    
    # Настройки отжига:
    # - maxiter: максимальное число итераций
    # - initial_temp: начальная "температура" (высокая = больше случайных шагов вначале)
    # - visit: параметр распределения шагов
    # - accept: параметр принятия решений
    res = dual_annealing(
        objective,
        bounds=bounds,
        maxiter=,          # Увеличим число итераций (отжиг требует больше вычислений)
        initial_temp=5000,      # Высокая начальная температура для широкого поиска
        visit=2.6,             # Параметр распределения шагов
        accept=-5.0,           # Параметр принятия ухудшений
        no_local_search=True    # Отключаем локальную оптимизацию (для скорости)
    )
    
    return res.x

#Снова создаём сетку для генерации случайных токов
x = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
y = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
z = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
grid = np.stack((X, Y, Z), axis=-1)  # shape: (20,20,20,3)
np.random.seed(42) # Seed можно менять и получать разные "случайные" токи
coil_currents = np.random.uniform(-10, 10, len(coils_config))
# Основной расчет
print("Генерация начального поля...")
B_initial = create_field_from_coils(coil_currents, grid)

print("Оптимизация токов...")
optimal_currents = optimize_currents(B_initial)

# Применяем оптимальные токи
print("Расчет компенсированного поля...")
B_compensated = B_initial.copy()

# Создаем сетку для конечного расчета
x = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
y = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
z = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_SIZE)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
grid = np.stack((X, Y, Z), axis=-1)

for i, coil in enumerate(coils_config):
    B_compensated += bio_savart(
        np.array(coil["pos"]),
        np.array(coil["axis"]),
        optimal_currents[i],
        grid,
        n_points=20
    ).reshape(B_initial.shape)

# Анализ результатов
def calculate_ppm(field):
    B_norm = np.linalg.norm(field, axis=-1)
    return (B_norm - np.mean(B_norm)) / np.mean(B_norm) * 1e6

initial_ppm = calculate_ppm(B_initial)
optimized_ppm = calculate_ppm(B_compensated)

print("\nРезультаты оптимизации:")
print(f"Исходное поле: max_ppm={np.max(np.abs(initial_ppm)):.1f}, std_ppm={np.std(initial_ppm):.1f}")
print(f"Компенсированное поле: max_ppm={np.max(np.abs(optimized_ppm)):.1f}, std_ppm={np.std(optimized_ppm):.1f}")

print("\nОптимальные токи в катушках [А]:")
for i, coil in enumerate(coils_config, 1):
    pos_mm = np.array(coil["pos"]) * 1000
    print(f"Катушка {i:2d} @ [{pos_mm[0]:.0f} {pos_mm[1]:.0f} {pos_mm[2]:.0f}] мм: {optimal_currents[i-1]:.3f}")

# Визуализация
print("\nВизуализация результатов...")
B_initial_norm = np.linalg.norm(B_initial, axis=-1) # преобразуем поля в модули, что бы визуализировать
B_compensated_norm = np.linalg.norm(B_compensated, axis=-1)

print("Передача данных в visualize_3d_arrays...")
visualize_3d_arrays(B_initial_norm, B_compensated_norm, "Оптимизированное поле","Исходное поле")

