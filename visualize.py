'''
Файл содержит описание функции, визуализирующей два массива одинаковых размеров ( a*a*a )
и закоментированного примера использования

1. Для использования в своём коде поместить visualize.py в одной директории c кодом

> dir/
    > your_code.py
    > visualize.py

2. Импортировать как
from visualize import visualize_3d_arrays 

3. После этого можно использовать функцию visualize_3d_arrays:

Функция принимает два трёхмерных numpy массива и их имена. ( см. пример )

Списки (list) при необходимости можно преобразовать как: numpy.array(your_list) или np.array(your_list)
Как в примере.

'''



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def visualize_3d_arrays(array1, array2, array1_name="Array 1", array2_name="Array 2"):
    """
    Визуализирует два 3D массива с послойным просмотром и фиксированной цветовой шкалой
    
    Параметры:
    array1: - первый 3D массив ( Трёхмерный )
    array2: - второй 3D массив
    array1_name: str - название первого массива
    array2_name: str - название второго массива
    """
    # Проверка размеров массивов
    if array1.shape != array2.shape:
        raise ValueError("Оба массива должны иметь одинаковые размеры")
    
    current_array = array1
    current_array_name = array1_name
    current_layer = 0
    
    # Определяем общие границы цветовой шкады
    vmin = min(array1.min(), array2.min())
    vmax = max(array1.max(), array2.max())
    
    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.25)
    
    # Первоначальное отображение
    im = ax.imshow(current_array[:, :, current_layer], 
                  cmap='viridis', 
                  vmin=vmin, 
                  vmax=vmax,
                  interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title(f"{current_array_name} - Layer {current_layer}")
    
    # Функции для обновления отображения
    def update_display():
        im.set_array(current_array[:, :, current_layer])
        ax.set_title(f"{current_array_name} - Layer {current_layer}")
        fig.canvas.draw_idle()
    
    # Обработчики событий
    def next_layer(event):
        nonlocal current_layer
        if current_layer < current_array.shape[2] - 1:
            current_layer += 1
            update_display()
    
    def prev_layer(event):
        nonlocal current_layer
        if current_layer > 0:
            current_layer -= 1
            update_display()
    
    def switch_array(event):
        nonlocal current_array, current_array_name
        if current_array is array1:
            current_array = array2
            current_array_name = array2_name
        else:
            current_array = array1
            current_array_name = array1_name
        update_display()
    
    # Создаем кнопки
    button_height = 0.07
    button_width = 0.2
    vertical_pos = 0.05
    
    ax_prev = plt.axes([0.15, vertical_pos, button_width, button_height])
    ax_next = plt.axes([0.40, vertical_pos, button_width, button_height])
    ax_switch = plt.axes([0.65, vertical_pos, button_width, button_height])
    
    btn_prev = Button(ax_prev, 'Предыдущий слой', color='lightblue')
    btn_next = Button(ax_next, 'Следующий слой', color='lightblue')
    btn_switch = Button(ax_switch, 'Оптимизация', color='lightgreen')
    
    # Увеличиваем шрифт кнопок
    for btn in [btn_prev, btn_next, btn_switch]:
        btn.label.set_fontsize(12)
        btn.label.set_fontweight('bold')
    
    # Привязываем обработчики
    btn_prev.on_clicked(prev_layer)
    btn_next.on_clicked(next_layer)
    btn_switch.on_clicked(switch_array)
    
    plt.show()

'''
array1 = [
    [[7, -3, 2], 
     [5, -8, 4], 
     [-1, 6, 9]],
    
    [[-2, 4, -5], 
     [3, 7, -6], 
     [8, -9, 1]],
    
    [[5, 0, -4], 
     [2, -7, 3], 
     [6, -2, 8]]
]

array2 = [
    [[-7, -2, -5], 
     [-3, -9, -1], 
     [-4, -6, -8]],
    
    [[-5, -1, -9], 
     [-2, -4, -7], 
     [-3, -8, -6]],
    
    [[-4, -3, -2], 
     [-1, -5, -9], 
     [-7, -6, -8]]
]

visualize_3d_arrays(np.array(haha1), np.array(haha1), "Поле 1", "Второе поле мяу")
'''
