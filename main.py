import random
import sys
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QPainter, QBrush, QPen, QImage
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QMessageBox, \
    QGraphicsTextItem
import random
import numpy as np

# задаем матрицу расстояний между городами
distances = np.array([
    [0, 1, 2, 3, 9],
    [1, 0, 4, 2, 6],
    [2, 4, 0, 5, 7],
    [3, 2, 5, 0, 8],
    [9, 6, 7, 8, 0]])


# функция оценки приспособленности
def fitness(genome):
    total_distance = 0
    for i in range(len(genome) - 1):
        current_city = genome[i]
        next_city = genome[i + 1]
        total_distance += distances[current_city][next_city]
    # добавляем расстояние от последнего города до города с номером 0
    total_distance += distances[genome[-1]][0]
    # добавляем расстояние от города с номером 0 до первого города в маршруте
    total_distance += distances[0][genome[0]]
    return 1 / total_distance


def tournament_selection(population, fitnesses, tournament_size=3):
    idx = random.sample(range(len(population)), tournament_size)
    tournament = [population[i] for i in idx]
    tournament_fitnesses = [fitnesses[i] for i in idx]
    winner_idx = tournament_fitnesses.index(max(tournament_fitnesses))
    return tournament[winner_idx]

def crossover(parent1, parent2):
    child = [-1] * len(parent1)
    # выбираем случайный участок генов от первого родителя
    start = random.randint(0, len(parent1)-1)
    end = random.randint(start, len(parent1)-1)
    child[start:end+1] = parent1[start:end+1]
    # заполняем оставшиеся гены из второго родителя
    for i in range(len(parent2)):
        if parent2[i] not in child:
            for j in range(len(child)):
                if child[j] == -1:
                    child[j] = parent2[i]
                    break
    return child

def mutation(genome, mutation_rate=0.01):
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(genome)-1)
            genome[i], genome[j] = genome[j], genome[i]
    return genome

def genetic_algorithm(num_cities, population_size, num_generations):
    # создаем начальную популяцию геномов
    population = []
    for i in range(population_size):
        genome = list(range(1, num_cities))
        random.shuffle(genome)
        population.append([0] + genome)
    for generatгion in range(num_generations):
        # оцениваем каждый геном в популяции
        fitnesses = [fitness(genome) for genome in population]

        # выбираем лучших геномов для скрещивания
        selected = [tournament_selection(population, fitnesses) for i in range(population_size)]
        # скрещиваем и мутируем выбранных геномов
        offspring = []
        for i in range(0, population_size, 2):
            child1 = crossover(selected[i], selected[i + 1])
            child2 = crossover(selected[i + 1], selected[i])
            child1 = mutation(child1)
            child2 = mutation(child2)
            offspring.append(child1)
            offspring.append(child2)

        # заменяем старых геномов в популяции на новых
        population = offspring

    # оцениваем каждый геном в конечной популяции
    fitnesses = [fitness(genome) for genome in population]
    best_idx = np.argmax(fitnesses)
    best_genome = population[best_idx]
    best_fitness = fitnesses[best_idx]

    # выводим результаты
    print("Лучший геном: ", best_genome)
    print("Приспособленность: ", best_fitness)
    # вычисляем стоимость маршрута
    total_distance = 0
    for i in range(len(best_genome) - 1):
        current_city = best_genome[i]
        next_city = best_genome[i + 1]
        total_distance += distances[current_city][next_city]
    total_distance += distances[best_genome[-1]][0]
    print("Стоимость маршрута: ", total_distance)
    return best_genome

#РЕЩЕНИЕ ЗАДАЧИ УПАКОВКИ
#РЕЩЕНИЕ ЗАДАЧИ УПАКОВКИ
class Box:
    def __init__(self, length, width, weight):
        self.length = length
        self.width = width
        self.weight = weight
        self.items = []


class Item:
    def __init__(self, length, width, weight, color):
        self.length = length
        self.width = width
        self.weight = weight
        self.color = color


class ItemPosition:
    def __init__(self, item, x, y):
        self.item = item
        self.x = x
        self.y = y

    def move(self, delta_x, delta_y):
        return ItemPosition(self.item, self.x + delta_x, self.y + delta_y)
    def intersects(self, other):
        # Проверяем, пересекается ли данный предмет с другим предметом
        rect1 = QRectF(self.x, self.y, self.item.length, self.item.width)
        rect2 = QRectF(other.x, other.y, other.item.length, other.item.width)
        return rect1.intersects(rect2)

class TSCargo:
    def __init__(self, box):
        self.box = box
        self.solution = []
        self.used_positions = []
        self.current_weight = 0  # Текущий вес в упаковке
        self.remaining_items = []  # Список оставшихся неупакованных грузов
        self.flag = False

    def pack_items(self, items):
        sorted_items = sorted(items, key=lambda item: item.length * item.width)
        for item in sorted_items:
            if self.current_weight + item.weight > self.box.weight:
                self.remaining_items.append(item)
                self.flag = True
                continue
            best_fit_pos = None
            best_fit_score = float('inf')  # Начальное значение наилучшего показателя помещения

            for x in range(self.box.length - item.length + 1):
                for y in range(self.box.width - item.width + 1):
                    pos = ItemPosition(item, x, y)

                    # Проверяем, пересекается ли данная позиция с уже погруженными грузами
                    if any(pos.intersects(item_pos) for item_pos in self.used_positions):
                        continue

                    # Проверяем, находится ли позиция за уже погруженными грузами
                    if any(pos.x + pos.item.length > item_pos.x and pos.y + pos.item.width > item_pos.y
                           for item_pos in self.used_positions):
                        continue

                    # Рассчитываем показатель помещения для данной позиции
                    fit_score = self.calculate_fit_score(pos)

                    if fit_score < best_fit_score:
                        best_fit_pos = pos
                        best_fit_score = fit_score

            if best_fit_pos is not None:
                self.solution.append(best_fit_pos)
                self.used_positions.append(best_fit_pos)
                self.current_weight += item.weight


    def calculate_fit_score(self, pos):
        # Рассчитываем показатель помещения для данной позиции.
        # Чем меньше показатель, тем лучше позиция подходит для помещения груза.
        # В данном примере, мы используем сумму длины и ширины свободной области вокруг позиции.
        free_space_length = self.box.length - pos.x - pos.item.length
        free_space_width = self.box.width - pos.y - pos.item.width
        return free_space_length + free_space_width

    def get_remaining_items(self):
        return self.remaining_items

    def get_flag(self):
        return self.flag

# ПЕРВЫЙ ПОДХОДЯЩИЙ
class Solver:
    def __init__(self, boxes, cities):
        self.boxes = boxes
        self.solutions = [TSCargo(box) for box in self.boxes]
        self.current_box_index = 0
        for city in reversed(cities):
            if self.current_box_index >= len(self.boxes):
                break
            tscargo = self.solutions[self.current_box_index]
            tscargo.pack_items(city)
            if tscargo.get_flag():
                self.current_box_index += 1
            remaining_items = tscargo.get_remaining_items()
            if remaining_items is not None and self.current_box_index <= len(self.boxes)-1:
                tscargo = self.solutions[self.current_box_index]
                tscargo.pack_items(remaining_items)

    def get_total_boxes(self):
        return len(self.boxes)


class BoxGraphicsItem(QGraphicsRectItem):
    def __init__(self, box, items):
        super().__init__(0, 0, box.length, box.width)
        self.setBrush(QBrush(QColor(200, 200, 200)))
        self.setPen(QPen(Qt.black))

        if not items:
            self.setBrush(QBrush(QColor(100, 100, 100)))


class ItemGraphicsItem(QGraphicsRectItem):
    def __init__(self, item_pos, city_index):
        super().__init__(item_pos.x, item_pos.y, item_pos.item.length, item_pos.item.width)
        self.setBrush(QBrush(item_pos.item.color))

        self.text_item = QGraphicsTextItem(str(city_index))
        self.text_item.setParentItem(self)
        self.text_item.setPos(item_pos.x + 5, item_pos.y + 5)  # Позиция текста на грузе


class Window(QGraphicsView):
    def __init__(self, solver):
        super().__init__()

        self.solver = solver

        scene = QGraphicsScene(self)

        total_height = 0
        for i, tscargo in enumerate(solver.solutions):
            box_item = BoxGraphicsItem(tscargo.box, tscargo.solution)
            box_item.setPos(0, total_height)
            scene.addItem(box_item)

            for j, item_pos in enumerate(tscargo.solution):
                if item_pos.item:
                    item_item = ItemGraphicsItem(item_pos, j + 1)  # Передача номера города
                    item_item.setPos(0, total_height)
                    scene.addItem(item_item)

            total_height += tscargo.box.width + 10
        scene.setSceneRect(0, 0, solver.boxes[0].length, total_height)
        self.setScene(scene)

    def save_image(self, filename):
        image = QImage(self.scene().sceneRect().size().toSize(), QImage.Format_ARGB32)
        image.fill(Qt.transparent)

        painter = QPainter(image)
        self.scene().render(painter)
        painter.end()

        image.save(filename)



def main():
    # Создание ТС
    boxes = [
        Box(1400, 300, 50),
        Box(1400, 300, 100)
    ]

    best_route = genetic_algorithm(num_cities=len(distances[0]), population_size=1000, num_generations=1000)

    cargo_sizes = {
        1: [Item(200, 150, 10, QColor(0, 0, 255)), Item(150, 150, 10, QColor(0, 0, 255)),
            Item(50, 50, 10, QColor(0, 0, 255))],  # синий
        2: [Item(200, 200, 10, QColor(255, 0, 255)), Item(300, 100, 10, QColor(255, 0, 255))],  # розовый
        3: [Item(150, 150, 10, QColor(0, 255, 0)), Item(200, 200, 10, QColor(0, 255, 0))],  # зеленый
        4: [Item(100, 100, 10, QColor(0, 255, 255)), Item(100, 100, 10, QColor(0, 255, 255)),
            Item(300, 300, 30, QColor(0, 255, 255)), Item(50, 50, 10, QColor(0, 255, 255))]  # бирюзовый
    }
    cities = [cargo_sizes[city_index] for city_index in best_route if city_index != 0]

    # Решение задачи упаковки
    solver = Solver(boxes, cities)

    # Создание графического интерфейса и отображение результата
    app = QApplication(sys.argv)
    window = Window(solver)
    window.show()
    window.save_image("visualization.png")
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()



