from sympy import sympify, diff
from sympy.abc import x
from sympy.plotting import plot
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import csv


def save_calculation(func):
    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        with open(f'./cvs/{next(gen)}.csv', mode='w', encoding='utf-8') as w_file:
            file_writer = csv.writer(w_file, dialect='excel', delimiter=";", lineterminator="\r")
            for i in gen:
                file_writer.writerow(i)
    return wrapper


class OneDimensionalOptimization:
    saved_pictures = 0

    def __init__(self, expression: str, diapason: list):
        self.__f = sympify(expression)
        # производные
        self.__df = diff(self.__f, x)
        self.__ddf = diff(self.__f, x, x)
        self.__diapason = tuple([x] + diapason)

    def draw(self, name: str = 'main', directory='./img', save_mode=False):
        # получение точек графиков
        plots = plot((self.__f, self.__diapason), (self.__df, self.__diapason), (self.__ddf, self.__diapason),
                     show=False)
        xy_f, xy_df, xy_ddf = plots[0].get_points(), plots[1].get_points(), plots[2].get_points()

        # отрисовка графиков
        fig, axs = plt.subplots(1, 3)

        # первый
        axs[0].plot(xy_f[0], xy_f[1])
        axs[0].set_title(f"y={self.__f}")
        axs[0].grid()

        # второй
        axs[1].plot(xy_df[0], xy_df[1])
        axs[1].set_title(f"y={self.__df}")
        axs[1].grid()

        # третий
        axs[2].plot(xy_ddf[0], xy_ddf[1])
        axs[2].set_title(f"y={self.__ddf}")
        axs[2].grid()

        fig.suptitle(f'График функции y={self.__f} и ее первая и вторая производные.')
        fig.set_figheight(5)
        fig.set_figwidth(15)
        fig.subplots_adjust(wspace=.1)

        if save_mode:
            plt.savefig(f'{directory}/img_{name}_{self.saved_pictures}.png')
            self.saved_pictures += 1
        plt.show()

    @save_calculation
    def enumeration_method(self, eps: float):
        """
        Реализация метода перебора.
        :param eps: precision parameter
        """

        name = 'метод перебора'
        self.draw(name, save_mode=True)
        float_par = 4

        # count dots
        count = int((self.__diapason[2] - self.__diapason[1]) / eps + 1)

        # points x and y
        argument = np.linspace(self.__diapason[1], self.__diapason[2], count)
        argument = [round(_, float_par) for _ in argument]
        value = [round(self.__f.evalf(subs={x: ar}), float_par) for ar in argument]

        # send data for file
        yield name
        yield [self.__f]
        yield argument
        yield value

        # search minimum function
        min_index = 0
        min_value = value[0]
        for i in range(1, len(value)):
            if min_value > value[i]:
                min_value = value[i]
                min_index = i

        yield argument[min_index], value[min_index]

    @save_calculation
    def dichotomy_method(self, eps: float):
        """
        Реализация метода дихотомии.
        :param eps: precision parameter
        """

        name = 'метод дихотомии'
        self.draw(name, save_mode=True)
        headers = ['a', 'b', 'x1', 'x2', 'f1', 'f2', 'f1<f2', '|a-b|/2 <= eps']
        float_par = 4
        sigma = round(rnd.random() * 2 * eps, 4)

        # send required parameters
        yield name
        yield [self.__f]
        yield ['sigma=', sigma]
        yield headers

        a = self.__diapason[1]
        b = self.__diapason[2]

        while True:
            send_data = [a, b]

            x1 = (a + b - sigma) / 2
            x2 = (a + b + sigma) / 2
            f1 = self.__f.evalf(subs={x: x1})
            f2 = self.__f.evalf(subs={x: x2})

            # definition of a new segment
            if f1 <= f2:
                b = x2
            else:
                a = x1

            send_data += [x1, x2, f1, f2]
            send_data = list(map(lambda _: round(_, float_par), send_data))
            send_data += [f1 < f2, abs(a - b) / 2 <= eps]
            yield send_data

            # ending if
            if abs(a - b) / 2 <= eps:
                yield (b + a) / 2, self.__f.evalf(subs={x: (b + a) / 2})
                break

    @save_calculation
    def golden_section_method(self, eps: float):
        """
        Реализация метода золотого.
        :param eps: precision parameter
        """

        def __start_golden_coefficients(start: int, end: int):
            """
            Подсчет стартовых коэффициентов в методе золотого сечения
            """
            return start + (3 - np.sqrt(5)) * (end - start) / 2, start + (np.sqrt(5) - 1) * (end - start) / 2

        name = 'метод золотого сечения'
        self.draw(name, save_mode=True)
        headers = ['a', 'b', 'x1', 'x2', 'f1', 'f2', 'eps_i', 'f1<f2', '|a - b| < eps']
        float_par = 4

        yield name
        yield [self.__f]
        yield headers

        a = self.__diapason[1]
        b = self.__diapason[2]

        x1, x2 = __start_golden_coefficients(a, b)

        while True:
            f1, f2 = self.__f.evalf(subs={x: x1}), self.__f.evalf(subs={x: x2})

            send_data = [a, b, x1, x2, f1, f2, abs(b - a)]
            send_data = list(map(lambda _: round(_, float_par), send_data))
            send_data += [f1 < f2, abs(a - b) < eps]
            yield send_data

            if f1 <= f2:
                b = x2
                x2 = x1
                x1 = a + b - x1
            else:
                a = x1
                x1 = x2
                x2 = a + b - x2

            if abs(b - a) <= eps:
                yield a, b, '-', '-', '-', '-', abs(b-a), '-', abs(b-a) < eps
                yield (a + b) / 2, self.__f.evalf(subs={x: (a + b) / 2})
                break

    @save_calculation
    def fibonacci_method(self, eps: float):
        """
        Реализация метода Фибоначчи.
        :param eps: precision parameter
        """

        def __fibonacci_get_n():
            """
            Поиск необходимого числа иттераций
            """
            tmp = (self.__diapason[2] - self.__diapason[1]) / eps
            iter = 0
            while __fibonacci_number(iter) < tmp:
                iter += 1
            return iter - 2

        def __fibonacci_number(n: int):
            """
            Формула Бине для поиска числа Фибоначчи
            """
            tmp_1 = np.sqrt(5)
            tmp_2 = ((1 + tmp_1) / 2) ** n
            tmp_3 = ((1 - tmp_1) / 2) ** n
            return round(1 / tmp_1 * (tmp_2 - tmp_3))

        def __fibonacci_number_rec(n: int):
            """
            Рекурсивный алгоритм поиска чисел Фибоначчи
            """
            if n == 0:
                return 0
            elif n == 1:
                return 1
            else:
                return __fibonacci_number_rec(n - 1) + __fibonacci_number_rec(n - 2)

        name = 'метод Фибоначчи'
        self.draw(name, save_mode=True)
        headers = ['a', 'b', 'x1', 'x2', 'f1', 'f2', 'eps_i', 'f1<f2', 'i']
        steps = __fibonacci_get_n()
        float_par = 4

        yield name
        yield ['steps=', steps]
        yield headers

        a = self.__diapason[1]
        b = self.__diapason[2]
        if steps < 50:
            x1 = a + __fibonacci_number_rec(steps) * (b - a) / __fibonacci_number_rec(steps + 2)
        else:
            x1 = a + __fibonacci_number(steps) * (b - a) / __fibonacci_number(steps + 2)
        x2 = a + b - x1

        for i in range(1, steps):  # TODO: Подумать с какого коэффициента начинать подсчет
            f1, f2 = self.__f.evalf(subs={x: x1}), self.__f.evalf(subs={x: x2})

            send_data = [a, b, x1, x2, f1, f2]
            send_data = list(map(lambda _: round(_, float_par), send_data))
            send_data += [f1 < f2, i]
            yield send_data

            if f1 <= f2:
                b = x2
                x2 = x1
                x1 = a + b - x1
            else:
                a = x1
                x1 = x2
                x2 = a + b - x2

        yield (a + b) / 2, self.__f.evalf(subs={x: (a + b) / 2})

    def svenn_method(self, print_info=False):
        """
        Поиск интервала неопределенности методом Свенна
        :param print_info: print steps
        """
        x0 = 0
        h0 = 0.1

        # направление движения
        if self.__f.evalf(subs={x: x0}) < self.__f.evalf(subs={x: x0 + h0}):
            direction = -1
        else:
            direction = 1

        # первый шаг
        x1 = x0 + h0 * direction

        i = 1

        while True:
            i += 1
            x0 = x1
            x1 = x0 + 2 ** (i - 1) * h0 * direction
            f0 = self.__f.evalf(subs={x: x0})
            f1 = self.__f.evalf(subs={x: x1})

            if print_info:
                print(i, 2 ** (i - 1), x0, x1, f0, f1, f0 < f1)

            if f0 < f1:
                break

        return min(x0, x1), max(x0, x1)

    @save_calculation
    def newtons_method(self, eps: float):
        """
        метод Ньютона для поиска. Запускать до тех пор, пока количество шагов не будет минимальным. Так как может медленно сходится
        :param eps: precision parameter
        """

        def xi(xx):
            """
            Поиск следущего значения x
            :return: next x
            """
            return xx - self.__df.evalf(subs={x: xx}) / self.__ddf.evalf(subs={x: xx})

        name = 'метод Ньютона (касательных)'
        self.draw(name, save_mode=True)
        headers = ['xk', "f'(xk)"]
        float_par = 4
        x0 = rnd.random() * (self.__diapason[2] - self.__diapason[1]) + self.__diapason[1]

        yield name
        yield [x0]
        yield headers

        x_now = xi(x0)

        send_data = [x_now, self.__df.evalf(subs={x: x_now})]
        send_data = list(map(lambda _: round(_, float_par), send_data))
        yield send_data

        while True:
            x_now = xi(x_now)

            send_data = [x_now, self.__df.evalf(subs={x: x_now})]
            send_data = list(map(lambda _: round(_, float_par), send_data))
            yield send_data

            if abs(self.__df.evalf(subs={x: x_now})) <= eps:
                break

        yield x_now, self.__f.evalf(subs={x: x_now})

    @save_calculation
    def midpoint_method(self, eps: float):
        """
        Метод средней точки
        :param eps: точность поиска минимума
        """
        a, b = self.__diapason[1], self.__diapason[2]

        name = 'метод средней точки'
        self.draw(name, save_mode=True)
        headers = ['a', 'b', 'x_mid', 'df']
        float_par = 4

        yield name
        yield headers

        while True:
            x_mid = (a + b) / 2
            df = self.__df.evalf(subs={x: x_mid})

            send_data = [a, b, x_mid, df]
            send_data = list(map(lambda _: round(_, float_par), send_data))
            yield send_data

            if df > 0:
                b = x_mid
            else:
                a = x_mid

            if abs(df) <= eps:
                break

        yield x_mid, self.__f.evalf(subs={x: x_mid})

    @save_calculation
    def chord_method(self, eps: float):
        a, b = self.__diapason[1], self.__diapason[2]

        name = 'метод хорд'
        self.draw(name, save_mode=True)
        headers = ['a', 'b', 'x_mid', 'df']
        float_par = 4

        yield name
        yield headers

        while True:
            dfa, dfb = self.__df.evalf(subs={x: a}), self.__df.evalf(subs={x: b})
            x_mid = a - dfa / (dfa - dfb) * (a - b)
            df = self.__df.evalf(subs={x: x_mid})

            send_data = [a, b, x_mid, df]
            send_data = list(map(lambda _: round(_, float_par), send_data))
            yield send_data

            if df > 0:
                b = x_mid
            else:
                a = x_mid

            if abs(df) <= eps:
                # точность достигнута
                break

        yield x_mid, self.__f.evalf(subs={x: a})


if __name__ == '__main__':
    # f = OneDimensionalOptimization('sqrt(1+x**2)+exp(-2*x)', [0, 1])
    # f.enumeration_method(0.1)
    f = OneDimensionalOptimization('x**3/3-5*x+x*ln(x)', [1.5, 2])
    f.dichotomy_method(0.02)
    # f = OneDimensionalOptimization('x**3/3-5*x+x*ln(x)', [1.5, 2])
    # f.draw(save_mode=True)
    # f = OneDimensionalOptimization('x**4+8*x**3-6*x**2-72*x+90', [1.5, 2])
    # f.golden_section_method(0.05)
    # f = OneDimensionalOptimization('2*x**2+3*exp(-x)', [0, 1])
    # f.fibonacci_method(0.01)
    # f = OneDimensionalOptimization('x**3/2-100*sin(x)', [0, 3])
    # f.draw()
    # f = OneDimensionalOptimization('x**2+exp(-x)', [0, 3])
    # f.newtons_method(0.0001)


