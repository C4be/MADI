import numpy as np
import cv2 as cv


class Engine:
    # class settings
    __PATH = r'./video/'
    __WINDOW_HEIGHT = 600
    __WINDOW_WIDTH = 300
    __GAUSSIAN_BLUR = 1
    __THRESHOLD_MIN_VAL = 50
    __THRESHOLD_MAX_VAL = 100
    __LineLength = 80
    __MaxLineGap = 50
    __SET_WINDOWS = ['Config', 'Settings']

    def __init__(self, window_name: str, filename: str):
        self.__name = window_name
        self.__filename = filename
        self.__wins = {
            'frame': None,
            'grey': None,
            'gaussian': None,
            'canny': None,
            'mask': None,
            'hough': None,
        }

    def run(self):
        self.initial()
        Engine.create_setting_window()
        self.config_loop()
        self.main_loop()
        pass

    def processing(self):
        self.__wins['grey'] = Engine.to_gray(self.__wins['frame'])
        self.__wins['gaussian'] = Engine.reduce_noise(self.__wins['grey'])
        self.__wins['canny'] = Engine.canny(self.__wins['gaussian'])
        self.__wins['mask'] = Engine.mask(self.__wins['canny'])
        self.__wins['hough'] = Engine.Hough(self.__wins['frame'], self.__wins['mask'])

    def initial(self):
        full_path = Engine.__PATH + self.__filename
        cap = cv.VideoCapture(full_path)
        er, self.__wins['frame'] = cap.read()
        if er:
            self.processing()
        else:
            print('Can\'t receive frame from file. Exiting...')
        cap.release()
        cv.destroyAllWindows()

    def config_loop(self):
        full_path = Engine.__PATH + self.__filename
        cap = cv.VideoCapture(full_path)
        if not cap.isOpened():
            cap.release()
            cv.destroyAllWindows()
        er, self.__wins['frame'] = cap.read()
        copy = self.__wins.get('frame')
        while True:
            if er:
                self.__wins['frame'] = copy
                self.processing()
                self.resize_wins()
                self.display_wins()
            else:
                print('Can\'t receive frame from file. Exiting...')
                break

            if cv.waitKey(30) == ord('w') & 0xFF:
                print('Keyboard exiting...')
                break

        cap.release()
        cv.destroyAllWindows()

    def main_loop(self):
        full_path = Engine.__PATH + self.__filename
        cap = cv.VideoCapture(full_path)
        while cap.isOpened():
            er, self.__wins['frame'] = cap.read()
            if er:
                self.processing()
                self.resize_wins()
                self.display_wins()
            else:
                print('Can\'t receive frame from file. Exiting...')
                break

            if cv.waitKey(30) == ord('q') & 0xFF:
                print('Keyboard exiting...')
                break

        cap.release()
        cv.destroyAllWindows()

    @staticmethod
    def to_gray(frame):
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    @staticmethod
    def reduce_noise(frame):
        return cv.GaussianBlur(frame, (Engine.__GAUSSIAN_BLUR, Engine.__GAUSSIAN_BLUR), 0)

    @staticmethod
    def resize(frame):
        return cv.resize(frame, (Engine.__WINDOW_WIDTH, Engine.__WINDOW_HEIGHT))

    @staticmethod
    def canny(frame):
        return cv.Canny(frame, Engine.__THRESHOLD_MIN_VAL, Engine.__THRESHOLD_MAX_VAL)

    @staticmethod
    def mask(frame):
        height, width = frame.shape

        mask = np.zeros_like(frame)

        # create white triangle (white mask)
        triangle = np.array([[
            (180, height),
            (width // 2 - 40, int(height * .76)),
            (width // 2 + 15, int(height * .76)),
            (int(.8 * width), height)
        ]])

        # triangle = np.array([[
        #     (140, height),
        #     (width // 2 - 70, int(height * .5)),
        #     (width // 2 + 110, int(height * .5)),
        #     (int(1 * width), height - 90)
        # ]])

        cv.fillPoly(mask, triangle, 255)
        return cv.bitwise_and(frame, mask)

    @staticmethod
    def Hough(frame, canny):
        copy = frame.copy()
        lines = cv.HoughLinesP(canny, 1, np.pi / 180, 80, minLineLength=Engine.__LineLength, maxLineGap=Engine.__MaxLineGap)
        if not lines is None:
            rmx1, rmx2, rmy1, rmy2, rmcount = 0, 0, 0, 0, 0  # усреднение по правому
            lmx1, lmx2, lmy1, lmy2, lmcount = 0, 0, 0, 0, 0  # усреднение по правому
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # filter lines
                if abs(y1 - y2) / abs(x1 - x2) > 0.2:  # angle < 20 grad
                    # averange line
                    if (x2 - x1) / (y2 - y1) > 0:
                        rmx1 += x1
                        rmx2 += x2
                        rmy1 += y1
                        rmy2 += y2
                        rmcount += 1
                        # print(f'{x1} {x2} {y1} {y2} left')
                    else:
                        lmx1 += x1
                        lmx2 += x2
                        lmy1 += y1
                        lmy2 += y2
                        lmcount += 1
                        # print(f'{x1} {x2} {y1} {y2} right')
            if rmcount:
                cv.line(copy, (rmx1 // rmcount, rmy1 // rmcount), (rmx2 // rmcount, rmy2 // rmcount), (0, 0, 255), 3)
            if lmcount:
                cv.line(copy, (lmx1 // lmcount, lmy1 // lmcount), (lmx2 // lmcount, lmy2 // lmcount), (255, 0, 0), 3)

        return copy

    def resize_wins(self):
        """
        Изменение размера всех окон.
        """
        for key, win in self.__wins.items():
            if win is not None:
                self.__wins[key] = Engine.resize(win)
            else:
                print(f'Не удалось поменять размер окона ({key})')

    def display_wins(self):
        for key, win in self.__wins.items():
            if win is not None:
                cv.imshow(f'{self.__name} ({key})', win)
            else:
                print(f'Не удалось поменять размер окона ({key})')

    @staticmethod
    def upd_setting_win_height(val):
        Engine.__WINDOW_HEIGHT = val

    @staticmethod
    def upd_setting_win_width(val):
        Engine.__WINDOW_WIDTH = val

    @staticmethod
    def upd_setting_win_gaussian(val):
        if val == 0:
            Engine.__GAUSSIAN_BLUR = 1
        elif val % 2 == 0:
            Engine.__GAUSSIAN_BLUR = val - 1
        else:
            Engine.__GAUSSIAN_BLUR = val

    @staticmethod
    def upd_setting_win_thresh_min(val):
        Engine.__THRESHOLD_MIN_VAL = val

    @staticmethod
    def upd_setting_win_thresh_max(val):
        left = Engine.__THRESHOLD_MIN_VAL
        if val < left:
            Engine.__THRESHOLD_MAX_VAL = left + 1
            cv.setTrackbarPos('THR_MAX_VAL', Engine.__SET_WINDOWS[1], left + 1)

    @staticmethod
    def upd_line_len(val):
        Engine.__LineLength = val

    @staticmethod
    def upd_line_gap(val):
        Engine.__MaxLineGap = val

    @staticmethod
    def create_setting_window():
        """
        Создание окон для динамической настройки системы. (Trackbars)
        """
        # создание окон с трекбарами
        for win in Engine.__SET_WINDOWS:
            cv.namedWindow(win)
            cv.resizeWindow(win, 500, 200)

        # конфигурация окон
        cv.createTrackbar('HEIGHT', Engine.__SET_WINDOWS[0], 300, 720, Engine.upd_setting_win_height)
        cv.createTrackbar('WIDTH', Engine.__SET_WINDOWS[0], 600, 1280, Engine.upd_setting_win_width)
        cv.createTrackbar('GAUSSIAN_BLUR (ODD)', Engine.__SET_WINDOWS[1], 0, 11, Engine.upd_setting_win_gaussian)
        cv.createTrackbar('MIN_VAL', Engine.__SET_WINDOWS[1], 0, 125, Engine.upd_setting_win_thresh_min)
        cv.createTrackbar('MAX_VAL', Engine.__SET_WINDOWS[1], 126, 255, Engine.upd_setting_win_thresh_max)
        cv.createTrackbar('LineLength', Engine.__SET_WINDOWS[1], 0, 500, Engine.upd_line_len)
        cv.createTrackbar('LineGap', Engine.__SET_WINDOWS[1], 0, 500, Engine.upd_line_gap)


if __name__ == '__main__':
    e = Engine('App', 'test.mp4')
    e.run()
