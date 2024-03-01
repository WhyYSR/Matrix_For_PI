import flet as ft
import json


class MainWindow:
    solution_history = []
    rounding = 3
    method = 'Gauss'
    current_language = 'en'
    theme_mode = 'light'

    def __init__(self, page):
        self.page = page
        self.page.theme_mode = self.theme_mode
        self.page.window_maximized = True
        self.page.window_width = 1920
        self.theme = 'light'

        self.page.translations = LoadFiles.load_translations(self.page, "Translate.json")

    def main_window_page(self):
        """Создает интерфейс приложения."""
        self.page.controls.clear()
        MainWindow.create_top_panel(self)
        MainWindow.create_dimension_selection_page(self)

    def create_top_panel(self):
        settings_manager = SettingsManager(self.page)
        settings_button = ft.IconButton(
            icon=ft.icons.SETTINGS,
            on_click=lambda e: settings_manager.show_settings_page(),
            icon_size=39,
            style=ft.ButtonStyle(
                color={
                    ft.MaterialState.DEFAULT: ft.colors.BLACK if self.page.theme_mode == 'light' else ft.colors.WHITE},
                bgcolor={
                    ft.MaterialState.HOVERED: ft.colors.PURPLE if self.page.theme_mode == 'light' else ft.colors.PURPLE,
                    "": ft.colors.WHITE if self.page.theme_mode == 'light' else ft.colors.BLACK},
                overlay_color=ft.colors.TRANSPARENT,
                elevation={"pressed": 0, "": 1})
        )
        help_button = ft.TextButton("FAQ",
                                    on_click=lambda e: CreateHelpPage(self.page).show_help_page(),
                                    style=ft.ButtonStyle(
                                        color={
                                            ft.MaterialState.DEFAULT: ft.colors.BLACK if self.page.theme_mode == 'light' else ft.colors.WHITE},
                                        bgcolor={
                                            ft.MaterialState.HOVERED: ft.colors.PURPLE if self.page.theme_mode == 'light' else ft.colors.PURPLE,
                                            "": ft.colors.WHITE if self.page.theme_mode == 'light' else ft.colors.BLACK},
                                        overlay_color=ft.colors.TRANSPARENT,
                                        elevation={"pressed": 0, "": 1}),
                                    width=100,
                                    height=50)
        history_button = ft.TextButton("History",
                                       on_click=lambda e: CreateHistoryPage(self.page,
                                                                            self.current_language,
                                                                            self.solution_history).
                                       show_history_page(),
                                       style=ft.ButtonStyle(
                                           color={
                                               ft.MaterialState.DEFAULT: ft.colors.BLACK if self.page.theme_mode == 'light' else ft.colors.WHITE},
                                           bgcolor={
                                               ft.MaterialState.HOVERED: ft.colors.PURPLE if self.page.theme_mode == 'light' else ft.colors.PURPLE,
                                               "": ft.colors.WHITE if self.page.theme_mode == 'light' else ft.colors.BLACK},
                                           overlay_color=ft.colors.TRANSPARENT,
                                           elevation={"pressed": 0, "": 1}),
                                       width=100,
                                       height=50)

        buttons = ft.Row([
            help_button,
            history_button,
            settings_button],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            width=245)

        header = ft.Container(
            content=ft.Row([
                ft.Text("Linear solver ", size=30, color='black' if self.page.theme_mode == 'light' else 'white'),
                buttons],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            width=self.page.window_width,
            border=ft.border.all(1, 'black' if self.page.theme_mode == 'light' else "WHITE"),
            border_radius=10,
            bgcolor='white' if self.page.theme_mode == 'light' else 'black',
            padding=10,
            margin=-10)

        self.page.on_keyboard_event = KeyboardBindings(self.page).on_keyboard

        self.page.add(header)

    def create_dimension_selection_page(self):
        size_entry = ft.TextField(
            hint_text=self.page.translations['labels']['square_matrix_size'][self.current_language],
            hint_style=ft.TextStyle(color='black' if self.page.theme_mode == 'light' else 'yellow'),
            text_align=ft.TextAlign.CENTER,
            color='black' if self.page.theme_mode == 'light' else 'yellow',
            width=600,
            text_size=30,
            border_color='white' if self.page.theme_mode == 'light' else 'black'
        )

        matrix_input_size_slider = ft.Slider(
            min=1,
            max=10,
            width=100,
            active_color='green',
            inactive_color='red',
            divisions=9,
            adaptive=False,
            label='Matrix Input Size: {value}',
            round=0,
            thumb_color='blue',
            value=5,
            rotate=-1.55,
            on_change=lambda e: print(f"Selected input size: {matrix_input_size_slider.value}")
        )

        submit_button = ft.TextButton(
            self.page.translations['buttons']['confirm'][self.current_language],
            on_click=lambda e: CreateMatrixInputPage(self.page, size_entry.value).validate_and_create_matrix_input_page(
                size_entry.value),
            style=ft.ButtonStyle(
                color={
                    ft.MaterialState.DEFAULT: ft.colors.BLACK if self.page.theme_mode == 'light' else ft.colors.WHITE},
                bgcolor={
                    ft.MaterialState.HOVERED: ft.colors.PURPLE if self.page.theme_mode == 'light' else ft.colors.PURPLE,
                    "": ft.colors.WHITE if self.page.theme_mode == 'light' else ft.colors.BLACK},
                overlay_color=ft.colors.TRANSPARENT,
                elevation={"pressed": 0, "": 1}),
            width=150,
            height=50,
        )

        text_1 = ft.Container(
            alignment=ft.alignment.center,
            margin=ft.margin.only(left=150, right=25, top=50),
            content=ft.Row(
                [ft.Text(self.page.translations['menu']['Main_page_text_1'][self.current_language], size=35)],
                alignment=ft.MainAxisAlignment.CENTER),
            width=600,
            border=ft.border.all(0, 'white' if self.page.theme_mode == 'light' else "black"),
            border_radius=10,
            bgcolor='white' if self.page.theme_mode == 'light' else 'black',
            padding=0
        )
        pull = ft.Container(
            alignment=ft.alignment.center,
            margin=ft.margin.only(left=150, right=25, top=75),
            content=ft.Row([size_entry], alignment=ft.MainAxisAlignment.CENTER),
            width=600,
            border=ft.border.all(1, 'black' if self.page.theme_mode == 'light' else "WHITE"),
            border_radius=10,
            bgcolor='white' if self.page.theme_mode == 'light' else 'black',
            padding=0
        )
        button_container = ft.Container(
            alignment=ft.alignment.center,
            margin=ft.margin.only(left=150, right=25, top=50),
            content=ft.Row([submit_button], alignment=ft.MainAxisAlignment.CENTER),
            width=600,
            border=ft.border.all(0, 'white' if self.page.theme_mode == 'light' else "black"),
            border_radius=10,
            bgcolor='white' if self.page.theme_mode == 'light' else 'black',
            padding=0
        )
        self.page.add(text_1, pull, button_container)
        self.page.update()


class SettingsManager(MainWindow):
    def __init__(self, page):
        super().__init__(page)  # Вызываем конструктор родительского класса

    def show_settings_page(self):
        self.page.controls.clear()

        language_dropdown = ft.Dropdown(
            options=[
                ft.dropdown.Option('Русский'),
                ft.dropdown.Option('English')
            ],
            hint_text=self.page.translations['labels']['language'][self.current_language],
            on_change=lambda e: self.change_language(language_dropdown.value)
        )

        full_screen_dropdown = ft.Dropdown(
            options=[
                ft.dropdown.Option(self.page.translations['labels']['yes'][self.current_language]),
                ft.dropdown.Option(self.page.translations['labels']['no'][self.current_language])
            ],
            hint_text=self.page.translations['labels']['fullscreen_mode'][self.current_language],
            on_change=lambda e: self.change_full_screen_mode(full_screen_dropdown.value)
        )

        theme_dropdown = ft.Dropdown(
            options=[
                ft.dropdown.Option(self.page.translations['labels']['light'][self.current_language]),
                ft.dropdown.Option(self.page.translations['labels']['dark'][self.current_language])
            ],
            hint_text=self.page.translations['labels']['choose_theme'][self.current_language],
            on_change=lambda e: self.change_theme(theme_dropdown.value)
        )

        rounding_dropdown = ft.Dropdown(
            options=[
                ft.dropdown.Option('1'),
                ft.dropdown.Option('2'),
                ft.dropdown.Option('3'),
                ft.dropdown.Option('4'),
                ft.dropdown.Option('5')
            ],
            hint_text=self.page.translations['labels']['round_to'][self.current_language],
            on_change=lambda e: self.change_rounding(rounding_dropdown.value)
        )

        method_dropdown = ft.Dropdown(
            options=[
                ft.dropdown.Option(self.page.translations['labels']['gauss'][self.current_language]),
                ft.dropdown.Option('LU')
            ],
            hint_text=self.page.translations['labels']['choose_solution_method'][self.current_language],
            on_change=lambda e: self.change_method(method_dropdown.value)
        )

        back_button = CustomButton(self.page.translations['buttons']['back'][self.current_language],
                                   lambda e: self.main_window_page(), self.page)

        self.create_top_panel()

        self.page.add(ft.Text('\n\n\n\n\n\n\n\n\n\n\n'))

        self.page.add(ft.Column([ft.Row([theme_dropdown, full_screen_dropdown],
                                        alignment=ft.MainAxisAlignment.CENTER),
                                 ft.Row([rounding_dropdown, method_dropdown],
                                        alignment=ft.MainAxisAlignment.CENTER),
                                 ft.Row([language_dropdown],
                                        alignment=ft.MainAxisAlignment.CENTER),
                                 ft.Row([back_button],
                                        alignment=ft.MainAxisAlignment.CENTER)],
                                alignment=ft.MainAxisAlignment.CENTER))

        self.page.update()

    def change_full_screen_mode(self, mode: str):
        if mode in self.page.translations['labels']['yes'].values():
            self.page.window_full_screen = True
        else:
            self.page.window_full_screen = False
        self.page.update()

    def change_color(self, color: str):
        self.page.text_color = color.lower()
        self.page.update()

    def change_theme(self, theme: str):
        if theme in self.page.translations['labels']['light'].values():
            MainWindow.theme_mode = 'light'
        else:
            MainWindow.theme_mode = 'dark'
        self.show_settings_page()

    def change_rounding(self, rounding: str):
        MainWindow.rounding = int(rounding)
        self.page.update()

    def change_method(self, method: str):
        if method in self.page.translations['labels']['gauss']:
            MainWindow.method = 'Gauss'
        else:
            MainWindow.method = 'lu'

    def change_language(self, language: str):
        if language == 'English':
            MainWindow.current_language = 'en'
        elif language == 'Русский':
            MainWindow.current_language = 'ru'
        else:
            print("Translation not found for selected language.")

        self.show_settings_page()


class Error(Exception):
    """Базовый класс для других исключений"""
    pass


class InvalidInputError(MainWindow, Error):
    """Вызывается, когда ввод недействителен"""

    def __init__(self, page):
        super().__init__(page)

    def show_error_alert(self, message):
        alert_dialog = ft.AlertDialog(
            title=ft.Text(self.page.translations['messages']['error'][self.current_language]),
            content=ft.Text(message),
            actions=[ft.TextButton("OK", on_click=lambda e: self.close_dialog(self.page))],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.dialog = alert_dialog
        alert_dialog.open = True
        self.page.update()

    def close_dialog(self, page):
        """
        Закрывает диалоговое окно.

        Параметры:
        - page: объект страницы для отображения

        Возвращает:
        - None
        """
        page.dialog.open = False
        page.update()


class CreateMatrixInputPage(MainWindow):
    def __init__(self, page, size):
        super().__init__(page)
        self.size = size

    def clear_matrix(self, entries):
        """Эта функция очищает введенные значения матрицы. Она принимает один параметр: entries (список полей для ввода значений матрицы)."""
        for row in entries:
            for entry in row:
                entry.value = ""
        self.page.update()

    def create_entries(self, size):
        return [[ft.TextField(value="",
                              hint_text="0",
                              hint_style=ft.TextStyle(color='grey' if self.page.theme_mode == 'light' else 'green'),
                              color='black' if self.page.theme_mode == 'light' else 'yellow',
                              text_align=ft.TextAlign.CENTER,
                              width=200,
                              text_size=50)
                 for _ in range(int(size) + 1)]
                for _ in range(int(size))]

    def create_matrix_input_page(self, entries):
        """
        Создает страницу для ввода значений матрицы.

        Параметры:
        - page: объект страницы, на которой будет создан ввод
        - size: размер квадратной матрицы
        - entries: список полей для ввода значений матрицы
        """
        self.page.controls.clear()
        save_button = CustomButton(self.page.translations['buttons']['confirm'][self.current_language],
                                   lambda e: SolutionPage(self.page,
                                                          self.size,
                                                          self.current_language,
                                                          entries,
                                                          self.method).
                                   show_create_matrix_page(entries), self.page)
        back_button = CustomButton(self.page.translations['buttons']['back'][self.current_language],
                                   lambda e: MainWindow.main_window_page(self), self.page)
        clear_button = CustomButton(self.page.translations['buttons']['clear'][self.current_language],
                                    lambda e: self.clear_matrix(entries), self.page)

        MainWindow.create_top_panel(self)

        self.page.add(ft.Row([ft.Text(self.page.translations['menu']['final_solve'][self.current_language],
                                      size=30,
                                      color='black' if self.page.theme_mode == 'light' else 'purple')],
                             alignment=ft.MainAxisAlignment.CENTER))
        for row in entries:
            self.page.add(ft.Row(row, alignment=ft.MainAxisAlignment.CENTER))  # Выравнивание по центру
        save_button.enabled = False  # Блокируем кнопку "Сохранить" при открытии страницы
        self.page.add(ft.Row([back_button, clear_button, save_button],
                             alignment=ft.MainAxisAlignment.CENTER))  # Выравнивание кнопки по центру
        self.page.update()

    def validate_and_create_matrix_input_page(self, size_value):
        try:
            size = int(size_value)
            if (2 <= size <= 5) == 0:
                InvalidInputError(self.page).show_error_alert(
                    self.page.translations['messages']['matrix_size'][self.current_language])
            else:
                self.create_matrix_input_page(self.create_entries(size))
        except ValueError:
            InvalidInputError(self.page).show_error_alert(
                self.page.translations['messages']['matrix_size'][self.current_language])


class SolutionPage(MainWindow):

    def __init__(self, page, size, current_language, entries, method):
        super().__init__(page)
        self.size = size
        self.current_language = current_language
        self.entries = entries
        self.method = method

    def is_valid_input(self, entries):
        for row in entries:
            for entry in row:
                cleaned_value = entry.value.replace(',', '.')
                try:
                    float(cleaned_value)
                except ValueError:
                    return False
        return True

    def show_create_matrix_page(self, entries):
        self.page.controls.clear()
        if self.is_valid_input(entries):
            coefficients_matrix = []
            constants_vector = []
            for row in entries:
                row_coefficients = []
                for entry in row[:-1]:
                    # Заменяем запятую на точку, если она есть
                    value = entry.value.replace(',', '.')
                    row_coefficients.append(float(value))
                coefficients_matrix.append(row_coefficients)
                # Также обрабатываем значение вектора констант
                value = row[-1].value.replace(',', '.')
                constants_vector.append(float(value))
            print(f"{self.page.translations['menu']['coefficients_matrix'][self.current_language]}:",
                  coefficients_matrix)
            print(f"{self.page.translations['menu']['constants_vector'][self.current_language]}:", constants_vector)
            solver = EquationSolver(self.page, self.size, self.current_language, entries)
            if self.method == 'Gauss':
                A, B = solver.the_triangular_matrix(coefficients_matrix, constants_vector)
                X = solver.backward_substitution(A, B)
            else:
                X = solver.solve_lu(coefficients_matrix, constants_vector)
            print(f"{self.page.translations['menu']['final_solve'][self.current_language]}:", X)
            for i in range(len(X)):
                X[i] = round(X[i], self.rounding)
            self.solution_history.append(X)
            print(self.solution_history)
            self.show_solution_page(X, entries)
        else:
            CreateMatrixInputPage(self.page, self.size).create_matrix_input_page(entries)
            InvalidInputError(self.page).show_error_alert(self.
                                                          page.
                                                          translations['messages']
                                                          ['invalid_input']
                                                          [self.current_language])

    def show_solution_page(self, solution, entries):

        self.page.controls.clear()

        MainWindow.create_top_panel(self)

        answer = ""
        for index, item in enumerate(solution, start=1):
            item_str = str(item)
            if index < len(solution):
                answer += f"x{index} = {item_str:{len(item_str) + 4}}\n"
            else:
                answer += f"x{index} = {item_str}"

        answer = ft.Text(answer,
                         color='black' if self.page.theme_mode == 'light' else 'green',
                         size=30)

        back_button = CustomButton(self.page.translations['buttons']['back'][self.current_language],
                                   lambda e: CreateMatrixInputPage(self.page, self.size).
                                   create_matrix_input_page(entries),
                                   self.page)

        exit_button = CustomButton(self.page.translations['buttons']['exit'][self.current_language],
                                   lambda e: self.page.window_close(),
                                   self.page)

        restart_button = CustomButton(self.page.translations['buttons']['restart'][self.current_language],
                                      lambda e: MainWindow.main_window_page(self),
                                      self.page)

        self.page.add(ft.Column([
            ft.Row([ft.Text(self.page.translations['labels']['solve_system'][self.current_language],
                            color='black' if self.page.theme_mode == 'light' else 'purple',
                            size=35)],
                   alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([answer],
                   alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([back_button, restart_button],
                   alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([exit_button],
                   alignment=ft.MainAxisAlignment.CENTER)],
            alignment=ft.MainAxisAlignment.CENTER))

        self.page.update()


class EquationSolver(MainWindow):
    def __init__(self, page, size, current_language, entries):
        super().__init__(page)
        self.size = size
        self.current_language = current_language
        self.entries = entries

    def the_triangular_matrix(self, A, B):
        """
        Приводит матрицу к треугольному виду.

        Параметры:
        - A: двумерный список (матрица коэффициентов системы уравнений)
        - B: список (столбец свободных членов)
        - page: объект страницы для отображения сообщений об ошибках

        Возвращает:
        - A: список (Матрица перемех) или None, если матрица не квадратная
        - B: список (решение системы уравнений)
        """
        n = len(A)

        if len(A) != len(A[0]):
            (InvalidInputError(self.page).
             show_error_alert(self.page.translations['messages']['non_square_matrix_gauss'][self.current_language]))
            return None, None

        det = 1
        epsilon = 1e-10

        for i in range(n):
            max_elem = abs(A[i][i])
            max_row = i

            # Поиск максимального элемента в текущем столбце
            for k in range(i + 1, n):
                if abs(A[k][i]) > max_elem:
                    max_elem = abs(A[k][i])
                    max_row = k

            # Обмен строк для улучшения численной устойчивости
            A[i], A[max_row] = A[max_row], A[i]
            B[i], B[max_row] = B[max_row], B[i]

            # Приведение матрицы к треугольному виду
            for j in range(i + 1, n):
                coef = A[j][i] / A[i][i]
                for k in range(i, n):
                    A[j][k] -= coef * A[i][k]
                B[j] -= coef * B[i]

            # Расчет определителя
            det *= A[i][i]  # знак не важен

        A_test = [row for row in A if any(abs(el) > epsilon for el in row)]
        if len(A) != len(A_test):
            (InvalidInputError(self.page).
             show_error_alert(self.page.translations['messages']['non_square_matrix_gauss'][self.current_language]))
            return None, None

        return A, B

    def solve_lu(self, A, b):
        """
        Решает систему линейных уравнений методом LU-разложения.

        Параметры:
        - A: двумерный список (матрица коэффициентов системы уравнений)
        - b: список (столбец свободных членов)

        Возвращает:
        - x: список (решение системы уравнений)
        """
        L, U = self.lu_decomposition(A)  # Вызов LU-разложения для матрицы A
        y = self.forward_substitution(L, b)  # Прямая подстановка для нахождения y
        x = self.backward_substitution(U, y)  # Обратная подстановка для нахождения x
        return x  # Возвращение решения системы уравнений

    def lu_decomposition(self, A):
        """
        Выполняет LU-разложение матрицы A.

        Параметры:
        - A: двумерный список (исходная матрица)

        Возвращает:
        - L: двумерный список (нижнетреугольная матрица L)
        - U: двумерный список (верхнетреугольная матрица U)
        """
        n = len(A)
        L = [[0.0] * n for _ in range(n)]  # Инициализация матрицы L нулями
        U = [[0.0] * n for _ in range(n)]  # Инициализация матрицы U нулями

        for i in range(n):
            L[i][i] = 1.0  # Установка единиц на диагонали матрицы L

        for k in range(n):
            U[k][k] = A[k][k]  # Присваивание элементу диагонали матрицы U соответствующего элемента исходной матрицы A
            for j in range(k + 1, n):
                L[j][k] = A[j][k] / U[k][k]  # Вычисление элементов нижнетреугольной матрицы L
                U[k][j] = A[k][
                    j]  # Присваивание элементу верхнетреугольной матрицы U соответствующего элемента исходной матрицы A
            for i in range(k + 1, n):
                for j in range(k + 1, n):
                    A[i][j] -= L[i][k] * U[k][j]  # Обновление элементов исходной матрицы A

        return L, U  # Возвращение матриц L и U

    def forward_substitution(self, L, b):
        """
        Выполняет прямую подстановку.

        Параметры:
        - L: двумерный список (нижнетреугольная матрица L)
        - b: список (столбец свободных членов)

        Возвращает:
        - y: список (результат прямой подстановки)
        """
        n = len(L)
        y = [0.0] * n  # Инициализация вектора y нулями
        for i in range(n):
            y[i] = b[i]  # Присваивание элементу вектора y значения соответствующего элемента вектора b
            for j in range(i):
                y[i] -= L[i][j] * y[j]  # Прямая подстановка
        return y  # Возвращение вектора y

    def backward_substitution(self, U, y):
        """
        Выполняет обратную подстановку.

        Параметры:
        - U: двумерный список (верхнетреугольная матрица U)
        - y: список (результат прямой подстановки)

        Возвращает:
        - x: список (решение системы уравнений)
        """
        n = len(U)
        x = [0.0] * n  # Инициализация вектора x нулями
        for i in range(n - 1, -1, -1):
            x[i] = y[i]  # Присваивание элементу вектора x значения соответствующего элемента вектора y
            for j in range(i + 1, n):
                x[i] -= U[i][j] * x[j]  # Обратная подстановка
            x[i] /= U[i][i]  # Деление элемента вектора x на соответствующий диагональный элемент матрицы U
        return x  # Возвращение вектора x


class CreateHistoryPage(MainWindow):
    def __init__(self, page, current_language, solution_history):
        super().__init__(page)
        self.current_language = current_language
        self.solution_history = solution_history

    def show_history_page(self):
        self.page.controls.clear()

        MainWindow.create_top_panel(self)

        for i, sol in enumerate(self.solution_history, start=1):
            solution_text = ft.Row(
                [ft.Text(f"{self.page.translations['messages']['solution'][self.current_language]} {i}: {sol}",
                         color='black' if self.page.theme_mode == 'light' else 'green',
                         size=35)],
                alignment=ft.MainAxisAlignment.CENTER)
            self.page.add(solution_text)

            # Добавляем кнопку для возврата на предыдущую страницу
        back_button = CustomButton(self.page.translations['buttons']['back'][self.current_language],
                                   lambda e: MainWindow.main_window_page(self),
                                   self.page)
        self.page.add(ft.Row([back_button], alignment=ft.MainAxisAlignment.CENTER))

        self.page.update()


class CreateHelpPage(MainWindow):
    def __init__(self, page):
        super().__init__(page)

    def show_help_page(self):
        self.page.controls.clear()

        back_button = CustomButton(self.page.translations['buttons']['back'][self.current_language],
                                   lambda e: MainWindow.main_window_page(self),
                                   self.page)

        faq_1 = self.create_expansion_tile(self.page.translations['menu']['faq_1'][self.current_language],
                                           self.page.translations['menu']['ans_1'][self.current_language]
                                           )
        faq_2 = self.create_expansion_tile(self.page.translations['menu']['faq_2'][self.current_language],
                                           self.page.translations['menu']['ans_2'][self.current_language]
                                           )
        faq_3 = self.create_expansion_tile(self.page.translations['menu']['faq_3'][self.current_language],
                                           self.page.translations['menu']['ans_3'][self.current_language]
                                           )
        faq_4 = self.create_expansion_tile(self.page.translations['menu']['faq_4'][self.current_language],
                                           self.page.translations['menu']['ans_4'][self.current_language]
                                           )
        MainWindow.create_top_panel(self)

        text_container = ft.Container(
            content=ft.Column([ft.Text(self.page.translations['menu']['FAQ_page_text'][self.current_language], size=35),
                               ft.Text('\n'),
                               faq_1,
                               faq_2,
                               faq_3,
                               faq_4, ],
                              alignment=ft.MainAxisAlignment.START),

            width=self.page.window_width,
            border_radius=0,
            bgcolor='white' if self.page.theme_mode == 'light' else 'black',
            padding=10,
            margin=ft.margin.only(left=175, top=25))
        self.page.add(text_container)

        self.page.add(ft.Row([back_button], alignment=ft.MainAxisAlignment.CENTER))

        self.page.update()

    def create_expansion_tile(self, title, controls):
        return (ft.ExpansionTile(
            title=ft.Text(title, size=20),
            width=1200,
            subtitle=ft.Text(),
            affinity=ft.TileAffinity.PLATFORM,
            maintain_state=False,
            collapsed_text_color='black' if self.page.theme_mode == 'light' else 'yellow',
            text_color='black' if self.page.theme_mode == 'light' else 'yellow',
            controls=[ft.ListTile(title=ft.Text(controls, size=18))],
        ))


class LoadFiles:
    def __init__(self):
        pass

    def load_translations(self, file_path):
        """
        Загружает переводы из файла JSON.

        Параметры:
        - file_path: str, путь к файлу с переводами.

        Возвращает:
        - dict, словарь с переводами.
        """
        translations = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                translations = json.load(file)
        except FileNotFoundError:
            print(f"Файл с переводами не найден: {file_path}")
        except json.JSONDecodeError:
            print(f"Ошибка декодирования файла с переводами: {file_path}")

        return translations


class KeyboardBindings(MainWindow):
    def __init__(self, page):
        super().__init__(page)  # Вызываем конструктор родительского класса

    def bind_keyboard_events(self):
        self.page.on_keyboard_event = self.on_keyboard

    def on_keyboard(self, e: ft.KeyboardEvent):
        if e.key == "H":
            CreateHelpPage(self.page).show_help_page()

    def show_faq_page(self):
        CreateHelpPage(self.page).show_help_page()  # Показываем страницу FAQ


class CustomButton(ft.TextButton):
    """
    Класс для создания настраиваемой кнопки.

    Атрибуты:
    - text: Текст, отображаемый на кнопке.
    - on_click_action: Действие, выполняемое при нажатии на кнопку.
    - page: Объект страницы, связанный с кнопкой.
    - width: Ширина кнопки (по умолчанию 150).
    - height: Высота кнопки (по умолчанию 50).
    """

    def __init__(self, text, on_click_action, page, width=150, height=50):
        """
        Инициализация настраиваемой кнопки.

        Параметры:
        - text: Текст, отображаемый на кнопке.
        - on_click_action: Действие, выполняемое при нажатии на кнопку.
        - page: Объект страницы, связанный с кнопкой.
        - width: Ширина кнопки (по умолчанию 150).
        - height: Высота кнопки (по умолчанию 50).
        """
        super().__init__(
            text,
            width=width,
            height=height,
            on_click=on_click_action,
            style=ft.ButtonStyle(
                color={ft.MaterialState.DEFAULT: ft.colors.WHITE if page.theme_mode == 'light' else ft.colors.BLACK},
                bgcolor={ft.MaterialState.HOVERED: ft.colors.GREEN if page.theme_mode == 'light' else ft.colors.GREEN,
                         "": ft.colors.BLACK if page.theme_mode == 'light' else ft.colors.WHITE},
                overlay_color=ft.colors.TRANSPARENT,
                elevation={"pressed": 0, "": 1},
                animation_duration=500,
                shape={
                    ft.MaterialState.HOVERED: ft.RoundedRectangleBorder(radius=20),
                    ft.MaterialState.DEFAULT: ft.RoundedRectangleBorder(radius=4),
                },
            ),
        )
        self.page = page


def main(page: ft.Page):
    """Это основная функция, которая запускает приложение.
    Она принимает один параметр: page (страница, на которой отображается приложение)."""
    interface = MainWindow(page)
    interface.main_window_page()


if __name__ == "__main__":
    ft.app(main)
