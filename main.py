import flet as ft

solution_history = []


def the_triangular_matrix(A, B, page):
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
        show_error_alert(page, "Матрица не квадратная, следовательно, не имеет четкого решения по методу Гаусса")
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
        show_error_alert(page, "Матрица является линейно зависимой и следовательно не является квадратной")
        return None, None

    return A, B


def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]  # Инициализация матрицы L нулями
    U = [[0.0] * n for _ in range(n)]  # Инициализация матрицы U нулями

    for i in range(n):
        L[i][i] = 1.0  # Установка единиц на диагонали матрицы L

    for k in range(n):
        U[k][k] = A[k][k]  # Присваивание элементу диагонали матрицы U соответствующего элемента исходной матрицы A
        for j in range(k + 1, n):
            L[j][k] = A[j][k] / U[k][k]  # Вычисление элементов нижнетреугольной матрицы L
            U[k][j] = A[k][j]  # Присваивание элементу верхнетреугольной матрицы U соответствующего элемента исходной матрицы A
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] -= L[i][k] * U[k][j]  # Обновление элементов исходной матрицы A

    return L, U  # Возвращение матриц L и U


def forward_substitution(L, b):
    n = len(L)
    y = [0.0] * n  # Инициализация вектора y нулями
    for i in range(n):
        y[i] = b[i]  # Присваивание элементу вектора y значения соответствующего элемента вектора b
        for j in range(i):
            y[i] -= L[i][j] * y[j]  # Прямая подстановка
    return y  # Возвращение вектора y


def backward_substitution(U, y):
    n = len(U)
    x = [0.0] * n  # Инициализация вектора x нулями
    for i in range(n - 1, -1, -1):
        x[i] = y[i]  # Присваивание элементу вектора x значения соответствующего элемента вектора y
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]  # Обратная подстановка
        x[i] /= U[i][i]  # Деление элемента вектора x на соответствующий диагональный элемент матрицы U
    return x  # Возвращение вектора x


def solve_lu(A, b):
    L, U = lu_decomposition(A)  # Вызов LU-разложения для матрицы A
    y = forward_substitution(L, b)  # Прямая подстановка для нахождения y
    x = backward_substitution(U, y)  # Обратная подстановка для нахождения x
    return x  # Возвращение решения системы уравнений


def solve_system(page, entries):
    if is_valid_input(entries):
        coefficients_matrix = []
        constants_vector = []
        for row in entries:
            row_coefficients = []
            for entry in row[:-1]:
                row_coefficients.append(float(entry.value))
            coefficients_matrix.append(row_coefficients)
            constants_vector.append(float(row[-1].value))
        print("Матрица коэффициентов:", coefficients_matrix)
        print("Вектор констант:", constants_vector)
        if page.method == 'gauss':
            A, B = the_triangular_matrix(coefficients_matrix, constants_vector, page)
            X = backward_substitution(A, B)
        else:
            X = solve_lu(coefficients_matrix, constants_vector)
        print("Решение системы уравнений:", X)
        for i in range(len(X)):
            X[i] = round(X[i], page.rounding)
        solution_history.append(X)
        show_solution_page(page, X, entries)
    else:
        show_invalid_input_alert(page)


def create_matrix_input_page(page: ft.Page, size: int, entries):
    """
    Создает страницу для ввода значений матрицы.

    Параметры:
    - page: объект страницы, на которой будет создан ввод
    - size: размер квадратной матрицы
    - entries: список полей для ввода значений матрицы
    """
    page.controls.clear()

    save_button = CustomButton("Продолжить", lambda e: solve_system(page, entries), page)
    back_button = CustomButton('Назад', lambda e: create_dimension_selection_page(page), page)
    clear_button = CustomButton('Очистить', lambda e: clear_matrix(page, entries), page)

    page.add(ft.Row([ft.Text("Введите значения матрицы",
                             size=30,
                             color='black' if page.theme_mode == 'light' else 'purple')],
                    alignment=ft.MainAxisAlignment.CENTER))
    for row in entries:
        page.add(ft.Row(row, alignment=ft.MainAxisAlignment.CENTER))  # Выравнивание по центру
    save_button.enabled = False  # Блокируем кнопку "Сохранить" при открытии страницы
    page.add(ft.Row([back_button, clear_button, save_button],
                    alignment=ft.MainAxisAlignment.CENTER))  # Выравнивание кнопки по центру
    page.update()


def clear_matrix(page, entries):
    """Эта функция очищает введенные значения матрицы. Она принимает один параметр: entries (список полей для ввода значений матрицы)."""
    for row in entries:
        for entry in row:
            entry.value = ""
    page.update()


def create_dimension_selection_page(page: ft.Page):
    """
    Создает страницу для выбора размера матрицы.

    Параметры:
    - page: объект страницы, на которой будет создан выбор размера
    """
    page.controls.clear()

    size_entry = ft.TextField(hint_text="Размер квадратной матрицы",
                              hint_style=ft.TextStyle(color='black' if page.theme_mode == 'light' else 'yellow'),
                              text_align=ft.TextAlign.CENTER,
                              color='black' if page.theme_mode == 'light' else 'yellow',
                              width=600,
                              text_size=30)
    submit_button = CustomButton("Подтвердить",
                                 lambda e: validate_and_create_matrix_input_page(page, size_entry.value),
                                 page,
                                 width=250,
                                 height=50)
    setting_button = CustomButton('Настройки',
                                  lambda e: create_settings_page(page),
                                  page,
                                  width=250,
                                  height=50,
                                  )

    page.add(ft.Column([ft.Row([size_entry],
                               alignment=ft.MainAxisAlignment.CENTER),
                        ft.Row([submit_button],
                               alignment=ft.MainAxisAlignment.CENTER),
                        ft.Row([setting_button],
                               alignment=ft.MainAxisAlignment.CENTER)],
                       alignment=ft.MainAxisAlignment.CENTER))
    page.update()


def validate_and_create_matrix_input_page(page, size_value):
    """Эта функция проверяет введенный размер матрицы и создает страницу для ввода значений матрицы.
    Она принимает два параметра:
    page (страница, на которой будет создан ввод) и
    size_value (введенное значение размера матрицы)."""
    try:
        size = int(size_value)
        if (2 <= size <= 5) == 0:
            show_error_alert(page, "Размер матрицы должен быть 2 <= и <= 5")
        else:
            create_matrix_input_page(page, size, create_entries(size, page))
    except ValueError:
        show_error_alert(page, "Пожалуйста, введите числовое значение для размера матрицы")


def create_entries(size, page):
    return [[ft.TextField(value="",
                          hint_text="0",
                          hint_style=ft.TextStyle(color='black' if page.theme_mode == 'light' else 'green'),
                          color='black' if page.theme_mode == 'light' else 'yellow',
                          text_align=ft.TextAlign.CENTER,
                          width=180,
                          text_size=50)
             for _ in range(size + 1)]
            for _ in range(size)]


def is_valid_input(entries):
    for row in entries:
        for entry in row:
            try:
                float(entry.value)
            except ValueError:
                return False
    return True


def show_invalid_input_alert(page):
    alert_message = "Пожалуйста, введите только числовые значения."
    alert_dialog = ft.AlertDialog(
        title=ft.Text("Ошибка"),
        content=ft.Text(alert_message),
        actions=[ft.TextButton("OK", on_click=lambda e: close_dialog(page))],
        actions_alignment=ft.MainAxisAlignment.END,
    )
    page.dialog = alert_dialog
    alert_dialog.open = True
    page.update()


def close_dialog(page):
    page.dialog.open = False
    page.update()


def show_solution_page(page: ft.Page, solution, entries):
    """эта функция отображает страницу с решением системы уравнений. Она принимает три параметра:
    page (страница, на которой будет отображено решение),
    solution (решение системы уравнений) и
    entries (список полей для ввода значений матрицы)."""
    page.controls.clear()

    answer = ""
    for index, item in enumerate(solution, start=1):
        item_str = str(item)
        if index < len(solution):
            answer += f"x{index} = {item_str:{len(item_str) + 4}}\n"
        else:
            answer += f"x{index} = {item_str}"

    answer = ft.Text(answer,
                     color='black' if page.theme_mode == 'light' else 'green',
                     size=30)

    back_button = CustomButton("Назад",
                               lambda e: create_matrix_input_page(page, len(entries), entries),
                               page)

    exit_button = CustomButton("Выход",
                               lambda e: page.window_close(),
                               page)

    restart_button = CustomButton("В начало",
                                  lambda e: create_dimension_selection_page(page),
                                  page,
                                  width=150)
    history_button = CustomButton("История решений",
                                  lambda e: show_history_page(page, entries),
                                  page,
                                  width=150)

    page.add(ft.Column([
         ft.Row([ft.Text('Решение системы уравнений:',
                         color='black' if page.theme_mode == 'light' else 'purple',
                         size=35)],
                alignment=ft.MainAxisAlignment.CENTER),
         ft.Row([answer],
                alignment=ft.MainAxisAlignment.CENTER),
         ft.Row([back_button, restart_button, history_button],
                alignment=ft.MainAxisAlignment.CENTER),
         ft.Row([exit_button],
                alignment=ft.MainAxisAlignment.CENTER)],
         alignment=ft.MainAxisAlignment.CENTER))

    page.update()


def show_history_page(page: ft.Page, entries):
    # Очищаем страницу
    page.controls.clear()
    for i, sol in enumerate(solution_history, start=1):
        solution_text = ft.Row([ft.Text(f"Решение {i}: {sol}",
                                        color='black' if page.theme_mode == 'light' else 'green',
                                        size=35)],
                               alignment=ft.MainAxisAlignment.CENTER)
        page.add(solution_text)

        # Добавляем кнопку для возврата на предыдущую страницу
    back_button = CustomButton("Назад",
                               lambda e: create_matrix_input_page(page, len(entries), entries),
                               page)
    page.add(ft.Row([back_button], alignment=ft.MainAxisAlignment.CENTER))

    page.update()


def show_error_alert(page, message):
    """Эта функция отображает сообщение об ошибке. Она принимает два параметра:
       page (страница, на которой отображается сообщение об ошибке)
       и message (текст сообщения об ошибке)."""
    alert_dialog = ft.AlertDialog(
        title=ft.Text("Ошибка"),
        content=ft.Text(message),
        actions=[ft.TextButton("OK", on_click=lambda e: close_dialog(page))],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    page.dialog = alert_dialog
    alert_dialog.open = True
    page.update()


def create_settings_page(page: ft.Page):
    page.controls.clear()

    page.add(ft.Text('\n\n\n\n\n\n\n\n\n\n\n'))

    color_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option('Blue'),
            ft.dropdown.Option('Red'),
            ft.dropdown.Option('Green'),
            ft.dropdown.Option('Black')
        ],
        hint_text='Выберите цвет текста',
        on_change=lambda e: change_color(page, color_dropdown.value)
    )

    full_screen_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option('Нет'),
            ft.dropdown.Option('Да')
        ],
        hint_text='Полноэкранный режим',
        on_change=lambda e: change_full_screen_mode(page, full_screen_dropdown.value)
    )

    theme_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option('Light'),
            ft.dropdown.Option('dark')
        ],
        hint_text='Выберите тему',
        on_change=lambda e: change_theme(page, theme_dropdown.value)
    )

    rounding_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option('1'),
            ft.dropdown.Option('2'),
            ft.dropdown.Option('3'),
            ft.dropdown.Option('4'),
            ft.dropdown.Option('5')
        ],
        hint_text='Округлять до',
        on_change=lambda e: change_rounding(page, rounding_dropdown.value)
    )

    method_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option('Гаусс'),
            ft.dropdown.Option('LU')
        ],
        hint_text='Выберите метод решения',
        on_change=lambda e: change_method(page, method_dropdown.value)
    )

    back_button = CustomButton("Назад", lambda e: create_dimension_selection_page(page), page)

    page.add(ft.Column([ft.Row([theme_dropdown, full_screen_dropdown],
                               alignment=ft.MainAxisAlignment.CENTER),
                       ft.Row([rounding_dropdown, method_dropdown],
                              alignment=ft.MainAxisAlignment.CENTER),
                       ft.Row([back_button],
                              alignment=ft.MainAxisAlignment.CENTER)],
                       alignment=ft.MainAxisAlignment.CENTER))

    page.update()


def change_full_screen_mode(page: ft.Page, mode: str):
    if mode == 'Да':
        page.window_full_screen = True
    else:
        page.window_full_screen = False
    page.update()


def change_color(page: ft.Page, color: str):
    page.text_color = color.lower()
    page.update()


def change_theme(page: ft.Page, theme: str):
    if theme.lower() == 'light':
        page.theme_mode = 'light'
    else:
        page.theme_mode = 'dark'
    page.update()


def change_rounding(page: ft.Page, rounding: str):
    page.rounding = int(rounding)
    page.update()


def change_method(page: ft.Page, method: str):
    if method.lower() == 'гаусс':
        page.method = 'gauss'
    else:
        page.method = 'lu'
    page.update()


class CustomButton(ft.TextButton):
    def __init__(self, text, on_click_action, page, width=150, height=50):
        super().__init__(
            text,
            width=width,
            height=height,
            on_click=on_click_action,
            style=ft.ButtonStyle(
                color={ft.MaterialState.DEFAULT: ft.colors.WHITE if page.theme_mode == 'light' else ft.colors.BLACK},
                bgcolor={ft.MaterialState.HOVERED: ft.colors.GREEN if page.theme_mode == 'light' else ft.colors.GREEN,
                         "": ft.colors.BLACK if page.theme_mode == 'light' else ft.colors.PURPLE},
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
    page.window_maximized = True
    page.rounding = 3
    page.method = 'gauss'
    solution_history = []
    create_dimension_selection_page(page)


if __name__ == "__main__":
    ft.app(main)

