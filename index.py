import numpy as np

def inverse_matrix(A):
    """Обчислення оберненої матриці"""
    try:
        inv_A = np.linalg.inv(A)
        return inv_A
    except np.linalg.LinAlgError:
        return "Матриця вироджена, не має оберненої"

def matrix_rank(A):
    """Обчислення рангу матриці"""
    return np.linalg.matrix_rank(A)

def gaussian_elimination(A, b):
    """Розв'язання СЛАР методом Гауса"""
    try:
        augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
        n = len(b)
        
        for i in range(n):
            # Пошук максимального елемента у стовпці i
            max_row = np.argmax(abs(augmented_matrix[i:, i])) + i
            augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]
            
            # Приведення до одиничного елемента
            augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]
            
            # Приведення нижче розташованих рядків до нуля
            for j in range(i + 1, n):
                augmented_matrix[j] -= augmented_matrix[j, i] * augmented_matrix[i]
        
        # Зворотний хід
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = augmented_matrix[i, -1] - np.sum(augmented_matrix[i, i + 1:n] * x[i + 1:n])
        
        return x
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    # Перша матриця 2x2
    A = np.array([[2, 1], [5, 3]])
    print("Матриця A:")
    print(A, "\n")

    # Обробка введення другої матриці розміром 2x1
    n = A.shape[0]
    while True:
        try:
            print("Введіть вектор B (через пробіл, довжина має бути", n, "):")
            b_input = input().split()
            if len(b_input) != n:
                raise ValueError("Невірна кількість елементів у векторі B")
            b = np.array(list(map(float, b_input)))
            break
        except ValueError as e:
            print("Помилка вводу:", e, "Спробуйте ще раз.")
    
    # Втконуємо пошук оберненої матриці
    print("\n-- Обернена матриця --")
    print(inverse_matrix(A))
    
    # Втконуємо обчислення рангу матриці
    print("\nРанг матриці:", matrix_rank(A))
    
    # Втконуємо розв'язання СЛАР методом Гауса
    print("\n-- Розв'язок методом Гауса --")
    print(gaussian_elimination(A, b))