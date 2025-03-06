import numpy as np

def inverse_matrix(A):
    """Обчислення оберненої матриці за методом Гаусса (ЗЖВ) згідно блок-схемі"""
    n = len(A)
    A = A.astype(float)
    C = np.eye(n)  # Створюємо одиничну матрицю для оберненої матриці
    protocol = ["\n\nПротокол обчислення оберненої матриці методом Гаусса:"]

    # Формування розширеної матриці (A|I)
    augmented_matrix = np.hstack((A, C))
    protocol.append("Розширена матриця:")
    protocol.append("\n".join([" ".join(map(str, row)) for row in augmented_matrix]))

    # Прямий хід (метод Гаусса)
    for i in range(n):
        protocol.append(f"\nКрок #{i+1}")
        protocol.append(f"Розв’язувальний елемент: A[{i+1}, {i+1}] = {augmented_matrix[i, i]:.2f}")
        augmented_matrix[:, :n], augmented_matrix[:, n:] = gaussian_elimination_step(augmented_matrix[:, :n], augmented_matrix[:, n:], i)
        protocol.append("Матриця після виконання ЗЖВ:")
        protocol.append("\n".join([" ".join([f"{x:.2f}" for x in row]) for row in augmented_matrix]))

    # Перевірка на виродженість
    if np.any(np.abs(np.diag(augmented_matrix[:, :n])) < 1e-12):
        return "Матриця вироджена, не має оберненої", "\n".join(protocol)

    inv_A = augmented_matrix[:, n:]
    protocol.append("Обернена матриця:")
    protocol.append("\n".join([" ".join([f"{x:.2f}" for x in row]) for row in inv_A]))
    return inv_A, "\n".join(protocol)

def gaussian_elimination_step(A, B, i):
    """Процедура ЗЖВ для обнулення елементів"""
    n = len(A)
    pivot = A[i, i]
    if np.abs(pivot) < 1e-12:
        raise ValueError("Матриця вироджена, не має оберненої")
    
    A[i] = A[i] / pivot
    B[i] = B[i] / pivot
    
    for j in range(n):
        if i != j:
            factor = A[j, i]
            A[j] -= factor * A[i]
            B[j] -= factor * B[i]
    return A, B

def matrix_rank(A):
    """Обчислює ранг матриці за допомогою ступінчастого вигляду"""
    A = A.astype(float)
    n, m = A.shape
    protocol = ["\n\nПротокол обчислення рангу матриці методом Гаусса:"]
    
    for i in range(min(n, m)):
        protocol.append(f"\nКрок #{i+1}")
        A, _ = gaussian_elimination_step(A, np.zeros(n), i)
        protocol.append("Матриця після виконання ЗЖВ:")
        protocol.append("\n".join([" ".join([f"{x:.2f}" for x in row]) for row in A]))
    
    rank = np.sum(np.any(np.abs(A) > 1e-12, axis=1))
    protocol.append(f"Ранг матриці: {rank}")
    return rank, "\n".join(protocol)

def gaussian_elimination_solve(A, B):
    """Розв'язання СЛАР методом Гауса з протоколом обчислень"""
    try:
        A = A.astype(float)
        B = B.astype(float).flatten()
        n = len(B)
        protocol = ["\n\nПротокол обчислення розв’язків СЛАУ методом Гаусса:"]

        for i in range(n):
            protocol.append(f"\nКрок #{i+1}")
            protocol.append(f"Розв’язувальний елемент: A[{i+1}, {i+1}] = {A[i, i]:.2f}")
            A, B = gaussian_elimination_step(A, B.reshape(-1, 1), i)
            protocol.append("Матриця після виконання ЗЖВ:")
            protocol.append("\n".join([" ".join([f"{A[j, k]:.2f}" for k in range(n)]) for j in range(n)]))

        x = np.zeros(n)
        protocol.append("Обчислення розв’язків:")
        for i in range(n - 1, -1, -1):
            x[i] = (B[i][0] - np.sum(A[i, i + 1:] * x[i + 1:])) / A[i, i]
            protocol.append(f"X[{i+1}] = {x[i]:.2f}")

        return x, "\n".join(protocol)
    except Exception as e:
        return str(e), "Помилка в обчисленнях"

if __name__ == "__main__":
    A = np.array([[-2, -1, -2], [4, -2, 1], [1, 3, -5]])
    B = np.array([1, 5, 3])
    
    print("1. Обернена матриця A:")
    inv_A, inv_protocol = inverse_matrix(A)
    print(inv_A)

    rank, rank_protocol = matrix_rank(A)
    print(f"\n2. Ранг матриці A: {rank}")

    print("\n3. Знаходження розв’язків СЛАУ методом Гаусса:")
    solution, protocol = gaussian_elimination_solve(A, B)
    print(f"Розв'язок: {solution}")

    # Виведення протоколів обчислення
    print(rank_protocol)
    print(inv_protocol)
    print(protocol)
