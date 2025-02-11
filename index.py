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

def gaussian_elimination(A, B):
    """Розв'язання СЛАР методом Гауса"""
    try:
        augmented_matrix = np.hstack((A, B))  # Об'єднуємо матрицю A та вектор B
        n = len(B)
        
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
    # Матриця A розміру 3x3
    A = np.array([[-2, -1, -2], [4, -2, 1], [1, 3, -5]])
    print("Матриця A:")
    print(A, "\n")

    # Матриця B (вектор) розміру 3x1
    B = np.array([[1], [5], [3]])
    print("Матриця B:")
    print(B, "\n")
    
    # Обчислення оберненої матриці для A
    print("1. Обернена матриця A:")
    print(inverse_matrix(A))
    
    # Обчислення рангу матриці A
    print("\n2. Ранг матриці A:", matrix_rank(A))
    
    # Розв'язання системи лінійних рівнянь методом Гауса
    print("\n3. Розв'язок системи рівнянь методом Гауса:")
    print(gaussian_elimination(A, B))
