def inputOneMatrix():
    # first matrix
    print("Enter size of matrix:")
    sizes = input()
    sizes = sizes.split()
    n1 = int(sizes[0])
    m1 = int(sizes[1])
    matrix = []
    n = range(n1)
    print("Enter matrix:")
    for i in range(n1):
        matrix.append([float(x) for x in input().split()])
    return (matrix, n1, m1)


def addMatrix(n1, n2, m1, m2, matrix1, matrix2):
    sumMatrix = list()
    if (n1 != n2 or m1 != m2):
        print("ERROR")
    else:
        print("The result is:")
        for n in range(n1):
            sumMatrix.append([])
            for m in range(m2):
                sumMatrix[n].append(matrix2[n][m] + matrix1[n][m])
    return sumMatrix


def multByScalar(n1, m1, matrix, scalar):
    new_matrix = list()
    print("The result is:")
    for n in range(n1):
        new_matrix.append([])
        for m in range(m1):
            new_matrix[n].append(scalar * matrix[n][m])

    return new_matrix


def matrixMultplication(n1, n2, m1, m2, matrix1, matrix2):
    # number of columns in the first matrix
    # equal the number of rows for the second matrix
    if m1 != n2:
        print("Error")
        return
    multipliedMatrix = []

    # row in multipliedMAtrix
    for n1_row in range(n1):
        multipliedMatrix.append([])

        for m2_col in range(m2):
            sum = 0
            for m1_colRow in range(m1):
                # print(str(matrix1[n1_row][m1_colRow] ))
                # print( str(matrix2[m1_colRow][m2_col]))
                multiplication = matrix1[n1_row][m1_colRow] * matrix2[m1_colRow][m2_col]
                sum += multiplication
            multipliedMatrix[n1_row].append(sum)

    return multipliedMatrix


def mainDiagonalTranspose(n, m, matrix):
    transposed = []
    for i in range(n):
        transposed.append([])
        for j in range(m):
            transposed[i].append(matrix[j][i])

    return transposed


def sideDiagonalTranspose(n, m, matrix):
    transposed = []
    rows = list(range(n))
    rows.reverse()
    col = list(range(m))
    col.reverse()
    counter_row = 0
    counter_col = 0

    for i in rows:
        transposed.append([])
        for j in col:
            transposed[counter_row].append(matrix[j][i])

        counter_row += 1
    return transposed


def verticalLineTranspose(n, m, matrix):
    transposed = []
    col = list(range(m))
    col.reverse()

    for i in range(n):
        transposed.append([])
        for j in col:
            transposed[i].append(matrix[i][j])
    return transposed


def horizontalLineTranspose(n, m, matrix):
    transposed = []
    row = list(range(n))
    row.reverse()
    counter = 0
    for i in row:
        transposed.append([])
        for j in range(m):
            transposed[counter].append(matrix[i][j])

        counter += 1
    return transposed


def getCofactor1(matrix, p, q, n):
    temp = []
    counter_i, counter_j = 0, 0
    flag = 0
    for row in range(n):
        flag = 0
        for col in range(n):
            # Copying into temporary matrix only those element
            # which are not in given row and column
            if row != p and col != q:
                if counter_j == 0:
                    temp.append([])
                temp[counter_i].append(matrix[row][col])
                counter_j += 1
                if counter_j == n - 1:
                    counter_i += 1
                # Row is filled, so increase row index and
                # reset col index

    return temp


# def determinant(matrix, n):
#     if (n == 1):
#         return matrix[0][0]
#     returnvalue = 0
#     for i in range(len(matrix)):
#         returnvalue += matrix[0][i] * cofactor(matrix, 0, i)
#     return returnvalue


def determinantOfMatrix(mat, n):
    temp = [0] * n  # temporary array for storing row
    total = 1
    det = 1  # initialize result

    # loop for traversing the diagonal elements
    for i in range(0, n):
        index = i  # initialize the index

        # finding the index which has non zero value
        while (mat[index][i] == 0 and index < n):
            index += 1

        if (index == n):  # if there is non zero element
            # the determinat of matrix as zero
            continue

        if (index != i):
            # loop for swaping the diagonal element row and index row
            for j in range(0, n):
                mat[index][j], mat[i][j] = mat[i][j], mat[index][j]

                # determinant sign changes when we shift rows
            # go through determinant properties
            det = det * int(pow(-1, index - i))

            # storing the values of diagonal row elements
        for j in range(0, n):
            temp[j] = mat[i][j]

            # traversing every row below the diagonal element
        for j in range(i + 1, n):
            num1 = temp[i]  # value of diagonal element
            num2 = mat[j][i]  # value of next row element

            # traversing every column of row
            # and multiplying to every row
            for k in range(0, n):
                # multiplying to make the diagonal
                # element and next row element equal

                mat[j][k] = (num1 * mat[j][k]) - (num2 * temp[k])

            total = total * num1  # Det(kA)=kDet(A);

    # mulitplying the diagonal elements to get determinant
    for i in range(0, n):
        det = det * mat[i][i]
    return float(det / total)  # Det(kA)/k=Det(A);


def minor(matrix, row, column):
    n = len(matrix)
    newMatrix = []
    counter = 0
    if n == 1:
        raise ValueError("Minor is not defined for 1x1 matrix")
    for i in range(n):
        for j in range(n):
            if i != row and j != column:
                if counter == 0:
                    newMatrix.append([])
                sd = matrix[i][j]
                newMatrix[counter].append(matrix[i][j])
                if len(newMatrix[counter]) == n-1:
                    counter += 1

    return determinant(newMatrix, len(newMatrix))


def cofactor(matrix, row, column):
    return ((-1) ** (row + column)) * minor(matrix, row, column)


def prinMatrix(matrix, n, m):
    for i in range(n):
        print(*matrix[i])
        print()


def cofactorMatrix(matrix, n):
    returnvalue = list()
    for i in range(n):
        returnvalue.append([])
        for j in range(n):
            returnvalue[i].append(cofactor(matrix, i, j, ))
    print("The cofactor matrix")
    prinMatrix(returnvalue, n, n)
    return returnvalue


def getCofactor(matrix, row, column):
    n = len(matrix)
    newMatrix = []
    counter = 0
    if n == 1:
        raise ValueError("Minor is not defined for 1x1 matrix")
    for i in range(n):
        for j in range(n):
            if i != row and j != column:
                if counter == 0:
                    newMatrix.append([])
                newMatrix[counter].append(matrix[i][j])
                if len(newMatrix[counter]) == n - 1:
                    counter += 1

    return newMatrix


def determinant(matrix, n):
    D = 0  # Initialize result

    #  Base case : if matrix contains single element
    if (n == 1):
        return matrix[0][0]

    temp = list()  # To store cofactors

    sign = 1  # To store sign multiplier

    # Iterate for each element of first row
    for f in range(n):
        # Getting Cofactor of matrix[0][f]
        temp = getCofactor(matrix, 0, f)
        D += sign * matrix[0][f] * determinant(temp, n - 1)

        # terms are to be added with alternate sign
        sign = -sign

    return D


# Function to get adjoint of A[N][N] in adj[N][N].
def adjoint(matrix, N):

    adj = [[ 0 for i in range(N)] for j in range(N)]
    if (N == 1):
        adj.append([])
        adj[0].append(1)
        return adj

    # temp is used to store cofactors of matrix[][]
    sign = 1

    for i in range(N):

        for j in range(N):
            # Get cofactor of A[i][j]
            temp = getCofactor(matrix, i, j)

            # sign of adj[j][i] positive if sum of row
            # and column indexes is even.
            if (i + j % 2) == 0:
                sign = 1
            else:
                sign = -1

            # Interchanging rows and columns to get the
            # transpose of the cofactor matrix
            adj[j][i] = (sign) * (determinant(temp, N - 1))

    print("Adjoint:")
    prinMatrix(matrix, N, N)

    return adj


def inverse(matrix1, n1):
    det = determinant(matrix1, n1)
    inverseDet = 1 / det
    adjugate = adjoint(matrix1, n1)
    inverse_matrix = multByScalar(n1, n1, adjugate, inverseDet)

    return inverse_matrix


def menu():
    print("1. Add matrices\n "
          "2. Multiply matrix by a constant\n"
          "3. Multiply matrices\n"
          "4. Transpose matrix\n"
          "5. Calculate a determinant\n"
          "6. Inverse matrix\n"
          "0. Exit\n"
          )
    choice = int(input())
    print("Your choice:" + str(choice))
    return choice


def transposeMenu():
    print("1. Main diagonal"
          "2. Side diagonal"
          "3. Vertical line"
          "4. Horizontal line")
    choice = int(input())
    print("Your choice:" + str(choice))
    return choice


def main():
    choice = menu()
    while choice != 0:
        if choice == 1:
            data = inputOneMatrix()
            matrix1 = data[0]
            n1 = data[1]
            m1 = data[2]
            data = inputOneMatrix()
            matrix2 = data[0]
            n2 = data[1]
            m2 = data[2]
            prinMatrix(addMatrix(n1, n2, m1, m2, matrix1, matrix2), n1, m1)
        elif choice == 2:
            data = inputOneMatrix()
            matrix1 = data[0]
            n1 = data[1]
            m1 = data[2]
            print("Enter constant: ")
            scalar = int(input())
            new_matrix = multByScalar(n1, m1, matrix1, scalar)
            prinMatrix(new_matrix, n1, m1)

        elif choice == 3:
            data = inputOneMatrix()
            matrix1 = data[0]
            n1 = data[1]
            m1 = data[2]
            data = inputOneMatrix()
            matrix2 = data[0]
            n2 = data[1]
            m2 = data[2]

            # n1, m1  = 4, 4
            # matrix1 = [
            #     [ 1, 2 ,2, 7],
            #     [3, 3, 4, 5],
            #     [5 ,0, 0, 1],
            #     [0 ,1, 0, 8]
            # ]
            # n2, m2 = 4, 4
            # matrix2 = [
            #     [9 ,8 ,7, 13],
            #     [15, 14, 0, 1],
            #     [3, 7, 2, 3],
            #     [0, 9, 0 ,35]
            #     ]

            new_matrix = matrixMultplication(n1, n2, m1, m2, matrix1, matrix2)
            prinMatrix(new_matrix, n1, m2)

        elif choice == 4:
            transposeType = transposeMenu()
            data = inputOneMatrix()
            matrix = data[0]
            n = data[1]
            m = data[2]
            new_matrix = list
            if transposeType == 1:
                new_matrix = mainDiagonalTranspose(n, m, matrix)
            elif transposeType == 2:
                new_matrix = sideDiagonalTranspose(n, m, matrix)
            elif transposeType == 3:
                new_matrix = verticalLineTranspose(n, m, matrix)
            elif transposeType == 4:
                new_matrix = horizontalLineTranspose(n, m, matrix)
            prinMatrix(new_matrix, n, m)


        elif choice == 5:
            data = inputOneMatrix()
            matrix1 = data[0]
            n1 = data[1]
            m1 = data[2]
            if n1 == m1:
                determ = determinantOfMatrix(matrix1, n1)
                print("The result is: ")
                print(determ)


        elif choice == 6:
            data = inputOneMatrix()
            matrix1 = data[0]
            n1 = data[1]
            m1 = data[2]
            # n1, m1 = 3, 3
            # matrix1 = [[1, 3, 1], [3, 1, 1], [2, 1, 2]]
            if n1 == m1:
                if determinantOfMatrix(matrix1, n1) == 0:
                    print("This matrix doesn't have an inverse.")
                else:
                    new_matrix = inverse(matrix1, n1)
                    prinMatrix(new_matrix, n1, m1)

        choice = menu()


if __name__ == '__main__':
    main()

# second matrix
# sizes = input().split()
# n2 = int(sizes[0])
# m2 = int(sizes[1])
# matrix2 = []
# for i in range(n2):
#     matrix2.append( [int(x) for x in input().split()]  )
