def solve_n_queens(n):
    def backtrack(row,cols,diagnol1,diagnol2,board):
        if row ==n:
            for r in board:
                print(" ".join(r))
            return True

        for col in range(n):
            if not (cols & (1 << col)) and not (diagnol1 &(1 << (row + col))) and not (diagnol2 &(1 << (row - col + n -1))):
                board[row][col] = 'Q'
                cols |= 1 << col
                diagnol1 |= 1 << (row + col)
                diagnol2 |= 1 << (row - col + n - 1)

                if backtrack(row +1,cols,diagnol1,diagnol2,board):
                    return True  

                board[row][col] = '.'
                cols &= ~(1 << col)
                diagnol1 &= ~(1 << (row + col))
                diagnol2 &= ~(1 << (row - col + n - 1))
        return False  

    board = [['.' for _ in range(n)] for _ in range(n)]
    if not backtrack(0,0,0,0,board):
        print("No solution for n =",n)
n = 8 
solve_n_queens(n)