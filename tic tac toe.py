import random
board = [
  ["-", "-", "-"],
  ["-", "-", "-"],
  ["-", "-", "-"]
]
def create_board(board):
  for row in board:
    for slot in row:
      print(slot, end = "")
    print()
create_board(board)

def players():
  players = ["X","O"]
  turn = 0
  while not is_board_full(board):
        create_board(board)
        if turn == 0:
            print("Người chơi 1!")
            row, col = play(board)
            board[row][col] = players[turn]
        else:
            print("Người chơi 2!")
            row, col = play(board)
            board[row][col] = players[turn]
            
        if check_winner(board):
            create_board(board)
            print("Người chơi 1 thắng!" if turn == 0 else "Người chơi 2 thắng!")
            break
            
        turn = 1- turn
        
    else:
        create_board(board)
        
            
              
def whoGoesFirst():
  if random.randint(0, 1) == 0:
    return 'player 1'
  else:
    return 'player 2'

  
  # check row
 def check_row(board, row):
    return (board[row][0] == board[row][1] and board[row][1] == board[row][2] and board[row][0] != " ")
  # check col
 def check_column(board, col):
    return (board[0][col] == board[1][col] and board[1][col] == board[2][col] and board[0][col] != " ")
  #check diag
  def check_diagonals(board):
    return (board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[0][0] != " ") or\
            (board[2][0] == board[1][1] and board[1][1] == board[0][2] and board[2][0] != " ")
            
  def check_winner(board):
    for i in range(3):
        if check_row(board, i):
            return True
        if check_column(board, i):
            return True
    if check_diagonals(board):
        return True
    return False
    
   def play(board): #chọn ô đúng
    while True: 
        row = input("Số dòng: ")
        while not row.isdigit() or int(row) < 1 or int(row) > 3:
            row = input("Dòng từ 1-3: ")
        row = int(row)
        col = input("Số cột: ")
        while not col.isdigit() or int(col) < 1 or int(col) > 3:
            col = input("Cột từ 1-3: ")
        col = int(col)
        if board[row-1][col-1] != " ":
            print("Chọn ô khác!")
        else:
            return (row-1, col-1)
     
   
   
   
   
   
