import copy
from typing import List, Tuple, Optional

# This code is written in collaboration with Github Copilot, where code suggestions are used when applicable
class DC3:
    def __init__(self):
        # initialize 4x5 grid with White = 0, Black = 1, Empty = white space
        self.grid = [
            [0, None, None, None, 1],
            [1, None, None, None, 0],
            [0, None, None, None, 1],
            [1, None, None, None, 0],
        ]
        self.current_player = 0 
        self.move_history = []

    def get_possible_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        # Valid moves for current player
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E
        for row in range(4):
            for col in range(5):
                if self.grid[row][col] == self.current_player:
                    for dr, dc in directions:
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < 4 and 0 <= new_col < 5 and self.grid[new_row][new_col] is None:
                            moves.append(((col, row), (new_col, new_row)))
        #print(f"Possible moves for player {self.current_player}: {moves}")  
        return moves
    
    def parse_move(self, move: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        # Parse move from string format to tuple, 1-indexed
        #if len(move) < 3 or len(move) > 3:
         #   print("Invalid move format")
          #  raise ValueError("Invalid move")
        
        x = int(move[0]) - 1
        y = int(move[1]) - 1
        direction = move[2].upper()

        if direction == 'N':
            return (x, y), (x, y - 1)
        elif direction == 'S':
            return (x, y), (x, y + 1)
        elif direction == 'W':
            return (x, y), (x - 1, y)
        elif direction == 'E':
            return (x, y), (x + 1, y)
        else:
            raise ValueError("Invalid direction")
        
    def check_win_state(self) -> Optional[int]:
        # Check if 3 consecutive pieces of the same colour exist
        for row in range(4):
            for col in range(5):
                if self.grid[row][col] is not None:
                    player = self.grid[row][col]

                    # horizontal check for 3 consecutive same colour pieces
                    if col <= 2 and (self.grid[row][col + 1] == player and self.grid[row][col + 2] == player ):
                        return player
                    
                    # vertical check for 3 consecutive same colour pieces
                    if row <= 1 and (self.grid[row + 1][col] == player and self.grid[row + 2][col] == player):
                        return player
                    
                    # diagonal check (down-right) for 3 consecutive same colour pieces
                    if row <= 1 and col <= 2 and (self.grid[row + 1][col + 1] == player and self.grid[row + 2][col + 2] == player):
                        return player
                    
                    # diagonal check (down-left) for 3 consecutive same colour pieces
                    if row <= 1 and col >= 2 and (self.grid[row + 1][col - 1] == player and self.grid[row + 2][col - 2] == player):
                        return player               
        return None

    def terminal_test(self) -> bool:
        # Check if win state or draw
        if self.check_win_state() is not None:
            return True

        # Check for draw (threefold repetition)
        current_position = (str(self.grid), self.current_player)
        if self.move_history.count(current_position) >= 3:
            print("Draw due to threefold repetition")
            return True

        return False
    
    def move(self, move: Tuple[Tuple[int, int], Tuple[int, int]]):
        # Execute move -> Parse string to tuples
        # Check if piece belongs to the current player
        # Check if target location move is legal
        # Execute move
        try:
            # If move is a string, parse it; if tuple, use directly
            if isinstance(move, str):
                (x, y), (dest_x, dest_y) = self.parse_move(move)
            else:
                (x, y), (dest_x, dest_y) = move

            # Check if initial position has current player's piece
            if self.grid[y][x] != self.current_player:
                raise ValueError("Piece does not belong to current player")

            # Check if move is legal
            possible_moves = self.get_possible_moves()
            if ((x, y), (dest_x, dest_y)) not in possible_moves:
                raise ValueError("Illegal move")

            # Execute move
            new_state = copy.deepcopy(self)

            piece =  new_state.grid[y][x]
            new_state.grid[y][x] = None
            new_state.grid[dest_y][dest_x] = piece

            if new_state.current_player == 0:
                new_state.current_player = 1
            else:
                new_state.current_player = 0

            new_state.move_history = self.move_history + [(str(new_state.grid), new_state.current_player)]
            return new_state

        except ValueError as e:
            print("Invalid move")
            raise

    def utility(self) -> int:
        winner = self.check_win_state()
        if winner == 0:
            return float('inf')
        elif winner == 1:
            return float('-inf')
        else:
            return 0

class DC3Agent:

    def __init__(self, depth: int):
        self.depth = depth
        self.nodes_traveled = 0

    # Minimax decision is written utilizing pseudocode from AIMA textbook
    def minimax_decision(self, state: DC3) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        self.nodes_traveled = 0
        best_score = float('-inf')
        best_move = None

        for move in state.get_possible_moves():
            new_state = state.move(move)
            score = self.min_value(new_state, self.depth - 1)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def max_value(self, state: DC3, depth: int) -> int:
        self.nodes_traveled += 1

        if state.terminal_test():
            return state.utility()
        if depth == 0:
            return self.heuristic(state)

        best_score = float('-inf')
        for move in state.get_possible_moves():
            new_state = state.move(move)
            score = self.min_value(new_state, depth - 1)
            best_score = max(best_score, score)

        return best_score

    def min_value(self, state: DC3, depth: int) -> int:
        self.nodes_traveled += 1

        if state.terminal_test():
            return state.utility()
        if depth == 0:
            return self.heuristic(state)

        best_score = float('inf')
        for move in state.get_possible_moves():
            new_state = state.move(move)
            score = self.max_value(new_state, depth - 1)
            best_score = min(best_score, score)

        return best_score
    
    def heuristic(self, state: DC3) -> int:
        # Simple heuristic: count number of 2-in-a-rows for each player
        score = 0
        for row in range(4):
            for col in range(5):
                if state.grid[row][col] is not None:
                    player = state.grid[row][col]

                    # horizontal check for 2 consecutive same colour pieces
                    if col <= 3 and (state.grid[row][col + 1] == player):
                        score += 1 if player == 0 else -1
                    
                    # vertical check for 2 consecutive same colour pieces
                    if row <= 2 and (state.grid[row + 1][col] == player):
                        score += 1 if player == 0 else -1
                    
                    # diagonal check (down-right) for 2 consecutive same colour pieces
                    if row <= 2 and col <= 3 and (state.grid[row + 1][col + 1] == player):
                        score += 1 if player == 0 else -1
                    
                    # diagonal check (down-left) for 2 consecutive same colour pieces
                    if row <= 2 and col >= 1 and (state.grid[row + 1][col - 1] == player):
                        score += 1 if player == 0 else -1               
        return score
    

def draw_board(state: DC3):
    print("  1 2 3 4 5")  # Column numbers
    for row in range(4):
        row_str = f"{row + 1} "  # Row number
        for col in range(5):
            if state.grid[row][col] is not None:
                row_str += str(state.grid[row][col]) + ','
            else:
                row_str += ' ,'
        print(row_str)
    print()

def play_game():
    game = DC3()
    agent = DC3Agent(depth=3)

    print("Do you want to play first? (y/n): ")
    first = input().strip().lower() == 'y'

    while not game.terminal_test():
        draw_board(game)
        if (game.current_player == 0 and first) or (game.current_player == 1 and not first):
            print("Your turn. Enter move (e.g., 23N for column 2, row 3, North): ")
            move = input().strip().upper()
            try:
                game = game.move(move)
            except ValueError:
                print("Invalid move. Try again.")
                continue
            if game.terminal_test():
                draw_board(game)
                winner = game.check_win_state()
                if winner == 0:
                    print("White wins!")
                elif winner == 1:
                    print("Black wins!")
                break
        else:
            print("Agent's turn...")
            move = agent.minimax_decision(game)
            print(f"Agent chose move: {move}")
            game = game.move(move)
            if game.terminal_test():
                draw_board(game)
                winner = game.check_win_state()
                if winner == 0:
                    print("White wins!")
                elif winner == 1:
                    print("Black wins!")
                break

if __name__ == "__main__":
    play_game()
