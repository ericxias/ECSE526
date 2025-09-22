import socket
import sys
import copy
from typing import List, Tuple, Optional

# Game Logic
class DC3:
    def __init__(self):
        # initialize 4x5 grid with White = 0, Black = 1, None = white space
        self.grid = [
            [0, None, None, None, 1],
            [1, None, None, None, 0],
            [0, None, None, None, 1],
            [1, None, None, None, 0],
        ]
        self.current_player = 0 
        self.grid_history = []

    def get_possible_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        # Valid moves for current player
        moves = []
        # N, S, E, W
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  
        for row in range(4):
            for col in range(5):
                if self.grid[row][col] == self.current_player:
                    for dr, dc in directions:
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < 4 and 0 <= new_col < 5 and self.grid[new_row][new_col] is None:
                            moves.append(((col, row), (new_col, new_row)))
        return moves
    
    def parse_move(self, move: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        # Parse move from string format to tuple, 1-indexed (mainly for testing/debugging)
        if len(move) < 3 or len(move) > 3:
            raise ValueError("Invalid move")
        
        x = int(move[0]) - 1
        y = int(move[1]) - 1
        direction = move[2].upper()

        if direction == 'N':
            return (x, y), (x, y - 1)
        elif direction == 'S':
            return (x, y), (x, y + 1)
        elif direction == 'E':
            return (x, y), (x + 1, y)
        elif direction == 'W':
            return (x, y), (x - 1, y)
        else:
            raise ValueError("Invalid direction")
        
    '''
    def draw_board(self):
        # Column numbers
        print("  1 2 3 4 5")  
        for row in range(4):
            # Row number
            row_str = []  
            for col in range(5):
                if self.grid[row][col] is not None:
                    row_str.append(str(self.grid[row][col]))
                else:
                    row_str.append(' ')
            print(f"{row + 1} " + ','.join(row_str))
        print()'''
        
    def check_win_state(self) -> Optional[int]:
        # Check if 3 consecutive pieces of the same colour exist
        for row in range(4):
            for col in range(5):
                if self.grid[row][col] is not None:
                    player = self.grid[row][col]

                    # horizontal check for 3 consecutive same colour pieces
                    if col <= 2 and (self.grid[row][col + 1] == player and self.grid[row][col + 2] == player ):
                        #print(f"Horizontal win detected found at ({row}, {col})")
                        #self.draw_board()
                        return player
                    
                    # vertical check for 3 consecutive same colour pieces
                    if row <= 1 and (self.grid[row + 1][col] == player and self.grid[row + 2][col] == player):
                        #print(f"Vertical win detected found at ({row}, {col})")
                        #self.draw_board()
                        return player
                    
                    # diagonal check (down-right) for 3 consecutive same colour pieces
                    if row <= 1 and col <= 2 and (self.grid[row + 1][col + 1] == player and self.grid[row + 2][col + 2] == player):
                        #print(f"Diagonal (down-right) win detected found at ({row}, {col})")
                        #self.draw_board()
                        return player
                    
                    # diagonal check (down-left) for 3 consecutive same colour pieces
                    if row <= 1 and col >= 2 and (self.grid[row + 1][col - 1] == player and self.grid[row + 2][col - 2] == player):
                        #print(f"Diagonal (down-left) win detected found at ({row}, {col})")
                        #self.draw_board()
                        return player
        return None

    def terminal_test(self) -> bool:
        if self.check_win_state() is not None:
            return True

        # Check for draw (threefold repetition)
        #if self.grid_history.count(str(self.grid)) >= 3:
        #    return True

        return False
    
    def move(self, move):
        # Execute move -> Parse string to tuples
        # Check if piece belongs to the current player
        # Check if target location move is legal
        # Execute move
        try:
            # If move is a string, parse it (used in local debugging); if tuple, use directly
            if isinstance(move, str):
                (col, row), (to_col, to_row) = self.parse_move(move)
            else:
                (col, row), (to_col, to_row) = move

            # Check if initial position has current player's piece
            if self.grid[row][col] != self.current_player:
                raise ValueError("Piece does not belong to current player")

            # Check if move is legal
            possible_moves = self.get_possible_moves()
            if ((col, row), (to_col, to_row)) not in possible_moves:
                raise ValueError("Illegal move")

            # Execute move
            new_state = copy.deepcopy(self)

            piece =  new_state.grid[row][col]
            new_state.grid[row][col] = None
            new_state.grid[to_row][to_col] = piece

            if new_state.current_player == 0:
                new_state.current_player = 1
            else:
                new_state.current_player = 0

            new_state.grid_history = self.grid_history + [str(new_state.grid)]
            return new_state

        except ValueError as e:
            raise

    def utility(self) -> int:
        winner = self.check_win_state()
        if winner == 0:
            return 10000
        elif winner == 1:
            return -10000
        else:
            return 0

# AI Agent with Alpha-Beta Pruning
class DC3Agent:

    def __init__(self, depth: int):
        self.depth = depth
        # Transposition table to store evaluated states
        self.trans_table = {}

    # Alpha Beta pruning is written utilizing pseudocode from AIMA textbook
    def alpha_beta(self, state: DC3) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        v = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        # Track the depth used to reach the score, lower depths are preferred
        best_depth = float('inf')

        for move in state.get_possible_moves():
            new_state = state.move(move)
            score, score_depth = self.min_value(new_state, self.depth - 1, alpha, beta)
            move_str = self.move_to_string(move) #debug
            print(f"Evaluating Move: {move_str} with value {score} and depth used {score_depth}") #debug

            if score > v or (score == v and score_depth < best_depth):
                v = score
                best_move = move
                best_depth = score_depth
                print(f"New Best Move: {move_str} with value {score} and depth used {score_depth}") #debug


            alpha = max(alpha, v)
        print(f"Best Move: {self.move_to_string(best_move)} with value {v}") #debug
        return best_move

    def max_value(self, state: DC3, depth: int, alpha: float, beta: float) -> Tuple[int, int]:
        if state.terminal_test():
            return state.utility(), self.depth - depth
        if depth == 0:
            return self.heuristic(state), self.depth

        v = float('-inf')
        best_depth = float('inf')

        for move in state.get_possible_moves():
            new_state = state.move(move)
            score, score_depth = self.min_value(new_state, depth - 1, alpha, beta)
            # lower depth is kept -> allows for quicker wins (i.e. win in next move)
            if score > v or (score == v and score_depth < best_depth):
                v = score
                best_depth = score_depth
            if v >= beta:
                return v, best_depth
            alpha = max(alpha, v)

        return v, best_depth

    def min_value(self, state: DC3, depth: int, alpha: float, beta: float) -> Tuple[int, int]:
        if state.terminal_test():
            return state.utility(), self.depth - depth
        if depth == 0:
            return self.heuristic(state), self.depth

        v = float('inf')
        best_depth = float('inf')

        for move in state.get_possible_moves():
            new_state = state.move(move)
            score, score_depth = self.max_value(new_state, depth - 1, alpha, beta)
            # lower depth is kept -> allows for quicker wins (i.e. win in next move)
            if score < v or (score == v and score_depth < best_depth):
                v = score
                best_depth = score_depth
            if v <= alpha:
                return v, best_depth
            beta = min(beta, v)

        return v, best_depth

    def heuristic(self, state: DC3) -> int:
        # Check if state was evaluated before in transposition table
        position = str(state.grid) + str(state.current_player)
        if position in self.trans_table:
            return self.trans_table[position]
        
        # Evaluate heuristic score for the state and store
        score = self.calculate_heuristic(state)
        self.trans_table[position] = score
        return score

    def calculate_heuristic(self, state: DC3) -> int:
        score = 0
        # Check for 2-in-a-rows for each player that can become 3-in-a-row, higher weight when open on both ends
        for row in range(4):
            for col in range(5):
                if state.grid[row][col] is not None:
                    player = state.grid[row][col]

                    # horizontal check for 2 consecutive same colour pieces
                    if col <= 3 and (state.grid[row][col + 1] == player):
                        if (col > 0 and state.grid[row][col - 1] is None):
                            # Check closest distance to another piece of same colour
                            distance = self.closest_distance_piece(state, row, col - 1, player)
                            if player == 0:
                                score += max(0, 50 * (10 - distance))
                            else:
                                score -= max(0, 50 * (10 - distance))
                        elif col < 3 and state.grid[row][col + 2] is None:
                            distance = self.closest_distance_piece(state, row, col + 2, player)
                            if player == 0:
                                score += max(0, 50 * (10 - distance))
                            else:
                                score -= max(0, 50 * (10 - distance))
                        else:
                            if player == 0:
                                score += 50
                            else:
                                score -= 50
                    
                    # vertical check for 2 consecutive same colour pieces
                    if row <= 2 and (state.grid[row + 1][col] == player):
                        if row > 0 and state.grid[row - 1][col] is None:
                            distance = self.closest_distance_piece(state, row - 1, col, player)
                            if player == 0:
                                score += max(0, 50 * (10 - distance))
                            else:
                                score -= max(0, 50 * (10 - distance))
                        elif row < 2 and state.grid[row + 2][col] is None:
                            distance = self.closest_distance_piece(state, row +2, col, player)
                            if player == 0:
                                score += max(0, 50 * (10 - distance))
                            else:
                                score -= max(0, 50 * (10 - distance))
                        else:
                            if player == 0:
                                score += 50
                            else:
                                score -= 50

                    # diagonal check (down-right) for 2 consecutive same colour pieces
                    if row <= 2 and col <= 3 and (state.grid[row + 1][col + 1] == player):
                        if row > 0 and col > 0 and state.grid[row - 1][col - 1] is None:
                            distance = self.closest_distance_piece(state, row - 1, col - 1, player)
                            if player == 0:
                                score += max(0, 50 * (10 - distance))
                            else:
                                score -= max(0, 50 * (10 - distance))
                        elif row < 2 and col < 3 and state.grid[row + 2][col + 2] is None:
                            distance = self.closest_distance_piece(state, row + 2, col + 2, player)
                            if player == 0:
                                score += max(0, 50 * (10 - distance))
                            else:
                                score -= max(0, 50 * (10 - distance))
                        else:
                            if player == 0:
                                score += 50
                            else:
                                score -= 50

                    # diagonal check (down-left) for 2 consecutive same colour pieces
                    if row <= 2 and col >= 1 and (state.grid[row + 1][col - 1] == player):
                        if row > 0 and col < 4 and state.grid[row - 1][col + 1] is None:
                            distance = self.closest_distance_piece(state, row - 1, col + 1, player)
                            if player == 0:
                                score += max(0, 50 * (10 - distance))
                            else:
                                score -= max(0, 50 * (10 - distance))
                        elif row < 2 and col > 1 and state.grid[row + 2][col - 2] is None:
                            distance = self.closest_distance_piece(state, row + 2, col - 2, player)
                            if player == 0:
                                score += max(0, 50 * (10 - distance))
                            else:
                                score -= max(0, 50 * (10 - distance))
                        else:
                            if player == 0:
                                score += 50
                            else:
                                score -= 50

        # Added score for more available moves
        if state.current_player == 0:
            score += len(state.get_possible_moves()) * 10
        else:
            score -= len(state.get_possible_moves()) * 10
        
        # Blocking two-in-a-rows of opponent
        opponent = 1 - state.current_player
        for row in range(4):
            for col in range(5):
                if state.grid[row][col] == opponent:
                    # horizontal check for 2 consecutive same colour pieces
                    if col <= 3 and state.grid[row][col + 1] == opponent:
                        if col > 0 and state.grid[row][col - 1] is None :
                            distance = self.closest_distance_piece(state, row, col - 1, state.current_player)
                            # if can block in <=2 move
                            if distance <= 2:
                                if state.current_player == 0:
                                    score += 100
                                else:
                                    score -= 100
                            else:
                                if state.current_player == 0:
                                    score -= 200
                                else:
                                    score += 200
                        elif col < 3 and state.grid[row][col + 2] is None:
                            distance = self.closest_distance_piece(state, row, col + 2, state.current_player)
                            # if can block in <=2 move
                            if distance <= 2:
                                if state.current_player == 0:
                                    score += 100
                                else:
                                    score -= 100
                            else:
                                if state.current_player == 0:
                                    score -= 200
                                else:
                                    score += 200
                                
                    
                    # vertical check for 2 consecutive same colour pieces
                    if row <= 2 and state.grid[row + 1][col] == opponent:
                        if row > 0 and state.grid[row - 1][col] is None:
                            distance = self.closest_distance_piece(state, row - 1, col, state.current_player)
                            # if can block in <=2 move
                            if distance <= 2:
                                if state.current_player == 0:
                                    score += 100
                                else:
                                    score -= 100
                            else:
                                if state.current_player == 0:
                                    score -= 200
                                else:
                                    score += 200 
                        if row < 2 and state.grid[row + 2][col] is None:
                            distance = self.closest_distance_piece(state, row + 2, col, state.current_player)
                            # if can block in <=2 move
                            if distance <= 2:
                                if state.current_player == 0:
                                    score += 100
                                else:
                                    score -= 100
                            else:
                                if state.current_player == 0:
                                    score -= 200
                                else:
                                    score += 200

                    # diagonal check (down-right) for 2 consecutive same colour pieces
                    if row <= 2 and col <= 3 and state.grid[row + 1][col + 1] == opponent:
                        if row > 0 and col > 0 and state.grid[row - 1][col - 1] is None:
                            distance = self.closest_distance_piece(state, row - 1, col - 1, state.current_player)
                            # if can block in <=2 move
                            if distance <= 2:
                                if state.current_player == 0:
                                    score += 100
                                else:
                                    score -= 100
                            else:
                                if state.current_player == 0:
                                    score -= 200
                                else:
                                    score += 200

                        if row < 2 and col < 3 and state.grid[row + 2][col + 2] is None:
                            distance = self.closest_distance_piece(state, row + 2, col + 2, state.current_player)
                            # if can block in <=2 move
                            if distance <= 2:
                                if state.current_player == 0:
                                    score += 100
                                else:
                                    score -= 100
                            else:
                                if state.current_player == 0:
                                    score -= 200
                                else:
                                    score += 200

                    # diagonal check (down-left) for 2 consecutive same colour pieces
                    if row <= 2 and col >= 1 and (state.grid[row + 1][col - 1] == opponent):
                        if row > 0 and col < 4 and state.grid[row - 1][col + 1] is None:
                            distance = self.closest_distance_piece(state, row - 1, col + 1, state.current_player)
                            # if can block in <=2 move
                            if distance <= 2:
                                if state.current_player == 0:
                                    score += 100
                                else:
                                    score -= 100
                            else:
                                if state.current_player == 0:
                                    score -= 200
                                else:
                                    score += 200

                        if row < 2 and col > 1 and state.grid[row + 2][col - 2] is None:
                            distance = self.closest_distance_piece(state, row + 2, col - 2, state.current_player)
                            # if can block in <=2 move
                            if distance <= 2:
                                if state.current_player == 0:
                                    score += 100
                                else:
                                    score -= 100
                            else:
                                if state.current_player == 0:
                                    score -= 200
                                else:
                                    score += 200


        return score
    
    def closest_distance_piece(self, state: DC3, row: int, col: int, player: int) -> int:
        # Calculate Manhattan distance to the closest piece to a two-in-a-row
        min_distance = float('inf')
        for r in range(4):
            for c in range(5):
                if state.grid[r][c] == player:
                    distance = abs(r - row) + abs(c - col)
                    if distance < min_distance:
                        min_distance = distance
        return min_distance if min_distance != float('inf') else 0

    def move_to_string(self, move: Tuple[Tuple[int, int], Tuple[int, int]]) -> str:
        # Convert move tuple to server format string
        (x, y), (dest_x, dest_y) = move
        
        # Determine direction
        if dest_x == x and dest_y == y - 1:
            direction = 'N'
        elif dest_x == x and dest_y == y + 1:
            direction = 'S'
        elif dest_x == x + 1 and dest_y == y:
            direction = 'E'
        elif dest_x == x - 1 and dest_y == y:
            direction = 'W'
        else:
            raise ValueError("Invalid move")
        
        # Convert to 1-indexed and format
        return f"{x + 1}{y + 1}{direction}"

# Server communication generated by GitHub Copilot, utilizing previously defined game loop logic as a prompt, 
# Assisted with server communication
class ServerGame:
    def __init__(self, server_host, server_port, game_id, color):
        self.server_host = server_host
        self.server_port = server_port
        self.game_id = game_id
        self.color = color.lower()
        self.agent = DC3Agent(depth=6)
        self.game_state = DC3()
        self.my_player = 0 if color.lower() == 'white' else 1
        self.socket = None
        
    def connect_to_server(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            
            # Send initial message: gameID and color
            init_message = f"{self.game_id} {self.color}\n"
            self.socket.send(init_message.encode())
            
            # Receive confirmation
            response = self.socket.recv(1024).decode().strip()
            print(f"Server response: {response}")
            
            return True
        except Exception as e:
            print(f"Error connecting to server: {e}")
            return False
        
    def draw_board(self, state: DC3):
        # Column numbers
        print("  1 2 3 4 5")  
        for row in range(4):
            # Row number
            row_str = []  
            for col in range(5):
                if state.grid[row][col] is not None:
                    row_str.append(str(state.grid[row][col]))
                else:
                    row_str.append(' ')
            print(f"{row + 1} " + ','.join(row_str))
        print()

    
    def play_game(self):
        if not self.connect_to_server():
            return
        
        try:
            while not self.game_state.terminal_test():
                self.draw_board(self.game_state)
                if self.game_state.current_player == self.my_player:
                    # calculate and send move
                    print(self.game_state.get_possible_moves())
                    move = self.agent.alpha_beta(self.game_state)
                    move_str = self.agent.move_to_string(move)
                    print(f"Sending move: {move_str}")
                    
                    self.socket.send((move_str + "\n").encode())
                    self.game_state = self.game_state.move(move_str)
                    
                    # Wait for server confirmation
                    try:
                        response = self.socket.recv(1024).decode().strip()
                        print(f"Server confirmed: {response}")
                    except Exception as e:
                        print(f"Error receiving confirmation: {e}")
                        break
                    
                    # Check if game ended after agent move
                    if self.game_state.terminal_test():
                        self.draw_board(self.game_state)
                        winner = self.game_state.check_win_state()
                        if winner == 0:
                            print("White wins!")
                        elif winner == 1:
                            print("Black wins!")
                        else:
                            print("Draw!")
                        break
                else:
                    # Opponent's turn
                    print("Waiting for opponent's move...")
                    
                    try:
                        response = self.socket.recv(1024).decode().strip()
                        if not response:
                            break
                        
                        print(f"Received opponent move: {response}")
                        
                        # Apply opponent's move
                        move_str = response.strip()
                        self.game_state = self.game_state.move(move_str)
                        
                        # Check if game ended after opponent's move
                        if self.game_state.terminal_test():
                            self.draw_board(self.game_state)
                            winner = self.game_state.check_win_state()
                            if winner == 0:
                                print("White wins!")
                            elif winner == 1:
                                print("Black wins!")
                            else:
                                print("Draw!")
                            break
                        
                    except socket.timeout:
                        print("Timeout waiting for opponent move")
                        break
                    except Exception as e:
                        print(f"Error receiving move: {e}")
                        break
                    
        except Exception as e:
            print(f"Game error: {e}")
        finally:
            if self.socket:
                self.socket.close()

def main():
    if len(sys.argv) != 4:
        print("Usage: python dc3newheuristic.py <server_host> <game_id> <color>")
        sys.exit(1)
    
    server_host = sys.argv[1]
    game_id = sys.argv[2]
    color = sys.argv[3]
    server_port = 12345  
    
    if color.lower() not in ['white', 'black']:
        print("Color must be 'white' or 'black'")
        sys.exit(1)
    
    print(f"Connecting to {server_host}:{server_port} as {color} player for game {game_id}")
    
    game = ServerGame(server_host, server_port, game_id, color)
    game.play_game()

if __name__ == "__main__":
    main()
