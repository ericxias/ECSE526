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
        if self.check_win_state() is not None:
            return True

        # Check for draw (threefold repetition)
        if self.grid_history.count(str(self.grid)) >= 3:
            return True

        return False
    
    def move(self, move):
        # Execute move -> Parse string to tuples
        # Check if piece belongs to the current player
        # Check if target location move is legal
        # Execute move
        try:
            # If move is a string, parse it (used in local debugging); if tuple, use directly
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
        self.nodes_traveled = 0

    # Alpha Beta pruning is written utilizing pseudocode from AIMA textbook
    def alpha_beta(self, state: DC3) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        v = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        for move in state.get_possible_moves():
            new_state = state.move(move)
            result = self.min_value(new_state, self.depth - 1, alpha, beta)

            if result > v:
                v = result
                best_move = move

            alpha = max(alpha, v)

        return best_move

    def max_value(self, state: DC3, depth: int, alpha: float, beta: float) -> int:
        self.nodes_traveled += 1

        if state.terminal_test():
            return state.utility()
        if depth == 0:
            return self.heuristic(state)

        v = float('-inf')
        for move in state.get_possible_moves():
            new_state = state.move(move)
            result = self.min_value(new_state, depth - 1, alpha, beta)
            v = max(v, result)
            if v >= beta:
                return v
            alpha = max(alpha, v)

        return v

    def min_value(self, state: DC3, depth: int, alpha: float, beta: float) -> int:
        self.nodes_traveled += 1

        if state.terminal_test():
            return state.utility()
        if depth == 0:
            return self.heuristic(state)

        v = float('inf')
        for move in state.get_possible_moves():
            new_state = state.move(move)
            score = self.max_value(new_state, depth - 1, alpha, beta)
            v = min(v, score)
            if v <= alpha:
                return v
            beta = min(beta, v)

        return v

    def heuristic(self, state: DC3) -> int:
        # Count number of 2-in-a-rows for each player
        score = 0
        # Check for an immediate win in next move
        for move in state.get_possible_moves():
            new_state = state.move(move)
            if new_state.check_win_state() == state.current_player:
                if state.current_player == 0:
                    return 10000
                else:
                    return -10000

        # Check if opponent can win in next move, and block it if possible
        opponent = 1 - state.current_player
        opp_state = copy.deepcopy(state)
        opp_state.current_player = opponent
        winning_moves = []
        for move in opp_state.get_possible_moves():
            if opp_state.move(move).check_win_state() == opponent:
                winning_moves.append(move[1])

        # Check if the winning moves can be blocked
        if len(winning_moves) > 0:
            '''can_block = False
            for opp_moves in winning_moves:
                # Check if we can move to the opponent's winning position 
                for moves in state.get_possible_moves():
                    if moves[1] == opp_moves:
                        can_block = True
                        if state.current_player == 0:
                            score += 2000
                        else:
                            score -= 2000
                        break      '''   
            # Optimization generated with AI to reduce number of iterations
            reachable_positions = set(move[1] for move in state.get_possible_moves())
            can_block = bool(winning_moves & reachable_positions)
            if can_block:
                if state.current_player == 0:
                    score += 2000
                else:
                    score -= 2000
            else:
                if state.current_player == 0:
                    score -= 5000
                else:
                    score += 5000

        trans_table = set()

        # Check number of 2-in-a-rows that can convert to 3-in-a-rows in two move
        for row in range(4):
            for col in range(5):
                if state.grid[row][col] is not None:
                    player = state.grid[row][col]

                    # horizontal check for 2 consecutive same colour pieces
                    if col <= 3 and (state.grid[row][col + 1] == player):
                        explored_move = (row, min(col, col + 1), 'H', player)
                        # Check if evaluation was already explored in trans table
                        if explored_move not in trans_table:
                            trans_table.add(explored_move)
                            # Check if left side is empty to complete 3-in-a-row
                            if col - 1 >= 0 and state.grid[row][col - 1] is None:
                                # Check if two moves makes 3-in-a-row
                                if self.two_move_win(state, player, col, row):
                                    if player == 0:
                                        score += 100
                                    else:
                                        score -= 100
                            # Check if right side is empty to complete 3-in-a-row
                            if col + 2 < 5 and state.grid[row][col + 2] is None:
                                # Check if two moves makes 3-in-a-row
                                if self.two_move_win(state, player, col + 1, row):
                                    if player == 0:
                                        score += 100
                                    else:
                                        score -= 100
                    
                    # vertical check for 2 consecutive same colour pieces
                    if row <= 2 and (state.grid[row + 1][col] == player):
                        explored_move = (min(row, row + 1), col, 'V', player)
                        if explored_move not in trans_table:
                            trans_table.add(explored_move)
                            # Check if above side is empty to complete 3-in-a-row
                            if row - 1 >= 0 and state.grid[row - 1][col] is None:
                                # Check if two moves makes 3-in-a-row
                                if self.two_move_win(state, player, col, row):
                                    if player == 0:
                                        score += 100
                                    else:
                                        score -= 100
                            # Check if below side is empty to complete 3-in-a-row
                            if row + 2 < 4 and state.grid[row + 2][col] is None:
                                # Check if two moves makes 3-in-a-row
                                if self.two_move_win(state, player, col, row + 1):
                                    if player == 0:
                                        score += 100
                                    else:
                                        score -= 100
                    # diagonal check (down-right) for 2 consecutive same colour pieces
                    if row <= 2 and col <= 3 and (state.grid[row + 1][col + 1] == player):
                        explored_move = (min(row, row + 1), min(col, col + 1), 'D1', player)
                        if explored_move not in trans_table:
                            trans_table.add(explored_move)
                            # Check if top-left side is empty to complete 3-in-a-row
                            if row - 1 >= 0 and col - 1 >= 0 and state.grid[row - 1][col - 1] is None:
                                # Check if two moves makes 3-in-a-row
                                if self.two_move_win(state, player, col, row):
                                    if player == 0:
                                        score += 100
                                    else:
                                        score -= 100
                            # Check if bottom-right side is empty to complete 3-in-a-row
                            if row + 2 < 4 and col + 2 < 5 and state.grid[row + 2][col + 2] is None:
                                # Check if two moves makes 3-in-a-row
                                if self.two_move_win(state, player, col + 1, row + 1):
                                    if player == 0:
                                        score += 100
                                    else:
                                        score -= 100
                    # diagonal check (down-left) for 2 consecutive same colour pieces
                    if row <= 2 and col >= 1 and (state.grid[row + 1][col - 1] == player):
                        explored_move = (min(row, row + 1), max(col, col - 1), 'D2', player)
                        if explored_move not in trans_table:
                            trans_table.add(explored_move)
                            # Check if top-right side is empty to complete 3-in-a-row
                            if row - 1 >= 0 and col + 1 < 5 and state.grid[row - 1][col + 1] is None:
                                # Check if two moves makes 3-in-a-row
                                if self.two_move_win(state, player, col, row):
                                    if player == 0:
                                        score += 100
                                    else:
                                        score -= 100
                    # diagonal check (down-left) for 2 consecutive same colour pieces
                    if row <= 2 and col >= 1 and (state.grid[row + 1][col - 1] == player):
                        explored_move = (min(row, row + 1), max(col, col - 1), 'D2', player)
                        if explored_move not in trans_table:
                            trans_table.add(explored_move)
                            # Check if top-right side is empty to complete 3-in-a-row
                            if row - 1 >= 0 and col + 1 < 5 and state.grid[row - 1][col + 1] is None:
                                # Check if two moves makes 3-in-a-row
                                if self.two_move_win(state, player, col, row):
                                    if player == 0:
                                        score += 100
                                    else:
                                        score -= 100
                            # Check if bottom-left side is empty to complete 3-in-a-row
                            if row + 2 < 4 and col - 2 >= 0 and state.grid[row + 2][col - 2] is None:
                                # Check if two moves makes 3-in-a-row
                                if self.two_move_win(state, player, col - 1, row + 1):
                                    if player == 0:
                                        score += 100
                                    else:
                                        score -= 100
        return score
    
    def two_move_win(self, state: DC3, player: int, col: int, row: int) -> bool:
        temp_state = copy.deepcopy(state)
        temp_state.current_player = player
        for move in temp_state.get_possible_moves():
            move_state = temp_state.move(move)
            move_state.current_player = player
            for next_move in move_state.get_possible_moves():
                if move_state.move(next_move).check_win_state() == player:
                    return True
        return False
            

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
        self.agent = DC3Agent(depth=5)
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
                    
                    # Check if game ended after my move
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
        print("Usage: python dc3_server_agent.py <server_host> <game_id> <color>")
        print("Example: python dc3_server_agent.py 156trlinux-1.ece.mcgill.ca game37 white")
        sys.exit(1)
    
    server_host = sys.argv[1]
    game_id = sys.argv[2]
    color = sys.argv[3]
    server_port = 12345  # Default port from the server code
    
    if color.lower() not in ['white', 'black']:
        print("Color must be 'white' or 'black'")
        sys.exit(1)
    
    print(f"Connecting to {server_host}:{server_port} as {color} player for game {game_id}")
    
    game = ServerGame(server_host, server_port, game_id, color)
    game.play_game()

if __name__ == "__main__":
    main()
