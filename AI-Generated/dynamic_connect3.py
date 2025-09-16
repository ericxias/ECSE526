"""
Dynamic Connect-3 Game Implementation with Minimax AI Agent
"""

import copy
from typing import List, Tuple, Optional, Set
from enum import Enum

class Player(Enum):
    WHITE = 1
    BLACK = -1
    EMPTY = 0

class GameState:
    def __init__(self):
        """Initialize the 5x4 grid with the starting position"""
        self.grid = [[Player.EMPTY for _ in range(4)] for _ in range(5)]
        self.setup_initial_position()
        self.current_player = Player.WHITE
        self.move_history = []  # For threefold repetition detection
        
    def setup_initial_position(self):
        """Setup initial position: leftmost columns W-B-W-B, rightmost B-W-B-W"""
        # Leftmost column (column 0): W-B-W-B
        self.grid[0][0] = Player.WHITE
        self.grid[1][0] = Player.BLACK
        self.grid[2][0] = Player.WHITE
        self.grid[3][0] = Player.BLACK
        
        # Rightmost column (column 3): B-W-B-W
        self.grid[0][3] = Player.BLACK
        self.grid[1][3] = Player.WHITE
        self.grid[2][3] = Player.BLACK
        self.grid[3][3] = Player.WHITE
    
    def get_board_hash(self) -> str:
        """Get a string representation of the board for repetition detection"""
        board_str = ""
        for row in range(4):
            for col in range(5):
                if self.grid[col][row] == Player.WHITE:
                    board_str += "W"
                elif self.grid[col][row] == Player.BLACK:
                    board_str += "B"
                else:
                    board_str += "."
        return board_str + str(self.current_player.value)
    
    def is_valid_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Check if a move is valid"""
        from_col, from_row = from_pos
        to_col, to_row = to_pos
        
        # Check bounds
        if not (0 <= from_col < 5 and 0 <= from_row < 4):
            return False
        if not (0 <= to_col < 5 and 0 <= to_row < 4):
            return False
        
        # Check if source has current player's piece
        if self.grid[from_col][from_row] != self.current_player:
            return False
        
        # Check if destination is empty
        if self.grid[to_col][to_row] != Player.EMPTY:
            return False
        
        # Check if move is one square (horizontal or vertical)
        col_diff = abs(to_col - from_col)
        row_diff = abs(to_row - from_row)
        
        return (col_diff == 1 and row_diff == 0) or (col_diff == 0 and row_diff == 1)
    
    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Make a move and return True if successful"""
        if not self.is_valid_move(from_pos, to_pos):
            return False
        
        from_col, from_row = from_pos
        to_col, to_row = to_pos
        
        # Record move for history
        move = (from_pos, to_pos, self.current_player)
        self.move_history.append(move)
        
        # Make the move
        self.grid[to_col][to_row] = self.current_player
        self.grid[from_col][from_row] = Player.EMPTY
        
        # Switch players
        self.current_player = Player.BLACK if self.current_player == Player.WHITE else Player.WHITE
        
        return True
    
    def undo_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], player: Player):
        """Undo a move"""
        from_col, from_row = from_pos
        to_col, to_row = to_pos
        
        # Restore the piece
        self.grid[from_col][from_row] = player
        self.grid[to_col][to_row] = Player.EMPTY
        
        # Switch back the current player
        self.current_player = player
        
        # Remove from history
        if self.move_history:
            self.move_history.pop()
    
    def get_valid_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get all valid moves for the current player"""
        moves = []
        
        for col in range(5):
            for row in range(4):
                if self.grid[col][row] == self.current_player:
                    # Try all adjacent positions
                    for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        new_col, new_row = col + dc, row + dr
                        if self.is_valid_move((col, row), (new_col, new_row)):
                            moves.append(((col, row), (new_col, new_row)))
        
        return moves
    
    def check_winner(self) -> Optional[Player]:
        """Check if there's a winner (3 in a row)"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonal
        
        for col in range(5):
            for row in range(4):
                if self.grid[col][row] != Player.EMPTY:
                    player = self.grid[col][row]
                    
                    for dc, dr in directions:
                        count = 1
                        # Check forward
                        c, r = col + dc, row + dr
                        while (0 <= c < 5 and 0 <= r < 4 and 
                               self.grid[c][r] == player):
                            count += 1
                            c, r = c + dc, r + dr
                        
                        # Check backward
                        c, r = col - dc, row - dr
                        while (0 <= c < 5 and 0 <= r < 4 and 
                               self.grid[c][r] == player):
                            count += 1
                            c, r = c - dc, r - dr
                        
                        if count >= 3:
                            return player
        
        return None
    
    def is_threefold_repetition(self) -> bool:
        """Check for threefold repetition"""
        current_hash = self.get_board_hash()
        count = 0
        
        # Check all previous positions
        temp_state = copy.deepcopy(self)
        for move in reversed(self.move_history):
            from_pos, to_pos, player = move
            temp_state.undo_move(from_pos, to_pos, player)
            if temp_state.get_board_hash() == current_hash:
                count += 1
                if count >= 2:  # Including current position makes 3
                    return True
        
        return False
    
    def is_game_over(self) -> Tuple[bool, Optional[Player]]:
        """Check if game is over and return (is_over, winner)"""
        winner = self.check_winner()
        if winner is not None:
            return True, winner
        
        if self.is_threefold_repetition():
            return True, None  # Draw
        
        if not self.get_valid_moves():
            return True, None  # No moves available - draw
        
        return False, None
    
    def count_runs_of_two(self, player: Player) -> int:
        """Count runs of exactly 2 pieces for the given player"""
        count = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        checked_positions = set()
        
        for col in range(5):
            for row in range(4):
                if self.grid[col][row] == player:
                    for dc, dr in directions:
                        # Check if we can form a run of 2 in this direction
                        if (col, row, dc, dr) not in checked_positions:
                            run_length = 1
                            positions = [(col, row)]
                            
                            # Check forward
                            c, r = col + dc, row + dr
                            while (0 <= c < 5 and 0 <= r < 4 and 
                                   self.grid[c][r] == player):
                                run_length += 1
                                positions.append((c, r))
                                c, r = c + dc, r + dr
                            
                            # If we have exactly 2 in a row, count it
                            if run_length == 2:
                                count += 1
                                # Mark these positions as checked for this direction
                                for pos in positions:
                                    checked_positions.add((pos[0], pos[1], dc, dr))
                                    checked_positions.add((pos[0], pos[1], -dc, -dr))
        
        return count
    
    def heuristic_evaluation(self) -> int:
        """Heuristic evaluation: H(n) = runs_of_2_white - runs_of_2_black"""
        white_runs = self.count_runs_of_two(Player.WHITE)
        black_runs = self.count_runs_of_two(Player.BLACK)
        return white_runs - black_runs
    
    def display(self):
        """Display the current board state"""
        print("  0 1 2 3 4")
        for row in range(4):
            print(f"{row} ", end="")
            for col in range(5):
                if self.grid[col][row] == Player.WHITE:
                    print("W ", end="")
                elif self.grid[col][row] == Player.BLACK:
                    print("B ", end="")
                else:
                    print(". ", end="")
            print()
        print(f"Current player: {'White' if self.current_player == Player.WHITE else 'Black'}")
        print(f"Heuristic value: {self.heuristic_evaluation()}")
        print()


class MinimaxAgent:
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.nodes_evaluated = 0
    
    def minimax(self, state: GameState, depth: int, maximizing_player: bool, 
                alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[int, Optional[Tuple]]:
        """
        Minimax algorithm with alpha-beta pruning
        Returns (evaluation, best_move)
        """
        self.nodes_evaluated += 1
        
        # Check if game is over
        is_over, winner = state.is_game_over()
        if is_over:
            if winner == Player.WHITE:
                return 1000, None
            elif winner == Player.BLACK:
                return -1000, None
            else:  # Draw
                return 0, None
        
        # Check depth limit
        if depth >= self.max_depth:
            return state.heuristic_evaluation(), None
        
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            return state.heuristic_evaluation(), None
        
        best_move = None
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                from_pos, to_pos = move
                
                # Make move
                original_player = state.current_player
                state.make_move(from_pos, to_pos)
                
                # Recursive call
                eval_score, _ = self.minimax(state, depth + 1, False, alpha, beta)
                
                # Undo move
                state.undo_move(from_pos, to_pos, original_player)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in valid_moves:
                from_pos, to_pos = move
                
                # Make move
                original_player = state.current_player
                state.make_move(from_pos, to_pos)
                
                # Recursive call
                eval_score, _ = self.minimax(state, depth + 1, True, alpha, beta)
                
                # Undo move
                state.undo_move(from_pos, to_pos, original_player)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return min_eval, best_move
    
    def get_best_move(self, state: GameState) -> Optional[Tuple]:
        """Get the best move for the current player"""
        self.nodes_evaluated = 0
        maximizing = state.current_player == Player.WHITE
        _, best_move = self.minimax(state, 0, maximizing)
        return best_move


def play_game():
    """Play a game between human and AI or AI vs AI"""
    game = GameState()
    ai_agent = MinimaxAgent(max_depth=6)
    
    print("Dynamic Connect-3 Game")
    print("Initial board:")
    game.display()
    
    while True:
        is_over, winner = game.is_game_over()
        if is_over:
            if winner == Player.WHITE:
                print("White wins!")
            elif winner == Player.BLACK:
                print("Black wins!")
            else:
                print("Game ends in a draw!")
            break
        
        current_player_name = "White" if game.current_player == Player.WHITE else "Black"
        print(f"{current_player_name}'s turn")
        
        # Get move from AI
        best_move = ai_agent.get_best_move(game)
        
        if best_move is None:
            print("No valid moves available!")
            break
        
        from_pos, to_pos = best_move
        print(f"AI chooses: Move from {from_pos} to {to_pos}")
        print(f"Nodes evaluated: {ai_agent.nodes_evaluated}")
        
        game.make_move(from_pos, to_pos)
        game.display()


if __name__ == "__main__":
    play_game()
