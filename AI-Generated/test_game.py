"""
Test suite for Dynamic Connect-3 game implementation
"""

import unittest
from dynamic_connect3 import GameState, Player, MinimaxAgent

class TestGameState(unittest.TestCase):
    
    def setUp(self):
        self.game = GameState()
    
    def test_initial_setup(self):
        """Test that the initial board is set up correctly"""
        # Check leftmost column: W-B-W-B
        self.assertEqual(self.game.grid[0][0], Player.WHITE)
        self.assertEqual(self.game.grid[1][0], Player.BLACK)
        self.assertEqual(self.game.grid[2][0], Player.WHITE)
        self.assertEqual(self.game.grid[3][0], Player.BLACK)
        
        # Check rightmost column: B-W-B-W
        self.assertEqual(self.game.grid[0][3], Player.BLACK)
        self.assertEqual(self.game.grid[1][3], Player.WHITE)
        self.assertEqual(self.game.grid[2][3], Player.BLACK)
        self.assertEqual(self.game.grid[3][3], Player.WHITE)
        
        # Check middle columns are empty
        for col in [1, 2]:
            for row in range(4):
                self.assertEqual(self.game.grid[col][row], Player.EMPTY)
        
        # Check white starts first
        self.assertEqual(self.game.current_player, Player.WHITE)
    
    def test_valid_moves(self):
        """Test move validation"""
        # Valid move: White piece from (0,0) to (0,1)
        self.assertTrue(self.game.is_valid_move((0, 0), (0, 1)))
        
        # Invalid move: piece doesn't belong to current player
        self.assertFalse(self.game.is_valid_move((1, 0), (1, 1)))  # Black piece
        
        # Invalid move: destination not empty
        self.assertFalse(self.game.is_valid_move((0, 0), (1, 0)))  # Black piece at destination
        
        # Invalid move: too far
        self.assertFalse(self.game.is_valid_move((0, 0), (0, 2)))
        
        # Invalid move: diagonal
        self.assertFalse(self.game.is_valid_move((0, 0), (1, 1)))
    
    def test_make_move(self):
        """Test making moves"""
        # Make a valid move
        result = self.game.make_move((0, 0), (0, 1))
        self.assertTrue(result)
        
        # Check piece moved
        self.assertEqual(self.game.grid[0][1], Player.WHITE)
        self.assertEqual(self.game.grid[0][0], Player.EMPTY)
        
        # Check player switched
        self.assertEqual(self.game.current_player, Player.BLACK)
        
        # Try invalid move
        result = self.game.make_move((0, 1), (0, 3))  # Too far
        self.assertFalse(result)
    
    def test_undo_move(self):
        """Test undoing moves"""
        # Make a move
        original_state = self.game.get_board_hash()
        self.game.make_move((0, 0), (0, 1))
        
        # Undo the move
        self.game.undo_move((0, 0), (0, 1), Player.WHITE)
        
        # Check state restored
        self.assertEqual(self.game.get_board_hash(), original_state)
        self.assertEqual(self.game.grid[0][0], Player.WHITE)
        self.assertEqual(self.game.grid[0][1], Player.EMPTY)
        self.assertEqual(self.game.current_player, Player.WHITE)
    
    def test_win_detection(self):
        """Test winning condition detection"""
        # Create a winning position for White
        self.game.grid[0] = [Player.WHITE, Player.WHITE, Player.WHITE, Player.EMPTY]
        winner = self.game.check_winner()
        self.assertEqual(winner, Player.WHITE)
        
        # Create a winning position for Black (diagonal)
        self.game = GameState()
        self.game.grid[0][0] = Player.BLACK
        self.game.grid[1][1] = Player.BLACK
        self.game.grid[2][2] = Player.BLACK
        winner = self.game.check_winner()
        self.assertEqual(winner, Player.BLACK)
    
    def test_heuristic_function(self):
        """Test the heuristic evaluation function"""
        # Initial position should have some evaluation
        initial_eval = self.game.heuristic_evaluation()
        
        # Create a position with white runs of 2
        self.game = GameState()
        # Clear the board
        for col in range(5):
            for row in range(4):
                self.game.grid[col][row] = Player.EMPTY
        
        # Add two white pieces in a row
        self.game.grid[0][0] = Player.WHITE
        self.game.grid[1][0] = Player.WHITE
        
        white_runs = self.game.count_runs_of_two(Player.WHITE)
        black_runs = self.game.count_runs_of_two(Player.BLACK)
        
        self.assertEqual(white_runs, 1)
        self.assertEqual(black_runs, 0)
        self.assertEqual(self.game.heuristic_evaluation(), 1)


class TestMinimaxAgent(unittest.TestCase):
    
    def setUp(self):
        self.game = GameState()
        self.agent = MinimaxAgent(max_depth=3)
    
    def test_minimax_returns_valid_move(self):
        """Test that minimax returns a valid move"""
        best_move = self.agent.get_best_move(self.game)
        self.assertIsNotNone(best_move)
        
        from_pos, to_pos = best_move
        self.assertTrue(self.game.is_valid_move(from_pos, to_pos))
    
    def test_minimax_prefers_winning_move(self):
        """Test that minimax chooses winning moves when available"""
        # Create a position where White can win in one move
        self.game = GameState()
        # Clear board
        for col in range(5):
            for row in range(4):
                self.game.grid[col][row] = Player.EMPTY
        
        # Set up almost-winning position for White
        self.game.grid[0][0] = Player.WHITE
        self.game.grid[1][0] = Player.WHITE
        # White can win by placing at (2,0)
        
        self.game.current_player = Player.WHITE
        
        # Add a white piece that can move to (2,0)
        self.game.grid[2][1] = Player.WHITE
        
        best_move = self.agent.get_best_move(self.game)
        
        # Should choose the winning move
        self.assertIsNotNone(best_move)
        from_pos, to_pos = best_move
        
        # Make the move and check if it wins
        self.game.make_move(from_pos, to_pos)
        winner = self.game.check_winner()
        self.assertEqual(winner, Player.WHITE)


def run_performance_test():
    """Test the performance of the minimax algorithm"""
    print("Performance Test:")
    print("================")
    
    game = GameState()
    
    for depth in [2, 3, 4, 5]:
        agent = MinimaxAgent(max_depth=depth)
        import time
        
        start_time = time.time()
        best_move = agent.get_best_move(game)
        end_time = time.time()
        
        print(f"Depth {depth}: {agent.nodes_evaluated} nodes, {end_time - start_time:.3f}s")

def demo_game_progression():
    """Demonstrate a few moves of the game"""
    print("Game Progression Demo:")
    print("=====================")
    
    game = GameState()
    agent = MinimaxAgent(max_depth=4)
    
    print("Initial position:")
    game.display()
    
    for move_num in range(5):
        print(f"Move {move_num + 1}:")
        best_move = agent.get_best_move(game)
        
        if best_move is None:
            print("No moves available!")
            break
        
        from_pos, to_pos = best_move
        current_player = "White" if game.current_player == Player.WHITE else "Black"
        print(f"{current_player} plays: {from_pos} -> {to_pos}")
        print(f"Nodes evaluated: {agent.nodes_evaluated}")
        
        game.make_move(from_pos, to_pos)
        game.display()
        
        is_over, winner = game.is_game_over()
        if is_over:
            if winner:
                print(f"Game over! {winner.name} wins!")
            else:
                print("Game over! Draw!")
            break

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*50 + "\n")
    
    # Run performance test
    run_performance_test()
    
    print("\n" + "="*50 + "\n")
    
    # Run demo
    demo_game_progression()
