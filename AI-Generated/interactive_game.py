"""
Interactive Dynamic Connect-3 Game
Human vs AI or AI vs AI gameplay
"""

from dynamic_connect3 import GameState, Player, MinimaxAgent
import re

def parse_move_input(input_str: str):
    """Parse move input like 'a1 b1' or '0,0 1,0'"""
    input_str = input_str.strip().lower()
    
    # Try letter-number format (a1 b1)
    pattern1 = r'([a-e])([1-4])\s+([a-e])([1-4])'
    match1 = re.match(pattern1, input_str)
    if match1:
        from_col = ord(match1.group(1)) - ord('a')
        from_row = int(match1.group(2)) - 1
        to_col = ord(match1.group(3)) - ord('a')
        to_row = int(match1.group(4)) - 1
        return (from_col, from_row), (to_col, to_row)
    
    # Try coordinate format (0,0 1,0)
    pattern2 = r'(\d),(\d)\s+(\d),(\d)'
    match2 = re.match(pattern2, input_str)
    if match2:
        from_col = int(match2.group(1))
        from_row = int(match2.group(2))
        to_col = int(match2.group(3))
        to_row = int(match2.group(4))
        return (from_col, from_row), (to_col, to_row)
    
    return None, None

def display_help():
    """Display help for move input"""
    print("\nMove format:")
    print("  Letter-number: 'a1 b1' (move from column a, row 1 to column b, row 1)")
    print("  Coordinates: '0,0 1,0' (move from col 0, row 0 to col 1, row 0)")
    print("  Columns: a-e (0-4), Rows: 1-4 (0-3)")
    print("  Type 'quit' to exit, 'help' for this message")
    print()

def play_human_vs_ai():
    """Play human vs AI"""
    game = GameState()
    ai_agent = MinimaxAgent(max_depth=6)
    
    print("Dynamic Connect-3: Human vs AI")
    print("You are White, AI is Black")
    display_help()
    game.display()
    
    while True:
        is_over, winner = game.is_game_over()
        if is_over:
            if winner == Player.WHITE:
                print("Congratulations! You win!")
            elif winner == Player.BLACK:
                print("AI wins!")
            else:
                print("Game ends in a draw!")
            break
        
        if game.current_player == Player.WHITE:
            # Human turn
            print("Your turn (White):")
            print("Valid moves:")
            valid_moves = game.get_valid_moves()
            for i, (from_pos, to_pos) in enumerate(valid_moves):
                col_letter = chr(ord('a') + from_pos[0])
                to_col_letter = chr(ord('a') + to_pos[0])
                print(f"  {i+1}. {col_letter}{from_pos[1]+1} to {to_col_letter}{to_pos[1]+1}")
            
            while True:
                user_input = input("Enter your move: ").strip()
                
                if user_input.lower() == 'quit':
                    return
                elif user_input.lower() == 'help':
                    display_help()
                    continue
                
                from_pos, to_pos = parse_move_input(user_input)
                
                if from_pos is None:
                    print("Invalid format. Type 'help' for format examples.")
                    continue
                
                if game.make_move(from_pos, to_pos):
                    break
                else:
                    print("Invalid move. Please try again.")
        
        else:
            # AI turn
            print("AI's turn (Black):")
            best_move = ai_agent.get_best_move(game)
            
            if best_move is None:
                print("AI has no valid moves!")
                break
            
            from_pos, to_pos = best_move
            from_col_letter = chr(ord('a') + from_pos[0])
            to_col_letter = chr(ord('a') + to_pos[0])
            print(f"AI moves: {from_col_letter}{from_pos[1]+1} to {to_col_letter}{to_pos[1]+1}")
            print(f"Nodes evaluated: {ai_agent.nodes_evaluated}")
            
            game.make_move(from_pos, to_pos)
        
        game.display()

def play_ai_vs_ai():
    """Play AI vs AI with different depths"""
    game = GameState()
    ai1 = MinimaxAgent(max_depth=4)  # White
    ai2 = MinimaxAgent(max_depth=6)  # Black
    
    print("Dynamic Connect-3: AI vs AI")
    print("AI 1 (White, depth 4) vs AI 2 (Black, depth 6)")
    game.display()
    
    move_count = 0
    while True:
        is_over, winner = game.is_game_over()
        if is_over:
            if winner == Player.WHITE:
                print("AI 1 (White) wins!")
            elif winner == Player.BLACK:
                print("AI 2 (Black) wins!")
            else:
                print("Game ends in a draw!")
            break
        
        move_count += 1
        current_ai = ai1 if game.current_player == Player.WHITE else ai2
        ai_name = "AI 1 (White)" if game.current_player == Player.WHITE else "AI 2 (Black)"
        
        print(f"Move {move_count}: {ai_name}'s turn")
        
        best_move = current_ai.get_best_move(game)
        
        if best_move is None:
            print(f"{ai_name} has no valid moves!")
            break
        
        from_pos, to_pos = best_move
        from_col_letter = chr(ord('a') + from_pos[0])
        to_col_letter = chr(ord('a') + to_pos[0])
        print(f"{ai_name} moves: {from_col_letter}{from_pos[1]+1} to {to_col_letter}{to_pos[1]+1}")
        print(f"Nodes evaluated: {current_ai.nodes_evaluated}")
        
        game.make_move(from_pos, to_pos)
        game.display()
        
        # Pause for readability
        input("Press Enter to continue...")

def test_heuristic():
    """Test the heuristic function with some sample positions"""
    print("Testing heuristic evaluation function...")
    
    # Test initial position
    game = GameState()
    print("Initial position:")
    game.display()
    print(f"White runs of 2: {game.count_runs_of_two(Player.WHITE)}")
    print(f"Black runs of 2: {game.count_runs_of_two(Player.BLACK)}")
    print()
    
    # Make a few moves to create some runs
    moves = [((0, 0), (1, 0)), ((3, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 3), (2, 2))]
    
    for from_pos, to_pos in moves:
        if game.make_move(from_pos, to_pos):
            print(f"After move {from_pos} -> {to_pos}:")
            game.display()
            print(f"White runs of 2: {game.count_runs_of_two(Player.WHITE)}")
            print(f"Black runs of 2: {game.count_runs_of_two(Player.BLACK)}")
            print()

def main():
    """Main menu"""
    while True:
        print("\nDynamic Connect-3 Game")
        print("1. Human vs AI")
        print("2. AI vs AI")
        print("3. Test Heuristic")
        print("4. Quit")
        
        choice = input("Choose option (1-4): ").strip()
        
        if choice == '1':
            play_human_vs_ai()
        elif choice == '2':
            play_ai_vs_ai()
        elif choice == '3':
            test_heuristic()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
