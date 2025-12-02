from typing import Optional, Tuple
from go_engine import GoBoard, Player, column_to_label, label_to_column
from mcts_agent import MCTSAgent, RandomAgent


class GameUI:
    def __init__(self, board_size: int = 9):
        """
        Initialize the game UI.
        
        Args:
            board_size: Size of the Go board (default 9)
        """
        self.board = GoBoard(board_size)
        self.ai_agent = None
        self.ai_player = None
    
    def display_board(self):
        """Display the current board state."""
        print("\n" + "=" * 40)
        print(self.board)
        print(f"Current player: {'Black (●)' if self.board.current_player == Player.BLACK else 'White (○)'}")
        print(f"Captures - Black: {self.board.captured_stones[Player.BLACK]}, "
              f"White: {self.board.captured_stones[Player.WHITE]}")
        print("=" * 40 + "\n")
    
    def parse_move(self, move_str: str) -> Optional[Tuple[int, int]]:
        """
        Parse a move string like 'A1' or 'D5' into board coordinates.
        
        Args:
            move_str: The move string to parse
        
        Returns:
            Tuple of (row, col) or None if invalid
        """
        move_str = move_str.strip().upper()
        
        if move_str == "PASS":
            return None
        
        if len(move_str) < 2:
            return None
        
        col_char = move_str[0]
        row_str = move_str[1:]
        
        try:
            col = label_to_column(col_char)
            row = int(row_str) - 1
            
            if 0 <= row < self.board.size and 0 <= col < self.board.size:
                return (row, col)
        except ValueError:
            pass
        
        return None
    
    def get_human_move(self) -> Optional[Tuple[int, int]]:
        """
        Get a move from the human player.
        
        Returns:
            Move coordinates or None for pass
        """
        while True:
            move_str = input(f"Enter your move (e.g., 'D5' or 'PASS'): ").strip()
            
            if move_str.upper() == "PASS":
                return None
            
            if move_str.upper() == "QUIT":
                return "QUIT"
            
            move = self.parse_move(move_str)
            
            if move is None:
                print("Invalid move format. Use format like 'D5' or 'PASS'.")
                continue
            
            row, col = move
            
            if self.board.is_legal_move(row, col):
                return move
            else:
                print("Illegal move! Try again.")
                print("Reasons a move might be illegal:")
                print("  - Position is already occupied")
                print("  - Move violates Ko rule")
                print("  - Move is suicidal (results in immediate capture without capturing opponent)")
    
    def play_human_vs_human(self):
        """Play a game of human vs human."""
        print("\n" + "=" * 40)
        print("    Go Game - Human vs Human")
        print("=" * 40)
        print("\nRules:")
        print("  - Enter moves like 'D5' (column letter + row number)")
        print("  - Type 'PASS' to pass your turn")
        print("  - Type 'QUIT' to exit")
        print("  - Game ends when both players pass consecutively")
        
        self.display_board()
        
        while not self.board.is_game_over():
            move = self.get_human_move()
            
            if move == "QUIT":
                print("Game quit by player.")
                return
            
            if move is None:
                self.board.pass_turn()
                print("Pass!")
            else:
                row, col = move
                self.board.make_move(row, col)
            
            self.display_board()
        
        self.show_final_score()
    
    def play_human_vs_ai(self, human_color: Player = Player.BLACK, 
                         ai_simulation_time: float = 5.0):
        """
        Play a game of human vs AI.
        
        Args:
            human_color: Color for human player (BLACK or WHITE)
            ai_simulation_time: Time in seconds for AI to think
        """
        self.ai_player = Player.WHITE if human_color == Player.BLACK else Player.BLACK
        self.ai_agent = MCTSAgent(simulation_time=ai_simulation_time)
        
        print("\n" + "=" * 40)
        print("    Go Game - Human vs AI (MCTS)")
        print("=" * 40)
        print(f"\nYou are playing as {'Black (●)' if human_color == Player.BLACK else 'White (○)'}")
        print(f"AI is playing as {'Black (●)' if self.ai_player == Player.BLACK else 'White (○)'}")
        print(f"AI thinking time: {ai_simulation_time} seconds per move")
        print("\nRules:")
        print("  - Enter moves like 'D5' (column letter + row number)")
        print("  - Type 'PASS' to pass your turn")
        print("  - Type 'QUIT' to exit")
        print("  - Game ends when both players pass consecutively")
        
        self.display_board()
        
        while not self.board.is_game_over():
            if self.board.current_player == human_color:
                # Human turn
                move = self.get_human_move()
                
                if move == "QUIT":
                    print("Game quit by player.")
                    return
                
                if move is None:
                    self.board.pass_turn()
                    print("You passed!")
                else:
                    row, col = move
                    self.board.make_move(row, col)
            else:
                # AI turn
                print(f"\nAI ({'Black' if self.ai_player == Player.BLACK else 'White'}) is thinking...")
                move = self.ai_agent.select_move(self.board)
                
                if move is None:
                    self.board.pass_turn()
                    print("AI passed!")
                else:
                    row, col = move
                    move_str = f"{column_to_label(col)}{row + 1}"
                    self.board.make_move(row, col)
                    print(f"AI played: {move_str}")
            
            self.display_board()
        
        self.show_final_score()
    
    def play_ai_vs_ai(self, simulation_time: float = 2.0):
        """
        Watch two AI agents play against each other.
        
        Args:
            simulation_time: Time in seconds for each AI to think
        """
        black_ai = MCTSAgent(simulation_time=simulation_time)
        white_ai = MCTSAgent(simulation_time=simulation_time)
        
        print("\n" + "=" * 40)
        print("    Go Game - AI vs AI (MCTS)")
        print("=" * 40)
        print(f"\nAI thinking time: {simulation_time} seconds per move")
        
        self.display_board()
        
        move_count = 0
        while not self.board.is_game_over() and move_count < 200:
            current_ai = black_ai if self.board.current_player == Player.BLACK else white_ai
            player_name = "Black" if self.board.current_player == Player.BLACK else "White"
            
            print(f"\n{player_name} AI is thinking...")
            move = current_ai.select_move(self.board)
            
            if move is None:
                self.board.pass_turn()
                print(f"{player_name} AI passed!")
            else:
                row, col = move
                move_str = f"{column_to_label(col)}{row + 1}"
                self.board.make_move(row, col)
                print(f"{player_name} AI played: {move_str}")
            
            self.display_board()
            move_count += 1
            
            # Optional: pause between moves
            input("Press Enter to continue...")
        
        self.show_final_score()
    
    def show_final_score(self):
        """Display the final score and winner."""
        print("\n" + "=" * 40)
        print("           GAME OVER")
        print("=" * 40)
        
        score = self.board.calculate_score()
        
        print(f"\nFinal Score:")
        print(f"  Black: {score['black']:.1f} points")
        print(f"    - Stones on board: {score['black_stones']}")
        print(f"    - Territory: {score['black_territory']}")
        print(f"    - Captures: {score['black_captures']}")
        print(f"\n  White: {score['white']:.1f} points")
        print(f"    - Stones on board: {score['white_stones']}")
        print(f"    - Territory: {score['white_territory']}")
        print(f"    - Captures: {score['white_captures']}")
        print(f"    - Komi: 6.5")
        
        print(f"\n{'Black' if score['winner'] == Player.BLACK else 'White'} wins!")
        print("=" * 40 + "\n")
    
    def show_menu(self):
        """Display the main menu and handle user choice."""
        while True:
            print("\n" + "=" * 40)
            print("         Go Game with MCTS AI")
            print("=" * 40)
            print("\n1. Play as Black vs AI")
            print("2. Play as White vs AI")
            print("3. Human vs Human")
            print("4. Watch AI vs AI")
            print("5. Quit")
            
            choice = input("\nSelect an option (1-5): ").strip()
            
            if choice == "1":
                time_str = input("AI thinking time in seconds (default 5): ").strip()
                try:
                    think_time = float(time_str) if time_str else 5.0
                except ValueError:
                    think_time = 5.0
                self.board = GoBoard()
                self.play_human_vs_ai(Player.BLACK, think_time)
            elif choice == "2":
                time_str = input("AI thinking time in seconds (default 5): ").strip()
                try:
                    think_time = float(time_str) if time_str else 5.0
                except ValueError:
                    think_time = 5.0
                self.board = GoBoard()
                self.play_human_vs_ai(Player.WHITE, think_time)
            elif choice == "3":
                self.board = GoBoard()
                self.play_human_vs_human()
            elif choice == "4":
                time_str = input("AI thinking time in seconds (default 2): ").strip()
                try:
                    think_time = float(time_str) if time_str else 2.0
                except ValueError:
                    think_time = 2.0
                self.board = GoBoard()
                self.play_ai_vs_ai(think_time)
            elif choice == "5":
                print("Thanks for playing!")
                break
            else:
                print("Invalid choice. Please select 1-5.")


def main():
    """Main entry point for the game UI."""
    ui = GameUI()
    ui.show_menu()


if __name__ == "__main__":
    main()
