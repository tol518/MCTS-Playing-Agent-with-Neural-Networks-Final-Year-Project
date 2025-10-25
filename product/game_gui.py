"""
Graphical User Interface for Go game using tkinter.
Provides a 2D visual board with clickable intersections and proper coloring.
"""

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional, Tuple
from go_engine import GoBoard, Player
from mcts_agent import MCTSAgent
import threading


class GoGUI:
    """
    Graphical Go game interface with 2D board visualization.
    """
    
    def __init__(self, board_size: int = 9):
        """
        Initialize the GUI.
        
        Args:
            board_size: Size of the Go board (default 9)
        """
        self.board_size = board_size
        self.board = GoBoard(board_size)
        self.ai_agent = None
        self.ai_player = None
        self.ai_thinking_time = 5.0
        self.is_ai_thinking = False
        
        self.cell_size = 50  # Size of each board cell in pixels
        self.margin = 40  # Margin around the board
        self.stone_radius = 20  # Radius of stones
        
        self.board_color = "#DEB887"  # Tan/beige for board
        self.line_color = "#000000"  # Black for grid lines
        self.black_stone_color = "#000000"
        self.white_stone_color = "#FFFFFF"
        self.stone_outline = "#333333"
        self.star_point_color = "#000000"
        self.highlight_color = "#FF6B6B"  # Red for last move
        self.legal_move_hint = "#90EE90"  # Light green for legal move preview
        
        self.last_move = None
        self.hover_pos = None
        
        self.root = tk.Tk()
        self.root.title("Go Game - 9x9 Board")
        self.root.resizable(False, False)
        
        self.create_widgets()
        self.update_display()
    
    def create_widgets(self):
        """Create all GUI widgets."""
        main_frame = tk.Frame(self.root, bg="#F5F5DC")
        main_frame.pack(padx=10, pady=10)
        
        title_label = tk.Label(
            main_frame,
            text="Go Game with MCTS AI",
            font=("Arial", 18, "bold"),
            bg="#F5F5DC"
        )
        title_label.pack(pady=(0, 10))
        
        info_frame = tk.Frame(main_frame, bg="#F5F5DC")
        info_frame.pack(pady=5)
        
        self.info_label = tk.Label(
            info_frame,
            text="Current Player: Black (●)",
            font=("Arial", 12),
            bg="#F5F5DC"
        )
        self.info_label.pack()
        
        self.capture_label = tk.Label(
            info_frame,
            text="Captures - Black: 0  |  White: 0",
            font=("Arial", 10),
            bg="#F5F5DC"
        )
        self.capture_label.pack()
        
        self.status_label = tk.Label(
            info_frame,
            text="",
            font=("Arial", 10, "italic"),
            fg="#666666",
            bg="#F5F5DC"
        )
        self.status_label.pack()
        
        # Canvas for the board
        canvas_size = self.cell_size * (self.board_size - 1) + 2 * self.margin
        self.canvas = tk.Canvas(
            main_frame,
            width=canvas_size,
            height=canvas_size,
            bg=self.board_color,
            highlightthickness=2,
            highlightbackground="#8B4513"
        )
        self.canvas.pack(pady=10)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_board_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)
        
        # Control buttons frame
        button_frame = tk.Frame(main_frame, bg="#F5F5DC")
        button_frame.pack(pady=10)
        
        self.pass_button = tk.Button(
            button_frame,
            text="Pass",
            command=self.on_pass,
            font=("Arial", 10),
            width=10,
            bg="#FFE4B5",
            activebackground="#FFD700"
        )
        self.pass_button.grid(row=0, column=0, padx=5)
        
        self.new_game_button = tk.Button(
            button_frame,
            text="New Game",
            command=self.show_game_setup,
            font=("Arial", 10),
            width=10,
            bg="#E6E6FA",
            activebackground="#D8BFD8"
        )
        self.new_game_button.grid(row=0, column=1, padx=5)
        
        self.quit_button = tk.Button(
            button_frame,
            text="Quit",
            command=self.on_quit,
            font=("Arial", 10),
            width=10,
            bg="#FFB6C1",
            activebackground="#FF69B4"
        )
        self.quit_button.grid(row=0, column=2, padx=5)
        
        self.draw_board()
    
    def draw_board(self):
        """Draw the Go board with grid lines and star points."""
        self.canvas.delete("all")
        
        for i in range(self.board_size):
            x = self.margin + i * self.cell_size
            self.canvas.create_line(
                x, self.margin,
                x, self.margin + (self.board_size - 1) * self.cell_size,
                fill=self.line_color,
                width=2
            )
            
            y = self.margin + i * self.cell_size
            self.canvas.create_line(
                self.margin, y,
                self.margin + (self.board_size - 1) * self.cell_size, y,
                fill=self.line_color,
                width=2
            )
        
        # Draw star points (for 9x9: center and corners)
        star_points = []
        if self.board_size == 9:
            star_points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]  # Traditional 9x9 star points
        
        for row, col in star_points:
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            self.canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4,
                fill=self.star_point_color,
                outline=self.star_point_color
            )
        
        # Draw coordinate labels
        for i in range(self.board_size):
            x = self.margin + i * self.cell_size
            self.canvas.create_text(
                x, self.margin - 20,
                text=chr(65 + i),
                font=("Arial", 10, "bold")
            )
            self.canvas.create_text(
                x, self.margin + (self.board_size - 1) * self.cell_size + 20,
                text=chr(65 + i),
                font=("Arial", 10, "bold")
            )
            
            y = self.margin + i * self.cell_size
            self.canvas.create_text(
                self.margin - 20, y,
                text=str(i + 1),
                font=("Arial", 10, "bold")
            )
            self.canvas.create_text(
                self.margin + (self.board_size - 1) * self.cell_size + 20, y,
                text=str(i + 1),
                font=("Arial", 10, "bold")
            )
        
        # Draw stones
        self.draw_stones()
    
    def draw_stones(self):
        """Draw all stones on the board."""
        for row in range(self.board_size):
            for col in range(self.board_size):
                stone = self.board.get_stone(row, col)
                if stone != Player.EMPTY:
                    self.draw_stone(row, col, stone)
        
        # Highlight last move
        if self.last_move:
            row, col = self.last_move
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            self.canvas.create_oval(
                x - 8, y - 8, x + 8, y + 8,
                outline=self.highlight_color,
                width=3,
                tags="highlight"
            )
        
        # Show hover preview
        if self.hover_pos and not self.is_ai_thinking:
            row, col = self.hover_pos
            if self.board.is_legal_move(row, col):
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                color = self.black_stone_color if self.board.current_player == Player.BLACK else self.white_stone_color
                self.canvas.create_oval(
                    x - self.stone_radius, y - self.stone_radius,
                    x + self.stone_radius, y + self.stone_radius,
                    fill=color,
                    outline=self.legal_move_hint,
                    width=3,
                    stipple="gray50",
                    tags="preview"
                )
    
    def draw_stone(self, row: int, col: int, player: Player):
        """Draw a single stone at the specified position."""
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        
        color = self.black_stone_color if player == Player.BLACK else self.white_stone_color
        
        # Draw stone with shadow effect
        self.canvas.create_oval(
            x - self.stone_radius + 2, y - self.stone_radius + 2,
            x + self.stone_radius + 2, y + self.stone_radius + 2,
            fill="#888888",
            outline=""
        )
        
        self.canvas.create_oval(
            x - self.stone_radius, y - self.stone_radius,
            x + self.stone_radius, y + self.stone_radius,
            fill=color,
            outline=self.stone_outline,
            width=2
        )
        
        # Add gradient effect for white stones
        if player == Player.WHITE:
            self.canvas.create_oval(
                x - self.stone_radius + 5, y - self.stone_radius + 5,
                x - self.stone_radius + 12, y - self.stone_radius + 12,
                fill="#FFFFFF",
                outline="",
                stipple="gray25"
            )
    
    def get_board_position(self, event) -> Optional[Tuple[int, int]]:
        """Convert mouse coordinates to board position."""
        x = event.x - self.margin
        y = event.y - self.margin
        
        # Find nearest intersection
        col = round(x / self.cell_size)
        row = round(y / self.cell_size)
        
        # Check if click is close enough to an intersection
        click_x = col * self.cell_size
        click_y = row * self.cell_size
        distance = ((x - click_x) ** 2 + (y - click_y) ** 2) ** 0.5
        
        if distance <= self.stone_radius and 0 <= row < self.board_size and 0 <= col < self.board_size:
            return (row, col)
        
        return None
    
    def on_mouse_move(self, event):
        """Handle mouse movement over the board."""
        if self.is_ai_thinking:
            return
        
        pos = self.get_board_position(event)
        if pos != self.hover_pos:
            self.hover_pos = pos
            self.canvas.delete("preview")
            self.draw_stones()
    
    def on_mouse_leave(self, event):
        """Handle mouse leaving the board."""
        self.hover_pos = None
        self.canvas.delete("preview")
        self.draw_stones()
    
    def on_board_click(self, event):
        """Handle click on the board."""
        if self.is_ai_thinking:
            self.status_label.config(text="Please wait, AI is thinking...", fg="#FF6B6B")
            return
        
        pos = self.get_board_position(event)
        if pos:
            row, col = pos
            self.make_human_move(row, col)
    
    def make_human_move(self, row: int, col: int):
        """Make a move for the human player."""
        if self.board.is_legal_move(row, col):
            self.board.make_move(row, col)
            self.last_move = (row, col)
            self.update_display()
            
            # Check if game is over
            if self.board.is_game_over():
                self.show_game_over()
                return
            
            # If playing against AI, make AI move
            if self.ai_agent and self.board.current_player == self.ai_player:
                self.root.after(100, self.make_ai_move)
        else:
            self.status_label.config(text="Illegal move! Try another position.", fg="#FF0000")
            self.root.after(2000, lambda: self.status_label.config(text=""))
    
    def make_ai_move(self):
        """Make a move for the AI player."""
        self.is_ai_thinking = True
        self.status_label.config(text="AI is thinking...", fg="#0066CC")
        self.pass_button.config(state=tk.DISABLED)
        self.canvas.config(cursor="watch")
        
        def ai_thread():
            move = self.ai_agent.select_move(self.board)
            self.root.after(0, lambda: self.complete_ai_move(move))
        
        thread = threading.Thread(target=ai_thread, daemon=True)
        thread.start()
    
    def complete_ai_move(self, move: Optional[Tuple[int, int]]):
        """Complete the AI move on the main thread."""
        self.is_ai_thinking = False
        self.canvas.config(cursor="")
        self.pass_button.config(state=tk.NORMAL)
        
        if move is None:
            self.board.pass_turn()
            self.last_move = None
            self.status_label.config(text="AI passed", fg="#666666")
        else:
            row, col = move
            self.board.make_move(row, col)
            self.last_move = (row, col)
            move_str = f"{chr(65 + col)}{row + 1}"
            self.status_label.config(text=f"AI played: {move_str}", fg="#008000")
        
        self.update_display()
        
        # Check if game is over
        if self.board.is_game_over():
            self.show_game_over()
    
    def on_pass(self):
        """Handle pass button click."""
        if self.is_ai_thinking:
            return
        
        self.board.pass_turn()
        self.last_move = None
        self.status_label.config(text="You passed", fg="#666666")
        self.update_display()
        
        # Check if game is over
        if self.board.is_game_over():
            self.show_game_over()
            return
        
        # If playing against AI, make AI move
        if self.ai_agent and self.board.current_player == self.ai_player:
            self.root.after(100, self.make_ai_move)
    
    def update_display(self):
        """Update all display elements."""
        # Update info labels
        current = "Black (●)" if self.board.current_player == Player.BLACK else "White (○)"
        self.info_label.config(text=f"Current Player: {current}")
        
        black_cap = self.board.captured_stones[Player.BLACK]
        white_cap = self.board.captured_stones[Player.WHITE]
        self.capture_label.config(text=f"Captures - Black: {black_cap}  |  White: {white_cap}")
        
        # Redraw board
        self.draw_board()
    
    def show_game_over(self):
        """Display game over dialog with final score."""
        score = self.board.calculate_score()
        
        winner_text = "Black wins!" if score['winner'] == Player.BLACK else "White wins!"
        
        message = f"{winner_text}\n\n"
        message += f"Final Score:\n"
        message += f"Black: {score['black']:.1f} points\n"
        message += f"  Stones: {score['black_stones']}\n"
        message += f"  Territory: {score['black_territory']}\n"
        message += f"  Captures: {score['black_captures']}\n\n"
        message += f"White: {score['white']:.1f} points\n"
        message += f"  Stones: {score['white_stones']}\n"
        message += f"  Territory: {score['white_territory']}\n"
        message += f"  Captures: {score['white_captures']}\n"
        message += f"  Komi: 6.5"
        
        result = messagebox.askquestion(
            "Game Over",
            message + "\n\nPlay again?",
            icon='info'
        )
        
        if result == 'yes':
            self.show_game_setup()
        else:
            self.on_quit()
    
    def show_game_setup(self):
        """Show game setup dialog."""
        setup_window = tk.Toplevel(self.root)
        setup_window.title("New Game Setup")
        setup_window.geometry("400x320")
        setup_window.resizable(False, False)
        setup_window.configure(bg="#F5F5DC")
        
        # Center the window
        setup_window.transient(self.root)
        setup_window.grab_set()
        
        tk.Label(
            setup_window,
            text="Select Game Mode",
            font=("Arial", 14, "bold"),
            bg="#F5F5DC"
        ).pack(pady=20)
        
        mode_var = tk.StringVar(value="human_vs_ai_black")
        
        modes = [
            ("Play as Black vs AI", "human_vs_ai_black"),
            ("Play as White vs AI", "human_vs_ai_white"),
            ("Human vs Human", "human_vs_human"),
        ]
        
        for text, value in modes:
            tk.Radiobutton(
                setup_window,
                text=text,
                variable=mode_var,
                value=value,
                font=("Arial", 11),
                bg="#F5F5DC"
            ).pack(anchor=tk.W, padx=40, pady=5)
        
        tk.Label(
            setup_window,
            text="AI Thinking Time (seconds):",
            font=("Arial", 10),
            bg="#F5F5DC"
        ).pack(pady=(15, 5))
        
        time_var = tk.StringVar(value="5")
        time_entry = tk.Entry(setup_window, textvariable=time_var, width=10, font=("Arial", 10))
        time_entry.pack(pady=(0, 10))
        
        # Separator line
        separator = tk.Frame(setup_window, height=2, bg="#999999")
        separator.pack(fill=tk.X, padx=30, pady=10)
        
        def start_game():
            try:
                think_time = float(time_var.get())
                if think_time < 0.5:
                    think_time = 0.5
                elif think_time > 60:
                    think_time = 60
            except ValueError:
                think_time = 5.0
            
            mode = mode_var.get()
            setup_window.destroy()
            self.start_new_game(mode, think_time)
        
        start_button = tk.Button(
            setup_window,
            text="▶ Start Game",
            command=start_game,
            font=("Arial", 12, "bold"),
            bg="#90EE90",
            activebackground="#7CFC00",
            width=18,
            height=2,
            cursor="hand2"
        )
        start_button.pack(pady=15)
    
    def start_new_game(self, mode: str, ai_time: float):
        """Start a new game with specified settings."""
        self.board = GoBoard(self.board_size)
        self.last_move = None
        self.hover_pos = None
        self.ai_thinking_time = ai_time
        
        if mode == "human_vs_ai_black":
            self.ai_player = Player.WHITE
            self.ai_agent = MCTSAgent(simulation_time=ai_time)
            self.status_label.config(text="You are Black (●). AI is White (○).", fg="#0066CC")
        elif mode == "human_vs_ai_white":
            self.ai_player = Player.BLACK
            self.ai_agent = MCTSAgent(simulation_time=ai_time)
            self.status_label.config(text="You are White (○). AI is Black (●).", fg="#0066CC")
            # AI makes first move
            self.root.after(500, self.make_ai_move)
        else:  # human_vs_human
            self.ai_player = None
            self.ai_agent = None
            self.status_label.config(text="Human vs Human mode", fg="#0066CC")
        
        self.update_display()
    
    def on_quit(self):
        """Handle quit action."""
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the GUI main loop."""
        # Show game setup on start
        self.root.after(100, self.show_game_setup)
        self.root.mainloop()


def main():
    """Main entry point for the GUI."""
    gui = GoGUI(board_size=9)
    gui.run()


if __name__ == "__main__":
    main()
