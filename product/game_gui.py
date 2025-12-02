"""
Graphical User Interface for Go game using tkinter.
Provides a 2D visual board with clickable intersections and proper coloring.
"""

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional, Tuple
from go_engine import GoBoard, Player, column_to_label
from mcts_agent import MCTSAgent
import threading


class GoGUI:
    """
    Modern graphical Go game interface with dark theme.
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
        
        self.cell_size = 52  # Size of each board cell in pixels
        self.margin = 45  # Margin around the board
        self.stone_radius = 22  # Radius of stones
        
        # Theme definitions
        self.themes = {
            'light': {
                'bg_dark': "#f0f0f0",
                'bg_medium': "#ffffff",
                'bg_light': "#e8e8e8",
                'accent': "#e94560",
                'accent_2': "#0088cc",
                'text_primary': "#1a1a1a",
                'text_secondary': "#666666",
                'button_bg': "#d0d0d0",
                'button_hover': "#c0c0c0",
            },
            'dark': {
                'bg_dark': "#1a1a2e",
                'bg_medium': "#16213e",
                'bg_light': "#0f3460",
                'accent': "#e94560",
                'accent_2': "#00d9ff",
                'text_primary': "#ffffff",
                'text_secondary': "#a0a0a0",
                'button_bg': "#0f3460",
                'button_hover': "#1a4a7a",
            }
        }
        
        # Detect system theme and use as default
        self.current_theme = self._detect_system_theme()
        self._apply_theme(self.current_theme)
        
        # Board colors (same for both themes)
        self.board_color = "#c4a35a"  # Warm wood tone
        self.board_dark = "#a08040"   # Darker wood for depth
        self.line_color = "#2c2416"   # Dark brown lines
        self.black_stone_color = "#1a1a1a"
        self.black_stone_highlight = "#404040"
        self.white_stone_color = "#f5f5f5"
        self.white_stone_shadow = "#c0c0c0"
        self.stone_outline = "#333333"
        self.star_point_color = "#2c2416"
        self.highlight_color = "#e94560"  # Accent for last move
        self.legal_move_hint = "#0088cc"  # Blue for preview
        
        self.last_move = None
        self.hover_pos = None
        
        self.root = tk.Tk()
        self.root.title("Go Game")
        self.root.resizable(False, False)
        self.root.configure(bg=self.bg_dark)
        
        self.create_widgets()
        self.update_display()
    
    def _detect_system_theme(self) -> str:
        """Detect system dark/light mode preference."""
        try:
            # Windows: Check registry for dark mode setting
            import winreg
            registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
            key = winreg.OpenKey(registry, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            winreg.CloseKey(key)
            return 'light' if value == 1 else 'dark'
        except:
            pass
        
        try:
            # macOS: Check defaults
            import subprocess
            result = subprocess.run(
                ['defaults', 'read', '-g', 'AppleInterfaceStyle'],
                capture_output=True, text=True
            )
            if 'Dark' in result.stdout:
                return 'dark'
            return 'light'
        except:
            pass
        
        # Default to light if detection fails
        return 'light'
    
    def _apply_theme(self, theme_name: str):
        """Apply a theme's colors."""
        theme = self.themes[theme_name]
        self.current_theme = theme_name
        self.bg_dark = theme['bg_dark']
        self.bg_medium = theme['bg_medium']
        self.bg_light = theme['bg_light']
        self.accent = theme['accent']
        self.accent_2 = theme['accent_2']
        self.text_primary = theme['text_primary']
        self.text_secondary = theme['text_secondary']
        self.button_bg = theme['button_bg']
        self.button_hover = theme['button_hover']
        self.legal_move_hint = theme['accent_2']
    
    def create_widgets(self):
        """Create all GUI widgets with modern styling."""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_dark)
        main_frame.pack(padx=20, pady=20)
        
        # Header with title
        header_frame = tk.Frame(main_frame, bg=self.bg_dark)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(
            header_frame,
            text="⚫ GO GAME ⚪",
            font=("Segoe UI", 24, "bold"),
            fg=self.text_primary,
            bg=self.bg_dark
        )
        title_label.pack()
        
        subtitle = tk.Label(
            header_frame,
            text="Monte Carlo Tree Search AI",
            font=("Segoe UI", 10),
            fg=self.text_secondary,
            bg=self.bg_dark
        )
        subtitle.pack()
        
        # Info panel
        info_panel = tk.Frame(main_frame, bg=self.bg_medium, padx=15, pady=10)
        info_panel.pack(fill=tk.X, pady=(0, 15))
        
        # Player indicator with colored dot
        player_frame = tk.Frame(info_panel, bg=self.bg_medium)
        player_frame.pack(fill=tk.X)
        
        self.player_indicator = tk.Canvas(player_frame, width=20, height=20, bg=self.bg_medium, highlightthickness=0)
        self.player_indicator.pack(side=tk.LEFT, padx=(0, 10))
        self.player_indicator.create_oval(2, 2, 18, 18, fill=self.black_stone_color, outline="#555")
        
        self.info_label = tk.Label(
            player_frame,
            text="Current Player: Black",
            font=("Segoe UI", 12, "bold"),
            fg=self.text_primary,
            bg=self.bg_medium
        )
        self.info_label.pack(side=tk.LEFT)
        
        # Captures display
        captures_frame = tk.Frame(info_panel, bg=self.bg_medium)
        captures_frame.pack(fill=tk.X, pady=(8, 0))
        
        self.black_cap_label = tk.Label(
            captures_frame,
            text="⚫ Captured: 0",
            font=("Segoe UI", 10),
            fg=self.text_secondary,
            bg=self.bg_medium
        )
        self.black_cap_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.white_cap_label = tk.Label(
            captures_frame,
            text="⚪ Captured: 0",
            font=("Segoe UI", 10),
            fg=self.text_secondary,
            bg=self.bg_medium
        )
        self.white_cap_label.pack(side=tk.LEFT)
        
        # Status label
        self.status_label = tk.Label(
            info_panel,
            text="",
            font=("Segoe UI", 10),
            fg=self.accent_2,
            bg=self.bg_medium
        )
        self.status_label.pack(fill=tk.X, pady=(8, 0))
        
        # Board frame with shadow effect
        board_outer = tk.Frame(main_frame, bg=self.bg_light, padx=3, pady=3)
        board_outer.pack()
        
        canvas_size = self.cell_size * (self.board_size - 1) + 2 * self.margin
        self.canvas = tk.Canvas(
            board_outer,
            width=canvas_size,
            height=canvas_size,
            bg=self.board_color,
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_board_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)
        
        # Control buttons frame
        button_frame = tk.Frame(main_frame, bg=self.bg_dark)
        button_frame.pack(pady=(20, 0))
        
        # Modern button style - use padx instead of fixed width so text fits
        btn_config = {
            'font': ("Segoe UI", 10),
            'height': 1,
            'bd': 0,
            'cursor': 'hand2',
            'relief': tk.FLAT,
            'padx': 12,
            'pady': 6,
        }
        
        # undo button
        self.undo_button = tk.Button(
            button_frame,
            text="↩ Undo",
            command=self.on_undo,
            bg=self.button_bg,
            fg=self.text_primary,
            activebackground=self.bg_medium,
            activeforeground=self.text_primary,
            **btn_config
        )
        self.undo_button.grid(row=0, column=0, padx=4)
        self._add_button_hover(self.undo_button, self.button_bg, self.button_hover)
        
        # redo button
        self.redo_button = tk.Button(
            button_frame,
            text="↪ Redo",
            command=self.on_redo,
            bg=self.button_bg,
            fg=self.text_primary,
            activebackground=self.bg_medium,
            activeforeground=self.text_primary,
            **btn_config
        )
        self.redo_button.grid(row=0, column=1, padx=4)
        self._add_button_hover(self.redo_button, self.button_bg, self.button_hover)
        
        self.pass_button = tk.Button(
            button_frame,
            text="⏭ Pass",
            command=self.on_pass,
            bg=self.button_bg,
            fg=self.text_primary,
            activebackground=self.bg_medium,
            activeforeground=self.text_primary,
            **btn_config
        )
        self.pass_button.grid(row=0, column=2, padx=4)
        self._add_button_hover(self.pass_button, self.button_bg, self.button_hover)
        
        self.new_game_button = tk.Button(
            button_frame,
            text="🎮 New",
            command=self.show_game_setup,
            bg=self.accent_2,
            fg=self.bg_dark,
            activebackground="#00b8d9",
            activeforeground=self.bg_dark,
            **btn_config
        )
        self.new_game_button.grid(row=0, column=3, padx=4)
        self._add_button_hover(self.new_game_button, self.accent_2, "#00b8d9")
        
        self.quit_button = tk.Button(
            button_frame,
            text="✕ Quit",
            command=self.on_quit,
            bg=self.accent,
            fg=self.text_primary,
            activebackground="#c73e54",
            activeforeground=self.text_primary,
            **btn_config
        )
        self.quit_button.grid(row=0, column=4, padx=4)
        self._add_button_hover(self.quit_button, self.accent, "#c73e54")
        
        self.draw_board()
    
    def _add_button_hover(self, button, normal_color, hover_color):
        """Add hover effect to button."""
        def on_enter(e):
            button.config(bg=hover_color)
        def on_leave(e):
            button.config(bg=normal_color)
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
    
    def draw_board(self):
        """Draw the Go board with grid lines and star points."""
        self.canvas.delete("all")
        
        # Draw wood grain texture effect (subtle gradient)
        for i in range(0, self.canvas.winfo_reqwidth(), 20):
            shade = "#c9a85e" if (i // 20) % 2 == 0 else "#c4a35a"
            self.canvas.create_line(i, 0, i, self.canvas.winfo_reqheight(), fill=shade, width=20)
        
        # Draw border/frame effect
        self.canvas.create_rectangle(
            self.margin - 15, self.margin - 15,
            self.margin + (self.board_size - 1) * self.cell_size + 15,
            self.margin + (self.board_size - 1) * self.cell_size + 15,
            fill=self.board_dark,
            outline=""
        )
        self.canvas.create_rectangle(
            self.margin - 10, self.margin - 10,
            self.margin + (self.board_size - 1) * self.cell_size + 10,
            self.margin + (self.board_size - 1) * self.cell_size + 10,
            fill=self.board_color,
            outline=""
        )
        
        # Draw grid lines
        for i in range(self.board_size):
            x = self.margin + i * self.cell_size
            # Vertical lines
            self.canvas.create_line(
                x, self.margin,
                x, self.margin + (self.board_size - 1) * self.cell_size,
                fill=self.line_color,
                width=1 if i != 0 and i != self.board_size - 1 else 2
            )
            
            y = self.margin + i * self.cell_size
            # Horizontal lines
            self.canvas.create_line(
                self.margin, y,
                self.margin + (self.board_size - 1) * self.cell_size, y,
                fill=self.line_color,
                width=1 if i != 0 and i != self.board_size - 1 else 2
            )
        
        # Draw star points (for 9x9: center and corners)
        star_points = []
        if self.board_size == 9:
            star_points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
        
        for row, col in star_points:
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            self.canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5,
                fill=self.star_point_color,
                outline=""
            )
        
        # Draw coordinate labels with better styling
        for i in range(self.board_size):
            label = column_to_label(i)
            x = self.margin + i * self.cell_size
            # Top labels
            self.canvas.create_text(
                x, self.margin - 25,
                text=label,
                font=("Segoe UI", 9, "bold"),
                fill=self.line_color
            )
            # Bottom labels
            self.canvas.create_text(
                x, self.margin + (self.board_size - 1) * self.cell_size + 25,
                text=label,
                font=("Segoe UI", 9, "bold"),
                fill=self.line_color
            )
            
            y = self.margin + i * self.cell_size
            # Left labels
            self.canvas.create_text(
                self.margin - 25, y,
                text=str(i + 1),
                font=("Segoe UI", 9, "bold"),
                fill=self.line_color
            )
            # Right labels
            self.canvas.create_text(
                self.margin + (self.board_size - 1) * self.cell_size + 25, y,
                text=str(i + 1),
                font=("Segoe UI", 9, "bold"),
                fill=self.line_color
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
        """Draw a single stone at the specified position - modern flat design."""
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        r = self.stone_radius
        
        if player == Player.BLACK:
            # Subtle shadow
            self.canvas.create_oval(
                x - r + 2, y - r + 2, x + r + 2, y + r + 2,
                fill="#2a2a2a", outline=""
            )
            # Main stone - solid flat black
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill="#1a1a1a", outline="#0a0a0a", width=1
            )
        else:  # WHITE
            # Subtle shadow
            self.canvas.create_oval(
                x - r + 2, y - r + 2, x + r + 2, y + r + 2,
                fill="#888888", outline=""
            )
            # Main stone - solid flat white
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill="#f0f0f0", outline="#aaaaaa", width=1
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
            self.status_label.config(text="❌ Illegal move! Try another position.", fg=self.accent)
            self.root.after(2000, lambda: self.status_label.config(text=""))
    
    def make_ai_move(self):
        """Make a move for the AI player."""
        self.is_ai_thinking = True
        self.status_label.config(text="🤔 AI is thinking...", fg=self.accent)
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
            self.status_label.config(text="⏭ AI passed", fg=self.text_secondary)
        else:
            row, col = move
            self.board.make_move(row, col)
            self.last_move = (row, col)
            move_str = f"{column_to_label(col)}{row + 1}"
            self.status_label.config(text=f"🎯 AI played: {move_str}", fg=self.accent_2)
        
        self.update_display()
        
        # Check if game is over
        if self.board.is_game_over():
            self.show_game_over()
    
    def on_undo(self):
        """go back one move"""
        if self.is_ai_thinking:
            return
        
        if self.board.undo():
            # update last_move to show the previous move marker
            if self.board.move_history:
                last = self.board.move_history[-1]
                if last[0] >= 0:  # not a pass
                    self.last_move = (last[0], last[1])
                else:
                    self.last_move = None
            else:
                self.last_move = None
            self.status_label.config(text="↩ Undid last move", fg=self.text_secondary)
            self.update_display()
        else:
            self.status_label.config(text="❌ Nothing to undo", fg=self.accent)
    
    def on_redo(self):
        """go forward one move"""
        if self.is_ai_thinking:
            return
        
        if self.board.redo():
            # update last_move marker
            if self.board.move_history:
                last = self.board.move_history[-1]
                if last[0] >= 0:
                    self.last_move = (last[0], last[1])
                else:
                    self.last_move = None
            else:
                self.last_move = None
            self.status_label.config(text="↪ Redid move", fg=self.text_secondary)
            self.update_display()
        else:
            self.status_label.config(text="❌ Nothing to redo", fg=self.accent)
    
    def on_pass(self):
        """Handle pass button click."""
        if self.is_ai_thinking:
            return
        
        self.board.pass_turn()
        self.last_move = None
        self.status_label.config(text="⏭ You passed", fg=self.text_secondary)
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
        # Update player indicator
        is_black = self.board.current_player == Player.BLACK
        self.info_label.config(text=f"Current Player: {'Black' if is_black else 'White'}")
        
        # Update indicator dot
        self.player_indicator.delete("all")
        color = self.black_stone_color if is_black else self.white_stone_color
        outline = "#555" if is_black else "#aaa"
        self.player_indicator.create_oval(2, 2, 18, 18, fill=color, outline=outline, width=2)
        
        # Update capture labels
        black_cap = self.board.captured_stones[Player.BLACK]
        white_cap = self.board.captured_stones[Player.WHITE]
        self.black_cap_label.config(text=f"⚫ Captured: {black_cap}")
        self.white_cap_label.config(text=f"⚪ Captured: {white_cap}")
        
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
        """Show modern game setup dialog."""
        setup_window = tk.Toplevel(self.root)
        setup_window.title("New Game")
        setup_window.geometry("420x540")
        setup_window.resizable(False, False)
        setup_window.configure(bg=self.bg_dark)
        
        # Center the window
        setup_window.transient(self.root)
        setup_window.grab_set()
        
        # Try dark title bar
        try:
            setup_window.update()
            from ctypes import windll, byref, sizeof, c_int
            hwnd = windll.user32.GetParent(setup_window.winfo_id())
            windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, byref(c_int(1)), sizeof(c_int))
        except:
            pass
        
        # Title
        tk.Label(
            setup_window,
            text="🎮 NEW GAME",
            font=("Segoe UI", 18, "bold"),
            fg=self.text_primary,
            bg=self.bg_dark
        ).pack(pady=(25, 5))
        
        tk.Label(
            setup_window,
            text="Select your game mode",
            font=("Segoe UI", 10),
            fg=self.text_secondary,
            bg=self.bg_dark
        ).pack(pady=(0, 15))
        
        # Theme selection panel
        theme_panel = tk.Frame(setup_window, bg=self.bg_medium, padx=20, pady=10)
        theme_panel.pack(fill=tk.X, padx=30)
        
        tk.Label(
            theme_panel,
            text="🎨 Appearance",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_primary,
            bg=self.bg_medium
        ).pack(side=tk.LEFT)
        
        theme_var = tk.StringVar(value=self.current_theme)
        
        theme_frame = tk.Frame(theme_panel, bg=self.bg_medium)
        theme_frame.pack(side=tk.RIGHT)
        
        light_rb = tk.Radiobutton(
            theme_frame,
            text="☀ Light",
            variable=theme_var,
            value="light",
            font=("Segoe UI", 10),
            fg=self.text_primary,
            bg=self.bg_medium,
            activebackground=self.bg_medium,
            selectcolor=self.bg_light,
            cursor="hand2"
        )
        light_rb.pack(side=tk.LEFT, padx=(0, 10))
        
        dark_rb = tk.Radiobutton(
            theme_frame,
            text="🌙 Dark",
            variable=theme_var,
            value="dark",
            font=("Segoe UI", 10),
            fg=self.text_primary,
            bg=self.bg_medium,
            activebackground=self.bg_medium,
            selectcolor=self.bg_light,
            cursor="hand2"
        )
        dark_rb.pack(side=tk.LEFT)
        
        # Mode selection panel
        mode_panel = tk.Frame(setup_window, bg=self.bg_medium, padx=20, pady=15)
        mode_panel.pack(fill=tk.X, padx=30, pady=(10, 0))
        
        mode_var = tk.StringVar(value="human_vs_ai_black")
        
        modes = [
            ("⚫ Play as Black vs AI", "human_vs_ai_black"),
            ("⚪ Play as White vs AI", "human_vs_ai_white"),
            ("👥 Human vs Human", "human_vs_human"),
        ]
        
        for text, value in modes:
            rb = tk.Radiobutton(
                mode_panel,
                text=text,
                variable=mode_var,
                value=value,
                font=("Segoe UI", 11),
                fg=self.text_primary,
                bg=self.bg_medium,
                activebackground=self.bg_medium,
                activeforeground=self.accent_2,
                selectcolor=self.bg_light,
                cursor="hand2"
            )
            rb.pack(anchor=tk.W, pady=6)
        
        # AI settings panel
        ai_panel = tk.Frame(setup_window, bg=self.bg_medium, padx=20, pady=15)
        ai_panel.pack(fill=tk.X, padx=30, pady=(15, 0))
        
        tk.Label(
            ai_panel,
            text="⏱ AI Thinking Time",
            font=("Segoe UI", 11, "bold"),
            fg=self.text_primary,
            bg=self.bg_medium
        ).pack(anchor=tk.W)
        
        time_frame = tk.Frame(ai_panel, bg=self.bg_medium)
        time_frame.pack(fill=tk.X, pady=(10, 0))
        
        time_var = tk.StringVar(value="5")
        time_entry = tk.Entry(
            time_frame, 
            textvariable=time_var, 
            width=8, 
            font=("Segoe UI", 11),
            bg=self.bg_light,
            fg=self.text_primary,
            insertbackground=self.text_primary,
            relief=tk.FLAT,
            justify=tk.CENTER
        )
        time_entry.pack(side=tk.LEFT)
        
        tk.Label(
            time_frame,
            text="seconds (0.5 - 60)",
            font=("Segoe UI", 10),
            fg=self.text_secondary,
            bg=self.bg_medium
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        def start_game():
            try:
                think_time = float(time_var.get())
                think_time = max(0.5, min(60, think_time))
            except ValueError:
                think_time = 5.0
            
            # Apply selected theme
            selected_theme = theme_var.get()
            if selected_theme != self.current_theme:
                self._apply_theme(selected_theme)
                self._rebuild_ui()
            
            mode = mode_var.get()
            setup_window.destroy()
            self.start_new_game(mode, think_time)
        
        # Start button
        start_button = tk.Button(
            setup_window,
            text="▶  START GAME",
            command=start_game,
            font=("Segoe UI", 13, "bold"),
            bg=self.accent_2,
            fg=self.bg_dark,
            activebackground="#00b8d9",
            activeforeground=self.bg_dark,
            width=20,
            height=2,
            bd=0,
            cursor="hand2",
            relief=tk.FLAT
        )
        start_button.pack(pady=25)
        self._add_button_hover(start_button, self.accent_2, "#00b8d9")
    
    def start_new_game(self, mode: str, ai_time: float):
        """Start a new game with specified settings."""
        self.board = GoBoard(self.board_size)
        self.last_move = None
        self.hover_pos = None
        self.ai_thinking_time = ai_time
        
        if mode == "human_vs_ai_black":
            self.ai_player = Player.WHITE
            self.ai_agent = MCTSAgent(simulation_time=ai_time)
            self.status_label.config(text="You are Black ⚫  •  AI is White ⚪", fg=self.accent_2)
        elif mode == "human_vs_ai_white":
            self.ai_player = Player.BLACK
            self.ai_agent = MCTSAgent(simulation_time=ai_time)
            self.status_label.config(text="You are White ⚪  •  AI is Black ⚫", fg=self.accent_2)
            # AI makes first move
            self.root.after(500, self.make_ai_move)
        else:  # human_vs_human
            self.ai_player = None
            self.ai_agent = None
            self.status_label.config(text="👥 Human vs Human mode", fg=self.accent_2)
        
        self.update_display()
    
    def _rebuild_ui(self):
        """Rebuild the UI with new theme colors."""
        # Update root background
        self.root.configure(bg=self.bg_dark)
        
        # Destroy and recreate all widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.create_widgets()
        self.update_display()
        
        # Apply dark title bar if dark theme on Windows
        if self.current_theme == 'dark':
            try:
                self.root.update()
                from ctypes import windll, byref, sizeof, c_int
                hwnd = windll.user32.GetParent(self.root.winfo_id())
                windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, byref(c_int(1)), sizeof(c_int))
            except:
                pass
        else:
            try:
                self.root.update()
                from ctypes import windll, byref, sizeof, c_int
                hwnd = windll.user32.GetParent(self.root.winfo_id())
                windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, byref(c_int(0)), sizeof(c_int))
            except:
                pass
    
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
