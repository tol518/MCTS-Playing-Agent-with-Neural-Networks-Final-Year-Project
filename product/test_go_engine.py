from go_engine import GoBoard, Player


def test_board_initialization():
    """Test that board initializes correctly."""
    board = GoBoard(9)
    assert board.size == 9
    assert board.current_player == Player.BLACK
    print("✓ Board initialization test passed")


def test_stone_placement():
    """Test basic stone placement."""
    board = GoBoard(9)
    
    # Place a black stone
    assert board.make_move(3, 3, Player.BLACK) == True
    assert board.get_stone(3, 3) == Player.BLACK
    
    # Try to place on occupied position
    assert board.make_move(3, 3, Player.WHITE) == False
    
    print("Stone placement test passed")


def test_capture():
    """Test stone capture logic."""
    board = GoBoard(9)
    
    # Create a capture scenario
    # Black stones surrounding a white stone at (4, 4)
    board.make_move(4, 4, Player.WHITE)  # White stone
    board.make_move(4, 3, Player.BLACK)  # Black surrounding
    board.make_move(4, 5, Player.BLACK)
    board.make_move(3, 4, Player.BLACK)
    
    # This move should capture the white stone
    board.make_move(5, 4, Player.BLACK)
    
    assert board.get_stone(4, 4) == Player.EMPTY
    assert board.captured_stones[Player.BLACK] == 1
    
    print("Capture test passed")


def test_liberties():
    """Test liberty counting."""
    board = GoBoard(9)
    
    # Place a stone in the middle
    board.set_stone(4, 4, Player.BLACK)
    assert board.count_liberties(4, 4) == 4  # 4 adjacent empty points
    
    # Place a stone in the corner
    board.set_stone(0, 0, Player.WHITE)
    assert board.count_liberties(0, 0) == 2  # 2 adjacent empty points
    
    # Place adjacent stones
    board.set_stone(4, 5, Player.BLACK)
    liberties = board.count_liberties(4, 4)
    assert liberties == 6  # Both stones form a group with 6 liberties
    
    print("Liberty counting test passed")


def test_group_detection():
    """Test connected group detection."""
    board = GoBoard(9)
    
    # Create a connected group
    board.set_stone(4, 4, Player.BLACK)
    board.set_stone(4, 5, Player.BLACK)
    board.set_stone(5, 4, Player.BLACK)
    
    group = board.get_group(4, 4)
    assert len(group) == 3
    assert (4, 4) in group
    assert (4, 5) in group
    assert (5, 4) in group
    
    print("Group detection test passed")


def test_suicide_rule():
    """Test that suicide moves are prevented."""
    board = GoBoard(9)
    
    # Create a situation where a move would be suicide
    # White stones surrounding position (4, 4)
    board.set_stone(4, 3, Player.WHITE)
    board.set_stone(4, 5, Player.WHITE)
    board.set_stone(3, 4, Player.WHITE)
    board.set_stone(5, 4, Player.WHITE)
    
    # Black cannot play at (4, 4) - it would be suicide
    assert board.is_legal_move(4, 4, Player.BLACK) == False
    
    print(" Suicide rule test passed")


def test_legal_moves():
    """Test legal move detection."""
    board = GoBoard(9)
    
    # Initially, all positions should be legal
    legal_moves = board.get_legal_moves(Player.BLACK)
    assert len(legal_moves) == 81  # 9x9 board
    
    # Place some stones
    board.make_move(4, 4, Player.BLACK)
    legal_moves = board.get_legal_moves(Player.WHITE)
    assert len(legal_moves) == 80
    assert (4, 4) not in legal_moves
    
    print("Legal moves test passed")


def test_pass():
    """Test pass functionality."""
    board = GoBoard(9)
    
    board.pass_turn()
    assert board.current_player == Player.WHITE
    assert board.pass_count == 1
    
    board.pass_turn()
    assert board.current_player == Player.BLACK
    assert board.pass_count == 2
    assert board.is_game_over() == True
    
    print("Pass test passed")


def test_scoring():
    """Test scoring calculation."""
    board = GoBoard(9)
    
    # Create a simple position
    # Black controls top-left corner
    for i in range(3):
        for j in range(3):
            if i < 2 or j < 2:
                board.set_stone(i, j, Player.BLACK)
    
    # White controls bottom-right corner
    for i in range(6, 9):
        for j in range(6, 9):
            if i > 6 or j > 6:
                board.set_stone(i, j, Player.WHITE)
    
    score = board.calculate_score()
    
    assert 'black' in score
    assert 'white' in score
    assert 'winner' in score
    assert score['black'] > 0
    assert score['white'] > 0
    
    print("Scoring test passed")


def test_board_clone():
    """Test board cloning."""
    board = GoBoard(9)
    board.make_move(4, 4, Player.BLACK)
    board.make_move(5, 5, Player.WHITE)
    
    cloned = board.clone()
    
    assert cloned.get_stone(4, 4) == Player.BLACK
    assert cloned.get_stone(5, 5) == Player.WHITE
    assert cloned.current_player == board.current_player
    
    # Modify clone and ensure original is unchanged
    cloned.make_move(6, 6, Player.BLACK)
    assert board.get_stone(6, 6) == Player.EMPTY
    assert cloned.get_stone(6, 6) == Player.BLACK
    
    print("Board cloning test passed")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 50)
    print("Running Go Engine Tests")
    print("=" * 50 + "\n")
    
    test_board_initialization()
    test_stone_placement()
    test_capture()
    test_liberties()
    test_group_detection()
    test_suicide_rule()
    test_legal_moves()
    test_pass()
    test_scoring()
    test_board_clone()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_all_tests()
