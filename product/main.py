from profiler_utils import run_with_profiler


def main():
    print("=" * 50)
    print("  Welcome to Go-Playing AI with MCTS!")
    print("  9x9 Board Implementation")
    print("=" * 50)
    print("\nSelect interface:")
    print("1. Graphical Interface (GUI) - Recommended")
    print("2. Terminal Interface (Text-based)")
    
    choice = input("\nEnter your choice (1 or 2, default 1): ").strip()
    
    if choice == "2":
        print("\nStarting terminal interface...")
        from game_ui import main
        main()
    else:
        print("\nStarting graphical interface...")
        try:
            from game_gui import main
            main()
        except ImportError as e:
            print(f"\nError loading GUI: {e}")
            print("Falling back to terminal interface...")
            from game_ui import main
            main()


if __name__ == "__main__":
    run_with_profiler("main", main)
