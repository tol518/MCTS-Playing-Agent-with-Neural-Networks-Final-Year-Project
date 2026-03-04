from game_gui import main
from profiler_utils import run_with_profiler

if __name__ == "__main__":
    print("Starting Go Game GUI...")
    run_with_profiler("play_gui", main)
