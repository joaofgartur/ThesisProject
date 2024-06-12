import argparse
import os


def execute_command(num_sessions: int, command: str):
    for i in range(num_sessions):
        session_name = f'session_{i}'
        command = f"tmux new-session -d -s {session_name} {command}"
        os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for the thesis.')
    parser.add_argument('-num_sessions', type=int, default='1', help='Number of parallel processes.')
    parser.add_argument('-command', type=str, default='echo "Hello, World!"', help='Command to execute.')
    args = parser.parse_args()

    num_sessions = args.num_sessions
    python_command = args.command

    execute_command(num_sessions, python_command)

