import numpy as np
from copy import deepcopy
import traceback
from agents import *
from ui import UIEngine
from time import sleep, time
import click
import logging
from store import AGENT_REGISTRY
from constants import *
import sys
from helpers import count_capture, execute_move, check_endgame, random_move, get_valid_moves

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)

class World:
    def __init__(
        self,
        player_1="random_agent",
        player_2="random_agent",
        board_size=None,
        display_ui=False,
        display_delay=0.4,
        display_save=False,
        display_save_path=None,
        autoplay=False,
        autoplay_runs=1,  # Number of autoplay games
    ):
        logger.info("Initialize the game world")
        self.player_1_name = player_1
        self.player_2_name = player_2
        if player_1 not in AGENT_REGISTRY:
            raise ValueError(
                f"Agent '{player_1}' is not registered. {AGENT_NOT_FOUND_MSG}"
            )
        if player_2 not in AGENT_REGISTRY:
            raise ValueError(
                f"Agent '{player_2}' is not registered. {AGENT_NOT_FOUND_MSG}"
            )

        p0_agent = AGENT_REGISTRY[player_1]
        p1_agent = AGENT_REGISTRY[player_2]
        logger.info(f"Registering p0 agent : {player_1}")
        self.p0 = p0_agent()
        logger.info(f"Registering p1 agent : {player_2}")
        self.p1 = p1_agent()

        self.autoplay = autoplay
        self.autoplay_runs = autoplay_runs
        self.current_game = 0
        self.player_1_wins = 0
        self.player_2_wins = 0
        self.draws = 0

        if autoplay:
            if not self.p0.autoplay or not self.p1.autoplay:
                raise ValueError(
                    f"Autoplay mode is not supported by one of the agents ({self.p0} -> {self.p0.autoplay}, {self.p1} -> {self.p1.autoplay}). Please set autoplay=True in the agent class."
                )

        self.player_names = {PLAYER_1_ID: PLAYER_1_NAME, PLAYER_2_ID: PLAYER_2_NAME}

        if board_size is None:
            self.board_size = np.random.choice([6, 8, 10, 12])
            logger.info(
                f"No board size specified. Randomly generating size: {self.board_size}x{self.board_size}"
            )
        else:
            self.board_size = board_size
            logger.info(f"Setting board size to {self.board_size}x{self.board_size}")

        self.chess_board = np.zeros((self.board_size, self.board_size), dtype=int)
        mid = self.board_size // 2
        self.chess_board[mid - 1][mid - 1] = 2
        self.chess_board[mid - 1][mid] = 1
        self.chess_board[mid][mid - 1] = 1
        self.chess_board[mid][mid] = 2

        self.turn = 0
        self.p0_time = []
        self.p1_time = []
        self.results_cache = ()
        self.display_ui = display_ui
        self.display_delay = display_delay
        self.display_save = display_save
        self.display_save_path = display_save_path

        if display_ui:
            logger.info(
                f"Initializing the UI Engine, with display_delay={display_delay} seconds"
            )
            self.ui_engine = UIEngine(self.board_size, self)
            self.render()

    def reset_board(self):
        """
        Reset the game board for a new game.
        """
        self.chess_board = np.zeros((self.board_size, self.board_size), dtype=int)
        mid = self.board_size // 2
        self.chess_board[mid - 1][mid - 1] = 2
        self.chess_board[mid - 1][mid] = 1
        self.chess_board[mid][mid - 1] = 1
        self.chess_board[mid][mid] = 2
        self.turn = 0
        self.p0_time = []
        self.p1_time = []
        self.results_cache = ()

    def get_current_player(self):
        return 1 if self.turn == 0 else 2

    def get_current_opponent(self):
        return 2 if self.turn == 0 else 1

    def update_player_time(self, time_taken):
        if not self.turn:
            self.p0_time.append(time_taken)
        else:
            self.p1_time.append(time_taken)

    def step(self):
        cur_player = self.get_current_player()
        opponent = self.get_current_opponent()
        valid_moves = get_valid_moves(self.chess_board, cur_player)

        if not valid_moves:
            logger.info(
                f"Player {self.player_names[self.turn]} must pass due to having no valid moves."
            )
        else:
            time_taken = None
            try:
                start_time = time()
                move_pos = self.get_current_agent().step(
                    deepcopy(self.chess_board),
                    cur_player,
                    opponent,
                )
                time_taken = time() - start_time
                self.update_player_time(time_taken)
                if count_capture(self.chess_board, move_pos, cur_player) == 0:
                    raise ValueError(f"Invalid move by player {cur_player}: {move_pos}")
            except BaseException as e:
                ex_type = type(e).__name__
                if (
                    "SystemExit" in ex_type
                    and isinstance(self.get_current_agent(), HumanAgent)
                ) or "KeyboardInterrupt" in ex_type:
                    sys.exit(0)
                print(
                    "An exception raised. The traceback is as follows:\n{}".format(
                        traceback.format_exc()
                    )
                )
                print("Executing Random Move!")
                move_pos = random_move(self.chess_board, cur_player)

            execute_move(self.chess_board, move_pos, cur_player)
            logger.info(
                f"Player {self.player_names[self.turn]} places at {move_pos}. Time taken this turn (in seconds): {time_taken}"
            )

        self.turn = 1 - self.turn
        results = check_endgame(
            self.chess_board, self.get_current_player(), self.get_current_opponent()
        )
        self.results_cache = results
        if self.display_ui:
            self.render()
        return results

    def get_current_agent(self):
        return self.p0 if self.turn == 0 else self.p1

    def render(self, debug=False):
        self.ui_engine.render(self.chess_board, debug=debug)
        sleep(self.display_delay)

    def play_games(self):
        for game_number in range(1, self.autoplay_runs + 1):
            self.current_game = game_number
            self.reset_board()
            is_end, p0_score, p1_score = self.step()
            while not is_end:
                is_end, p0_score, p1_score = self.step()
            if p0_score > p1_score:
                self.player_1_wins += 1
                winner = f"Player 1 ({self.player_1_name})"
            elif p1_score > p0_score:
                self.player_2_wins += 1
                winner = f"Player 2 ({self.player_2_name})"
            else:
                self.draws += 1
                winner = "Draw"
            total_games_played = self.current_game
            win_rate_p1 = (self.player_1_wins / total_games_played) * 100
            win_rate_p2 = (self.player_2_wins / total_games_played) * 100
            print(f"Game {self.current_game}/{self.autoplay_runs}: Winner: {winner}")
            print(f"Current Win Rates:")
            print(f"{self.player_1_name}: {win_rate_p1:.2f}%")
            print(f"{self.player_2_name}: {win_rate_p2:.2f}%")
            print(f"Draws: {self.draws}")
            print("-" * 40)


if __name__ == "__main__":
    total_games = 15
    world = World(
        player_1="student_agent",
        player_2="random_agent",
        autoplay=True,
        autoplay_runs=total_games,
        display_ui=False,
    )
    world.play_games()
