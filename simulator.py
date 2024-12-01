from world import World, PLAYER_1_NAME, PLAYER_2_NAME
import argparse
from utils import all_logging_disabled
import logging
import numpy as np
import datetime

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player_1", type=str, default="random_agent")
    parser.add_argument("--player_2", type=str, default="random_agent")
    parser.add_argument("--board_size", type=int, default=None)
    parser.add_argument(
        "--board_size_min",
        type=int,
        default=6,
        help="In autoplay mode, the minimum board size",
    )
    parser.add_argument(
        "--board_size_max",
        type=int,
        default=12,
        help="In autoplay mode, the maximum board size",
    )
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--display_delay", type=float, default=0.4)
    parser.add_argument("--display_save", action="store_true", default=False)
    parser.add_argument("--display_save_path", type=str, default="plots/")
    parser.add_argument("--autoplay", action="store_true", default=False)
    parser.add_argument("--autoplay_runs", type=int, default=100)
    args = parser.parse_args()
    return args


class Simulator:
    """
    Entry point of the game simulator.

    Parameters
    ----------
    args : argparse.Namespace
    """

    def __init__(self, args):
        self.args = args
        # Only play on even-sized boards
        self.valid_board_sizes = [ i for i in range(self.args.board_size_min, self.args.board_size_max+1) if i % 2 == 0 ]
        #print("Valid sizes: ",self.valid_board_sizes)

    def reset(self, swap_players=True, board_size=None):
        """
        Reset the game

        Parameters
        ----------
        swap_players : bool
            if True, swap the players
        board_size : int
            if not None, set the board size
        """
        if board_size is None:
            board_size = self.args.board_size
        if swap_players:
            player_1, player_2 = self.args.player_2, self.args.player_1
        else:
            player_1, player_2 = self.args.player_1, self.args.player_2

        self.world = World(
            player_1=player_1,
            player_2=player_2,
            board_size=board_size,
            display_ui=self.args.display,
            display_delay=self.args.display_delay,
            display_save=self.args.display_save,
            display_save_path=self.args.display_save_path,
            autoplay=self.args.autoplay,
        )

    def run(self, swap_players=True, board_size=None):
        self.reset(swap_players=swap_players, board_size=board_size)
        is_end, p0_score, p1_score = self.world.step()
        while not is_end:
            is_end, p0_score, p1_score = self.world.step()
        logger.info(
            f"Run finished. {PLAYER_1_NAME} player, agent {self.args.player_1}: {p0_score}. {PLAYER_2_NAME}, agent {self.args.player_2}: {p1_score}"
        )
        return p0_score, p1_score, self.world.p0_time, self.world.p1_time

    def autoplay(self):
        """
        Run multiple simulations of the gameplay and aggregate win %.
        Display results after each game.
        """
        p1_win_count = 0
        p2_win_count = 0
        draws = 0
        p1_times = []
        p2_times = []

        if self.args.display:
            logger.warning("Since running autoplay mode, display will be disabled")
        self.args.display = False

        for i in range(1, self.args.autoplay_runs + 1):
            swap_players = i % 2 == 0  # Alternate swapping
            board_size = self.valid_board_sizes[np.random.randint(len(self.valid_board_sizes))]
            p0_score, p1_score, p0_time, p1_time = self.run(
                swap_players=swap_players, board_size=board_size
            )

            # Handle swapped players
            if swap_players:
                # Swap roles and align scores
                current_p1 = self.args.player_2
                current_p2 = self.args.player_1
                p0_score, p1_score = p1_score, p0_score
                p0_time, p1_time = p1_time, p0_time
            else:
                current_p1 = self.args.player_1
                current_p2 = self.args.player_2

            # Determine the winner
            if p0_score > p1_score:
                p1_win_count += 1
                winner = f"Player 1 ({current_p1})"
            elif p1_score > p0_score:
                p2_win_count += 1
                winner = f"Player 2 ({current_p2})"
            else:
                draws += 1
                winner = "Draw"

            # Record times
            p1_times.extend(p0_time)
            p2_times.extend(p1_time)

            # Calculate win rates
            total_games_played = i
            win_rate_p1 = (p1_win_count / total_games_played) * 100
            win_rate_p2 = (p2_win_count / total_games_played) * 100

            # Print results
            print(f"Game {i}/{self.args.autoplay_runs}: Winner: {winner}")
            print(f"Scores - {current_p1}: {p0_score}, {current_p2}: {p1_score}")
            print(f"Current Win Rates:")
            print(f"{self.args.player_1}: {win_rate_p1:.2f}%") # changed from {current_p1} to {self.args.player_1}
            print(f"{self.args.player_2}: {win_rate_p2:.2f}%") # changed from {current_p2} to {self.args.player_2}
            print(f"Draws: {draws}")
            print("-" * 40)

        # Final summary
        logger.info(
            f"Player 1, agent {self.args.player_1}, win percentage: {p1_win_count / self.args.autoplay_runs:.2f}. Maximum turn time was {np.round(np.max(p1_times),5)} seconds."
        )
        logger.info(
            f"Player 2, agent {self.args.player_2}, win percentage: {p2_win_count / self.args.autoplay_runs:.2f}. Maximum turn time was {np.round(np.max(p2_times),5)} seconds."
        )


if __name__ == "__main__":
    args = get_args()
    simulator = Simulator(args)
    if args.autoplay:
        simulator.autoplay()
    else:
        simulator.run()
