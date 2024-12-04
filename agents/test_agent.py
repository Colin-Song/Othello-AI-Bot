#python simulator.py --player_1 test_agent --player_2 random_agent --display
#python simulator.py --player_1 test_agent --player_2 random_agent --autoplay --autoplay_runs 10
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("test_agent")
class TestAgent(Agent):
    def __init__(self):
        super(TestAgent, self).__init__()
        self.name = "TestAgent"
        self.time_limit = 1.995  # Time limit in seconds
        self.start_time = None  # Will be set at the beginning of step()
    

    def step(self, chess_board, player, opponent):

        class Node:
            def __init__(self, parent, move, board, player):
                self.parent = parent        # Parent node
                self.move = move            # The move that led to this state
                self.board = board
                self.children = []          # Child nodes
                self.visits = 0             # Number of times this node was visited
                self.value = 0.0            # Number of wins from this node
                self.untried_moves = get_valid_moves(board, player)
                self.value_boards = {
                    6: np.array([
                        [100, -50,  10,  10, -50, 100],
                        [-50, -50,  -2,  -2, -50, -50],
                        [ 10,  -2,   5,   5,  -2,  10],
                        [ 10,  -2,   5,   5,  -2,  10],
                        [-50, -50,  -2,  -2, -50, -50],
                        [100, -50,  10,  10, -50, 100]
                    ]),
                    8: np.array([
                        [100, -50,  10,   5,   5,  10, -50, 100],
                        [-50, -50,  -2,  -2,  -2,  -2, -50, -50],
                        [ 10,  -2,   5,   1,   1,   5,  -2,  10],
                        [  5,  -2,   1,   0,   0,   1,  -2,   5],
                        [  5,  -2,   1,   0,   0,   1,  -2,   5],
                        [ 10,  -2,   5,   1,   1,   5,  -2,  10],
                        [-50, -50,  -2,  -2,  -2,  -2, -50, -50],
                        [100, -50,  10,   5,   5,  10, -50, 100]
                    ]),
                    10: np.array([
                        [100, -50,  10,   5,   5,   5,   5,  10, -50, 100],
                        [-50, -50,  -2,  -2,  -2,  -2,  -2,  -2, -50, -50],
                        [ 10,  -2,   5,   1,   1,   1,   1,   5,  -2,  10],
                        [  5,  -2,   1,   0,   0,   0,   0,   1,  -2,   5],
                        [  5,  -2,   1,   0,   0,   0,   0,   1,  -2,   5],
                        [  5,  -2,   1,   0,   0,   0,   0,   1,  -2,   5],
                        [  5,  -2,   1,   0,   0,   0,   0,   1,  -2,   5],
                        [ 10,  -2,   5,   1,   1,   1,   1,   5,  -2,  10],
                        [-50, -50,  -2,  -2,  -2,  -2,  -2,  -2, -50, -50],
                        [100, -50,  10,   5,   5,   5,   5,  10, -50, 100]
                    ]),
                    12: np.array([
                        [100, -50,  10,   5,   5,   5,   5,   5,   5,  10, -50, 100],
                        [-50, -50,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2, -50, -50],
                        [ 10,  -2,   5,   1,   1,   1,   1,   1,   1,   5,  -2,  10],
                        [  5,  -2,   1,   0,   0,   0,   0,   0,   0,   1,  -2,   5],
                        [  5,  -2,   1,   0,   0,   0,   0,   0,   0,   1,  -2,   5],
                        [  5,  -2,   1,   0,   0,   0,   0,   0,   0,   1,  -2,   5],
                        [  5,  -2,   1,   0,   0,   0,   0,   0,   0,   1,  -2,   5],
                        [  5,  -2,   1,   0,   0,   0,   0,   0,   0,   1,  -2,   5],
                        [  5,  -2,   1,   0,   0,   0,   0,   0,   0,   1,  -2,   5],
                        [ 10,  -2,   5,   1,   1,   1,   1,   1,   1,   5,  -2,  10],
                        [-50, -50,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2, -50, -50],
                        [100, -50,  10,   5,   5,   5,   5,   5,   5,  10, -50, 100]
                    ])
                }

            def is_fully_expanded(self):
                return self.untried_moves is not None and len(self.untried_moves) == 0
            
            def best_child(self, c):
                choices_weights = [
                    (child.value / child.visits) + c * np.sqrt(
                        (2 * np.log(self.visits) / child.visits))
                    for child in self.children
                ]
                return self.children[np.argmax(choices_weights)]

        def tree_policy(node, board, player):
            cur_player = player
            opp = 3 - player
            next_board = board.copy()
            n = board.shape[0]
            corners = [[0, 0], [0, n-1], [n-1, 0], [n-1, n-1]] 
            # Selection and Expansion
            while True:
                if time.time() - self.start_time > self.time_limit:
                    # Return the current node and state to prevent errors
                    return node
                if node.untried_moves is None:
                    node.untried_moves = get_valid_moves(next_board, cur_player)
                
                if node.untried_moves:
                    # Expand
                    move = node.untried_moves.pop()
                    execute_move(next_board, move, cur_player)
                    child_node = Node(node, move, next_board, cur_player)
                    if move in corners:
                        update_value_board(child_node, next_board, player)
                    node.children.append(child_node)
                    return child_node
                else:
                    
                    if not node.children:
                        # No children to select from, return the node
                        return node
                    else:
                        # Select
                        node = node.best_child(np.sqrt(2))
                        execute_move(next_board, node.move, cur_player)
                        update_value_board(node, next_board, player)
                    # Switch players
                    cur_player, opp = opp, cur_player

        def default_policy(node, board, cur_player, opponent):
            # Rollout with depth limit
            sim_board = deepcopy(board)
            max_depth = 5
            depth = 0
            n = board.shape[0]
            corners = [[0, 0], [0, n-1], [n-1, 0], [n-1, n-1]] 
            while depth < max_depth:
                if time.time() - self.start_time > self.time_limit:
                    break  # Exit if time limit is reached
                valid_moves = get_valid_moves(sim_board, cur_player)
                if valid_moves:
                    # Use heuristic-based move selection
                    move, value = value_move(node, sim_board, valid_moves)
                    execute_move(sim_board, move, cur_player)
                    child_node = Node(node, move, sim_board, cur_player)
                    if move in corners:
                        update_value_board(child_node, sim_board, player)
                    node.children.append(child_node)
                    node = child_node
                else:
                    # Pass turn if no valid moves
                    cur_player, opponent = opponent, cur_player
                    valid_moves = get_valid_moves(sim_board, cur_player)
                    if not valid_moves:
                        # Game over
                        break
                cur_player, opponent = opponent, cur_player
                depth += 1
            # Heuristic evaluation
            return node, board_value(sim_board, player, opponent)
        
        def backup(node, reward):
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

            
        def value_move(node, board, valid_moves):
            value = -np.inf
            move = None
            n = board.shape[0]
            corners = [[0, 0], [0, n-1], [n-1, 0], [n-1, n-1]] 
            x = [(1,1), (1,n-2), (n-2,1), (n-2,n-2)]
            c = [(0,1), (1,0), (0,n-2), (1,n-1), (n-1,1), (n-2,0), (n-1,n-2), (n-2,n-1)]
            for next_move in valid_moves:
                if next_move in corners:
                    return next_move, float('inf')  # Corner capture is the best move
                
                if next_move in x:
                    print(next_move)
                    idx = x.index(next_move)
                    corner = corners[idx]
                    if board[corner[0]][corner[1]] != player:
                        continue  # Skip this move

                if next_move in c:
                    idc = c.index(next_move) // 2
                    corner = corners[idc]
                    if board[corner[0]][corner[1]] != player:
                        continue  # Skip this move

                temp_board = board.copy()
                execute_move(temp_board, next_move, player)
                # Check if this move gives away a corner to the opponent
                if any(move in get_valid_moves(temp_board, 3 - player) for move in corners):
                    continue  # Skip this move

                # check if this is a C tile and then check if adjacent corner is ours, if so then change value
                if node.value_boards[board.shape[0]][next_move[0]][next_move[1]] > value:
                    move = next_move
                    value = node.value_boards[board.shape[0]][next_move[0]][next_move[1]] 
            
            return move, value
        #python simulator.py --player_1 test_agent --player_2 random_agent --display

        def disc_count(board, player, opponent):
            """
            Calculate the disc difference between the player and the opponent.
            """
            return np.sum(board == player) - np.sum(board == opponent)
        
        def mobility(board, player, opponent):
            """
            Calculate the mobility difference between the player and the opponent.
            """
            player_moves = len(get_valid_moves(board, player))
            opponent_moves = len(get_valid_moves(board, opponent))
            return player_moves - opponent_moves
        
        def count_stable_discs(board, player):
            """
            Counts the number of stable discs for a given player on the board.
            Stable discs are those that cannot be flipped in any future move.
            """
            n = board.shape[0]
            stable = np.zeros((n, n), dtype=bool)

            # Directions: up, down, left, right, and the four diagonals
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]

            # Initialize stability from corners
            corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
            queue = []
            for corner in corners:
                if board[corner[0]][corner[1]] == player:
                    stable[corner[0]][corner[1]] = True
                    queue.append(corner)

            # Breadth-First Search to propagate stability
            while queue:
                i, j = queue.pop(0)
                for dx, dy in directions:
                    x, y = i + dx, j + dy
                    if 0 <= x < n and 0 <= y < n:
                        if not stable[x][y] and board[x][y] == player:
                            # Check if the line from (x, y) to the edge in direction (dx, dy) has no empty squares
                            xi, yi = x, y
                            stable_line = True
                            while 0 <= xi < n and 0 <= yi < n:
                                if board[xi][yi] == 0:
                                    stable_line = False
                                    break
                                xi += dx
                                yi += dy
                            if stable_line:
                                stable[x][y] = True
                                queue.append((x, y))

            # Count the number of stable discs
            return np.sum(stable)
        
        def corner_occupancy(board, player, opponent):
            """
            Calculate the corner occupancy difference between the player and the opponent.
            """
            n = board.shape[0]
            corners = [(0, 0), (0, n -1), (n -1, 0), (n -1, n -1)]
            player_corners = sum(1 for x, y in corners if board[x][y] == player)
            opponent_corners = sum(1 for x, y in corners if board[x][y] == opponent)
            return player_corners - opponent_corners

        def x_c_squares(board, player):
            """
            Apply penalties or bonuses for occupying X and C squares.
            """
            n = board.shape[0]
            corners = [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]
            x_squares = [(1,1), (1,n-2), (n-2,1), (n-2,n-2)]
            c_squares = [(0,1), (1,0), (0,n-2), (1,n-1), (n-1,1), (n-2,0), (n-1,n-2), (n-2,n-1)]

            score = 0

            # X-squares
            for idx, (x, y) in enumerate(x_squares):
                corner = corners[idx]
                if board[x][y] == player:
                    if board[corner[0]][corner[1]] != player:
                        score -= 50  # Penalty for occupying X-square when corner is not owned
                    elif board[corner[0]][corner[1]] == player:
                        score += 1


            # C-squares
            for idx, (x, y) in enumerate(c_squares):
                corner_idx = idx // 2  # Each corner has two C-squares
                corner = corners[corner_idx]
                if board[x][y] == player:
                    if board[corner[0]][corner[1]] != player:
                        score -= 30  # Penalty
                    elif board[corner[0]][corner[1]] == player:
                        score += 1

            return score

        
        def inner_control(board, player, opponent):
            """
            Evaluate control over the inner middle pieces.
            """
            n = board.shape[0]
            # Define inner middle region
            inner_start = n // 4
            inner_end = n - inner_start
            player_control = np.sum(board[inner_start:inner_end, inner_start:inner_end] == player)
            opponent_control = np.sum(board[inner_start:inner_end, inner_start:inner_end] == opponent)
            return player_control - opponent_control


#python simulator.py --player_1 test_agent --player_2 random_agent --display
        def board_value(sim_board, player, opponent):
            n = sim_board.shape[0]
            score = 0
            corners = [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]
            if any(move in get_valid_moves(sim_board, opponent) for move in corners):
                score -= 500  # Significant penalty

            # Calculate evaluation factors
            disc_diff = disc_count(sim_board, player, opponent)
            mobility_diff = mobility(sim_board, player, opponent)
            stability_diff = count_stable_discs(sim_board, player)
            corner_diff = corner_occupancy(sim_board, player, opponent)
            x_c_square_value = x_c_squares(sim_board, player)
            inner_pieces = inner_control(sim_board, player, opponent)

            # Determine game phase
            total_discs = np.sum(sim_board == 1) + np.sum(sim_board == 2)
            total_squares = n * n
            occupancy = total_discs / total_squares

            if occupancy < 0.2:
                # Early Game
                disc_diff_w = 0
                mobility_diff_w = 70
                stability_diff_w = 0
                corner_diff_w = 0
                x_c_square_value_w = 100
                inner_pieces_w = 60
            elif occupancy < 0.65:
                # Mid Game
                disc_diff_w = 0
                mobility_diff_w = 30
                stability_diff_w = 70
                corner_diff_w = 70
                x_c_square_value_w = 50
                inner_pieces_w = 30
            else:
                # Late Game
                disc_diff_w = 80
                mobility_diff_w = 0
                stability_diff_w = 30
                corner_diff_w = 100
                x_c_square_value_w = 0
                inner_pieces_w = 0

            # Calculate the score
            score = (disc_diff * disc_diff_w +
                    mobility_diff * mobility_diff_w +
                    stability_diff * stability_diff_w +
                    corner_diff * corner_diff_w +
                    x_c_square_value * x_c_square_value_w +
                    inner_pieces * inner_pieces_w)

            # Return the score directly
            return score

            
        def update_value_board(node, board, player):
            n = board.shape[0]
            corners = [[0, 0], [0, n-1], [n-1, 0], [n-1, n-1]] 
            x = [[1, 1], [1, n-2], [n-2, 1], [n-2, n-2]]
            c = [[0, 1], [1, 0], [0, n-2], [1, n-1], [n-1, 1], [n-2, 0], [n-1, n-2], [n-2, n-1]]

            for i in range(0, 4):
                if board[corners[i][0]][corners[i][1]] == player:
                    node.value_boards[n][x[i][0]][x[i][1]] = 50
                    node.value_boards[n][c[i*2][0]][c[i*2][1]] = 50
                    node.value_boards[n][c[i*2+1][0]][c[i*2+1][1]] = 50

        def select_algo(board):
            """
            Decide whether to use MCTS or Minimax based on the number of remaining moves.
            """
            # if there are 10 moves or less then use alphabeta pruning
            remaining_moves = np.sum(board == 0)
            if remaining_moves <= 15:
                return "minimax"
            # more than 10 moves use monte carlo tree search
            return "mcts"


        self.start_time = time.time()
        # Initialize the root node with the current game state
        board = deepcopy(chess_board)
        valid_moves = get_valid_moves(board, player)
        best_move = None
        if not valid_moves:
            return best_move  # No valid moves, pass the turn
        
        algo = select_algo(board)
        if algo == 'mcts':
            root_node = Node(None, None, board, player)
            node = root_node
            # MCTS main loop
            while time.time() - self.start_time < self.time_limit:
                expand_node = tree_policy(node, board, player)
                sim_node, reward = default_policy(expand_node, expand_node.board, player, opponent)
                backup(sim_node, reward)

            # Choose the move with the highest visit count
            if root_node.children:
                    best_move = max(root_node.children, key=lambda c: c.visits).move
            else:
                # If no moves were simulated, fall back to a random valid move
                best_move = random_move(chess_board, player)
            
            return best_move
        else:
            best_score = -float('inf')
            alpha = -float('inf')
            beta = float('inf')
            depth = 1

            while time.time() - self.start_time < self.time_limit:
                for move in valid_moves:
                    sim_board = deepcopy(chess_board)
                    execute_move(sim_board, move, player)
                    score = self.min_value(sim_board, depth - 1, alpha, beta, player, opponent)

                    if score > best_score:
                        best_score = score
                        best_move = move

                    alpha = max(alpha, best_score)

                depth += 1  # Increment depth for iterative deepening
            return best_move
        

    def max_value(self, board, depth, alpha, beta, player, opponent):
        """
        Maximize the score for the current player.
        """
        if depth == 0 or check_endgame(board, player, opponent)[0] or time.time() - self.start_time >= self.time_limit:
            return self.evaluate(board, player, opponent)

        max_score = -float('inf')
        valid_moves = get_valid_moves(board, player)

        for move in valid_moves:
            next_board = deepcopy(board)
            execute_move(next_board, move, player)
            score = self.min_value(next_board, depth - 1, alpha, beta, player, opponent)

            max_score = max(max_score, score)
            if max_score >= beta:
                return max_score  # Beta cut-off
            alpha = max(alpha, max_score)

        return max_score

    def min_value(self, board, depth, alpha, beta, player, opponent):
        """
        Minimize the score for the opponent.
        """
        if depth == 0 or check_endgame(board, player, opponent)[0] or time.time() - self.start_time >= self.time_limit:
            return self.evaluate(board, player, opponent)

        min_score = float('inf')
        valid_moves = get_valid_moves(board, opponent)

        for move in valid_moves:
            next_board = deepcopy(board)
            execute_move(next_board, move, opponent)
            score = self.max_value(next_board, depth - 1, alpha, beta, player, opponent)

            min_score = min(min_score, score)
            if min_score <= alpha:
                return min_score  # Alpha cut-off
            beta = min(beta, min_score)

        return min_score

    def evaluate(self, board, player, opponent):
        """
        Evaluate the board state based on the disc count difference.
        """
        player_score = np.sum(board == player)
        opponent_score = np.sum(board == opponent)
        return player_score - opponent_score
