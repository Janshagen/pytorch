from typing import Optional
import random


class Die:
    def __init__(self) -> None:
        self.value: int = 0
        self.keeping: bool = False

    def throw(self) -> None:
        self.value = random.randint(1, 6)

    def keep(self) -> None:
        self.keeping = True

    def reset(self) -> None:
        self.keeping = False

    def flip_state(self) -> None:
        self.keeping = not self.keeping


class Dice:
    def __init__(self) -> None:
        self.dice = [Die() for _ in range(5)]

    def reset(self) -> None:
        for die in self.dice:
            die.reset()

    def throw(self) -> None:
        for die in self.dice:
            if not die.keeping:
                die.throw()

    def get_dice_sum(self) -> int:
        dice_sum = 0
        for die in self.dice:
            dice_sum += die.value
        return dice_sum

    def count_of_die_values(self) -> list[int]:
        values = []
        dice_values = self.get_dice_values()
        for value in range(1, 7):
            values.append(dice_values.count(value))
        return values

    def get_dice_values(self):
        return [die.value for die in self.dice]


class Sheet:

    MOVE2INDEX: dict[str, int] = {
        "ones": 0, "twos": 1,
        "threes": 2, "fours": 3,
        "fives": 4, "sixes": 5,
        "one pair": 6, "two pair": 7,
        "three of a kind": 8, "four of a kind": 9,
        "small straight": 10, "big straight": 11,
        "full house": 12, "chance": 13,
        "yatzy": 14, "sum": 15
    }

    UPPER_MOVES = list(["ones", "twos",
                       "threes", "fours",
                        "fives", "sixes"])

    def __init__(self, player: int,
                 moves_made: Optional[dict[str, int]] = None,
                 moves_left: Optional[set[str]] = None,
                 done_with_upper_part: bool = False) -> None:

        if moves_made:
            self.points = moves_made
        else:
            self.points: dict[str, int] = {
                "ones": 0, "twos": 0,
                "threes": 0, "fours": 0,
                "fives": 0, "sixes": 0,
                "one pair": 0, "two pair": 0,
                "three of a kind": 0, "four of a kind": 0,
                "small straight": 0, "big straight": 0,
                "full house": 0, "chance": 0,
                "yatzy": 0, "sum": 0
            }

        if moves_left:
            self.moves_left = moves_left
        else:
            self.moves_left: set[str] = {
                "ones", "twos",
                "threes", "fours",
                "fives", "sixes",
                "one pair", "two pair",
                "three of a kind", "four of a kind",
                "small straight", "big straight",
                "full house", "chance",
                "yatzy"
            }

        self.player = player
        self.done_with_upper_part = done_with_upper_part

    def copy(self) -> 'Sheet':
        new_points = self.points.copy()
        new_available_moves = self.moves_left.copy()
        return Sheet(self.player, new_points, new_available_moves,
                     self.done_with_upper_part)

    def has_moves_left(self) -> bool:
        return len(self.moves_left) > 0

    def get_points(self, dice: Dice, move: str) -> int:
        count_of_die_values = dice.count_of_die_values()

        for i, possible_move in enumerate(self.UPPER_MOVES):
            die_value = i+1
            if move == possible_move:
                return count_of_die_values[i]*die_value

        for j, possible_move in enumerate(
                ["one pair", "three of a kind", "four of a kind", "yatzy"]):
            if move != possible_move:
                continue
            number_of_dice_necessary = j+2
            for i, count in reversed(list(enumerate(count_of_die_values))):
                die_value = i+1
                if count >= number_of_dice_necessary:
                    return 50 if move == "yatzy" else number_of_dice_necessary*die_value

        if move == "two pair":
            if (count_of_die_values.count(2) == 2) or \
                    (count_of_die_values.count(2) == 1 and
                     count_of_die_values.count(3) == 1):
                sum = 0
                for i, count in enumerate(count_of_die_values):
                    die_value = i+1
                    if count >= 2:
                        sum += 2*die_value
                return sum

        elif move == "full house":
            sum = 0
            if count_of_die_values.count(2) == 1 and count_of_die_values.count(3) == 1:
                for i, count in enumerate(count_of_die_values):
                    die_value = i+1
                    sum += count*die_value
                return sum

        if (move == "small straight" and
            count_of_die_values[0] and
            count_of_die_values[1] and
            count_of_die_values[2] and
            count_of_die_values[3] and
                count_of_die_values[4]):
            return 15

        elif (move == "big straight" and
              count_of_die_values[1] and
              count_of_die_values[2] and
              count_of_die_values[3] and
              count_of_die_values[4] and
              count_of_die_values[5]):
            return 20

        if move == "chance":
            return dice.get_dice_sum()

        return 0

    def get_score(self):
        return self.points["sum"]

    def make_move(self, move: str, value: int) -> None:
        self.points[move] = value
        self.points["sum"] += value

        self.moves_left.remove(move)
        self.handle_bonus(move)

    def handle_bonus(self, move: str) -> None:
        if self.done_with_upper_part:
            return
        if move not in self.UPPER_MOVES:
            return
        if not self.upper_moves_left():
            self.add_bonus()
            self.done_with_upper_part = True

    def upper_moves_left(self) -> bool:
        if "ones" in self.moves_left or \
            "twos" in self.moves_left or \
            "threes" in self.moves_left or \
            "fours" in self.moves_left or \
            "fives" in self.moves_left or \
                "sixes" in self.moves_left:
            return True
        return False

    def add_bonus(self) -> None:
        sum = 0
        for move in self.UPPER_MOVES:
            sum += self.points[move]
        if sum >= 63:
            self.points["sum"] += 50

    def __getitem__(self, move: str):
        return self.points[move]

    @staticmethod
    def move2index(move) -> int:
        return Sheet.MOVE2INDEX[move]


class YatzyGameState:
    GAME_NOT_OVER = 2
    DRAW = 0

    def __init__(self, sheets: list[Sheet], current_player: int,
                 status: int = GAME_NOT_OVER) -> None:
        # player is the player to make a move
        self.sheets: list[Sheet] = sheets
        self.dice = Dice()
        self.current_player: int = current_player

        self.status: int = status

    def copy(self) -> 'YatzyGameState':
        return YatzyGameState([sheet.copy() for sheet in self.sheets],
                              self.current_player, self.status)

    @staticmethod
    def new_game(starting_player: Optional[int] = None) -> 'YatzyGameState':
        starting_player = starting_player if starting_player == 1 else 0
        return YatzyGameState([Sheet(player=0), Sheet(player=1)], starting_player)

    def available_moves(self) -> set[str]:
        return self.sheets[self.current_player].moves_left

    def throw(self):
        self.dice.throw()

    def make_move(self, move: str) -> None:
        dice_sum = self.sheets[0].get_points(self.dice, move)

        self.sheets[self.current_player].make_move(move, dice_sum)
        self.dice.reset()

        self.next_player()
        self.update_status()

    def next_player(self) -> None:
        self.current_player = 0 if self.current_player == 1 else 1

    def update_status(self) -> None:
        self.status = self.calculate_game_status()

    def calculate_game_status(self) -> int:
        if self.sheets[-1].has_moves_left():
            return YatzyGameState.GAME_NOT_OVER

        scores = [sheet.get_score() for sheet in self.sheets]
        if scores[0] > scores[1]:
            return -1
        elif scores[1] > scores[0]:
            return 1
        return YatzyGameState.DRAW

    def game_over(self) -> bool:
        if self.status == YatzyGameState.GAME_NOT_OVER:
            return False
        return True

    def get_status(self):
        return self.status

    def get_dice(self):
        return self.dice.dice
