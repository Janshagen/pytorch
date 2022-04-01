import pygame
import random

# To make an MCTS AI


def availableMoves(players, horizontal_lines, vertical_lines):
    walks, placeWalls = [], []
    walks = findWalks(players, horizontal_lines, vertical_lines)
    placeWalls = findWalls(players, horizontal_lines, vertical_lines)

    for walk in walks:
        placeWalls.append(walk)
    return placeWalls


def findWalks(players, horizontal_lines, vertical_lines) -> list:
    walks = []
    opponents = [(player.r, player.c) for player in players[1:]]

    currentPos = (players[0].r, players[0].c)
    for _, (key, v) in enumerate(players[0].MOVES.items()):
        move, lineCheck = v

        primaryLines, secondaryLines = findLines(
            key, horizontal_lines, vertical_lines)
        newPos = (currentPos[0] + move[0], currentPos[1] + move[1])
        if primaryLines[currentPos[0] + lineCheck[0]][currentPos[1] + lineCheck[1]].occ:
            continue

        elif opponents.count(newPos):
            if primaryLines[newPos[0] + lineCheck[0]][newPos[1] + lineCheck[1]].occ or opponents.count((newPos[0] + move[0], newPos[1] + move[1])):
                # kolla sidorna
                movesToCheck = ((pygame.K_RIGHT, pygame.K_LEFT) if key in (
                    pygame.K_UP, pygame.K_DOWN) else (pygame.K_UP, pygame.K_DOWN))
                for key in movesToCheck:
                    move, lineCheck = players[0].MOVES[key]
                    if not secondaryLines[newPos[0] + lineCheck[0]][newPos[1] + lineCheck[1]].occ and not opponents.count((newPos[0] + move[0], newPos[1] + move[1])):
                        walks.append(
                            (currentPos, (newPos[0] + move[0], newPos[1] + move[1])))
            else:
                walks.append(
                    (currentPos, (newPos[0] + move[0], newPos[1] + move[1])))

        else:
            walks.append((currentPos, newPos))
        return walks


def findLines(key, horizontal_lines, vertical_lines):
    if key in (pygame.K_LEFT, pygame.K_RIGHT):
        return vertical_lines, horizontal_lines
    return horizontal_lines, vertical_lines


def findWalls(players, horizontal_lines, vertical_lines):
    pass


def randomMove(moves):
    return random.choice(moves)


def makeMove(move):
    pass


def gameEnd():
    pass


def nextPlayer(players, wall) -> None:
    """Changes order of the players list"""
    currentPlayer = players.pop(0)
    players.append(currentPlayer)
    wall.color = players[0].color
