from board import Board, Move
from engine import find_best_move


def parse_move(move_str: str) -> int:
    file = ord(move_str[0]) - ord('a')
    rank = int(move_str[1]) - 1
    return rank * 8 + file


def move_to_str(move_from: int, move_to: int) -> str:
    fr = chr(ord('a') + move_from % 8) + str(move_from // 8 + 1)
    to = chr(ord('a') + move_to % 8) + str(move_to // 8 + 1)
    return fr + to


def main() -> None:
    board = Board()
    depth = 2
    while True:
        print(board.get_fen())
        if board.turn == 'w':
            user_move = input('Enter your move (e.g., e2e4 or e7e8q): ')
            if len(user_move) < 4:
                break
            frm = parse_move(user_move[:2])
            to = parse_move(user_move[2:4])
            promotion = user_move[4] if len(user_move) > 4 else None
            board.push(Move(frm, to, promotion))
        else:
            best = find_best_move(board, depth)
            if best is None:
                print('No moves available.')
                break
            board.push(best)
            print(f'Engine move: {move_to_str(best.from_square, best.to_square)}')
        if board.is_game_over():
            print('Game over')
            break


if __name__ == '__main__':
    main()
