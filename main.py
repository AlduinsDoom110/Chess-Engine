from board import Board, Move, InvalidMoveError
from engine import find_best_move


def parse_move(move_str: str) -> int:
    if len(move_str) != 2:
        raise ValueError(f"Invalid square notation: {move_str}")
    file_char, rank_char = move_str[0], move_str[1]
    if file_char < 'a' or file_char > 'h' or not rank_char.isdigit():
        raise ValueError(f"Invalid square notation: {move_str}")
    rank_num = int(rank_char)
    if rank_num < 1 or rank_num > 8:
        raise ValueError(f"Invalid square notation: {move_str}")
    file = ord(file_char) - ord('a')
    rank = rank_num - 1
    return rank * 8 + file


def move_to_str(move_from: int, move_to: int) -> str:
    fr = chr(ord('a') + move_from % 8) + str(move_from // 8 + 1)
    to = chr(ord('a') + move_to % 8) + str(move_to // 8 + 1)
    return fr + to


def main() -> None:
    board = Board()
    while True:
        print(board.get_fen())
        if board.turn == 'w':
            user_move = input('Enter your move (e.g., e2e4 or e7e8q): ')
            if len(user_move) < 4:
                break
            try:
                frm = parse_move(user_move[:2])
                to = parse_move(user_move[2:4])
                promotion = user_move[4] if len(user_move) > 4 else None
                board.push(Move(frm, to, promotion))
            except (ValueError, InvalidMoveError) as e:
                print(f'Invalid move: {e}')
                continue
        else:
            best = find_best_move(board, 5)
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
