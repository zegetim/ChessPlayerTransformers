import chess
import random
import re
import torch
import numpy as np
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from chess_exam.chess_tournament.players import Player

class TransformerPlayer(Player):
    """
    Transformer chess encoder player.

    REQUIRED:
        Subclasses chess_tournament.players.Player
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "EncoderPlayer",
        model_id: str = "timzeg/distilbert-chess",
        temperature: float = 0.1,
        max_new_tokens: int = 5,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded components
        self.tokenizer = None
        self.model = None

    # -------------------------
    # Lazy loading
    # -------------------------
    def _load_model(self):
        if self.model is None:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                num_labels=1,
                problem_type="regression"
            )
            self.model.to(self.device)
            self.model.eval()

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    # random legal move
    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    # check if checkmate is available
    def checkmate_available(self, fen:str, move: str) -> bool:
        board = chess.Board(fen)
        board.push(move)
        return board.is_checkmate()
    
    # check if position is attacked
    def attacked(self, fen:str, move: str) -> bool:
        board = chess.Board(fen)
        if 'w' in fen:
            attacked = board.is_attacked_by(chess.BLACK, chess.parse_square(move[-2:]))
        else:
            attacked = board.is_attacked_by(chess.WHITE, chess.parse_square(move[-2:]))
        return attacked

    # check if cpature is available
    def capture(self, fen: str) -> list[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)

        best_moves = []
        checkmate_moves = []
        
        for move in moves:
            piecetype = board.piece_at(move.to_square)
            if self.checkmate_available(fen, move):
                checkmate_moves.append(move.uci())
            if board.is_capture(move):
                if not self.attacked(fen, move.uci()):
                    best_moves.append(move.uci())
        if len(checkmate_moves) > 0:
            return checkmate_moves
        elif len(best_moves) > 0:
            return best_moves
        # print("model wordt gebruikt")
        return [move.uci() for move in moves]
    
    # model chooses move with highest 
    def choose_move(self, fen: str, moves: list[str]) -> str:
        self._load_model()
        board = chess.Board(fen)
        prompt = [f"FEN: {fen} Move: {move}" for move in moves]
        inputs = self.tokenizer(prompt, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).tolist()

        if board.turn == chess.WHITE:
            best_move = np.argmax(scores)
        else:
            best_move = np.argmin(scores)
        # best_idx = np.argmax(scores)
        return moves[best_move]
    
    # -------------------------
    # Main API
    # -------------------------
    
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        try:
            capture_moves = self.capture(fen)
            if capture_moves:
                return self.choose_move(fen, capture_moves)
        except Exception:
            pass
        return self._random_legal(fen)
