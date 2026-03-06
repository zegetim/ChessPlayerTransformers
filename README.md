# Chess Transformer Player

Minimal template for the ML Chess assignment.

## Interface

Your class **must be named**:

TransformerPlayer

It must implement:

```python
get_move(self, fen: str) -> Optional[str]
```
Return: UCI move string (e2e4) OR None

## Installation

You MUST install the instructor package. In Colab:

```
git clone https://github.com/bylinina/chess_exam.git
cd chess-exam
pip install -e .
```

## Example

```python
from player import TransformerPlayer

p = TransformerPlayer("MyBot")
move = p.get_move("starting FEN")
print(move)
```