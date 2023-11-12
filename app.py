# DANS LE TERMINAL VSCODE
# deactivate conda
# python3 -m venv env
# source ./env/bin/activate
# pip install Flask Pillow

# cd PUZZLE_GAME
# flask run --debug

# app.py

from flask import Flask, render_template
from PIL import Image
import io
import random

app = Flask(__name__)

# Sample puzzle dimensions
rows = 2
cols = 2

class PuzzlePiece:
    def __init__(self, id, image, order, width, height):
        self.id = id
        self.image = image
        self.order = order
        self.width = width
        self.height = height

        # Compute row and col based on the shuffled order
        self.row = order // cols
        self.col = order % cols

# Sample image URL
base_image_path = 'static/images/puzzle.png'
base_image_color_path = 'static/images/puzzle_color.png'


# Perform image slicing and save sliced images
original_image = Image.open(base_image_path)
image_width, image_height = original_image.size
piece_width = image_width // cols
piece_height = image_height // rows

for j in range(rows):
    for i in range(cols):
        left = i * piece_width
        top = j * piece_height
        right = left + piece_width
        bottom = top + piece_height

        piece_image = original_image.crop((left, top, right, bottom))
        piece_image.save(f"static/images/sliced_{j}_{i}.png")
        
        
original_image_color = Image.open(base_image_color_path)
image_color_width, image_color_height = original_image_color.size
piece_color_width = image_color_width // cols
piece_color_height = image_color_height // rows

for j in range(rows):
    for i in range(cols):
        left_color = i * piece_color_width
        top_color = j * piece_color_height
        right_color = left_color + piece_color_width
        bottom_color = top_color + piece_color_height

        piece_image_color = original_image_color.crop((left_color, top_color, right_color, bottom_color))
        piece_image_color.save(f"static/images/sliced_color_{j}_{i}.png")
        
# Initialize puzzle pieces with correct slicing
puzzle = [
    [
        PuzzlePiece(
            id=i + j * cols,
            image=f"/static/images/sliced_{j}_{i}.png",
            order=i + j * cols,
            width=512/rows,  # adjust based on your image dimensions
            height=512/cols  # adjust based on your image dimensions
        ) for i in range(cols)
    ] for j in range(rows)
]

# Randomize the order of puzzle pieces
flat_puzzle = [piece for row in puzzle for piece in row]
random.shuffle(flat_puzzle)

# Update the order of puzzle pieces
for i, piece in enumerate(flat_puzzle):
    piece.order = i

# Update the row and col based on the shuffled order
for piece in flat_puzzle:
    piece.row = piece.order // cols
    piece.col = piece.order % cols

@app.route('/')
def index():
    return render_template('index.html', puzzle=puzzle)

if __name__ == '__main__':
    app.run(debug=True)