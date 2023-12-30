# app.py


# Main code for the Flask app
# Each puzzle page (sliding / free) has its own :
    # .html to specify display
    # .js to specify interactive content 
# The .css is common


# Requirements
from flask import Flask, render_template
from PIL import Image
import io
import random

app = Flask(__name__)

# Sample puzzle dimensions
rows = 3
cols = 3


# Initiate a class for the puzzle pieces
class PuzzlePiece:
    def __init__(self, id: int, image: str, order: int, width: float, height: float):
        self.id = id  # correct rank of the piece in the puzzle
        self.image = image # for display
        self.order = order # actual rank of the piece in the puzzle
        self.width = width # for display
        self.height = height # for display

        # Compute row and col based on the shuffled order
        self.row = order // cols # Example : the row of the 4th piece in a 3x3
        # puzzle is 2
        self.col = order % cols # Example : the column of the 4th piece in a 
        # 3x3 puzzle is 1

# Sample image URL (put the BW and color whole images in static/images)
base_image_path = 'static/images/puzzle.png'
base_image_color_path = 'static/images/puzzle_color.png'



# Black and White

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
# each puzzle piece is cut from the whole image : 
        piece_image = original_image.crop((left, top, right, bottom))
        piece_image.save(f"static/images/sliced_{j}_{i}.png")
        
        
        
# Color 

# Perform image slicing and save sliced images
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
# each puzzle piece is cut from the whole image : 
        piece_image_color = original_image_color.crop((left_color, top_color, right_color, bottom_color))
        piece_image_color.save(f"static/images/sliced_color_{j}_{i}.png")
        
 
# Pictures must be rectangular or squared.
#  Determine the scaling factor to ensure the smallest dimension is 512px
scaling_factor = max(512 / image_width, 512 / image_height)

# Initialize puzzle pieces with correct slicing
puzzle = [
    [
        PuzzlePiece(
            id=i + j * cols, # fixed
            image=f"/static/images/sliced_{j}_{i}.png", # link the image
            order=i + j * cols, # will change
            width=int(image_width * scaling_factor / rows),
            height=int(image_height * scaling_factor / cols)
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


# The different routes of the Flask app :
    
# Menu 
@app.route('/')
def index():
    return render_template('Index.html', puzzle=puzzle)

# Sliding puzzle
@app.route('/sliding_pieces')
def sliding_pieces():
    return render_template('sliding_pieces.html', puzzle=puzzle)

# Free puzzle
@app.route('/free_pieces')
def free_pieces():
    return render_template('free_pieces.html', puzzle=puzzle)

if __name__ == '__main__':
    app.run(debug=True)