# -*- coding: utf-8 -*-

### === Main code for the Flask app === ###

# Each puzzle page (sliding / free) has its own :
    # .html to specify display
    # .js to specify interactive content
    # The .css is common


# Requirements
import os
import random
import import_data as imp

from flask import Flask, request, session, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image

#from colorization_utils import ColorizationNet
from colorization_model.colorization_all_utils import ColorizationNet



# Paramètres YAML
config = imp.import_yaml_config("config.yaml")
IMAGES_URL = config.get("images_url")
EXAMPLES_URL = f"{IMAGES_URL}/examples"
EXAMPLES_WITH_COLOR_URL = f"{IMAGES_URL}/examples_with_color"
#EXAMPLES_URL = "https://minio.lab.sspcloud.fr/pregnard/MEP_images/images/examples"
#EXAMPLES_WITH_COLOR_URL = "https://minio.lab.sspcloud.fr/pregnard/MEP_images/images/examples_with_color"


# Create the local data folder
examples_folder = os.path.join("static", "images/examples")
os.makedirs(examples_folder, exist_ok=True)

examples_with_color_folder = os.path.join("static", "images/examples_with_color")
os.makedirs(examples_with_color_folder, exist_ok=True)

# Number of example files
NUM_EXAMPLES = 21

# Download the 'images/examples' and ' images/examples_with_colors' data folder from the provided URL
try:
    imp.download_files(EXAMPLES_URL, examples_folder, NUM_EXAMPLES)
except Exception as e:
    print(f"An error occurred while downloading examples files from S3: {e}")

try:
    imp.download_files(EXAMPLES_WITH_COLOR_URL, examples_with_color_folder, NUM_EXAMPLES)
except Exception as e:
    print(f"An error occurred while downloading examples-with-color files from S3: {e}")


app = Flask(__name__)
CORS(app)
app.secret_key = 'secret_key'

###################### COLORIZATION WITH DEEP LEARNING ######################

def load_model(model_path: str) -> ColorizationNet:
    """Function to load the model

    Args:
        model_path (str): location of the file containing optimal parameters

    Returns:
        ColorizationNet: neural network set with chosen parameters
    """
    model = ColorizationNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the model
model = load_model('model1.pth')

# Function to colorize the image
def colorize_image(image_path: str, model, output_format: str = 'png') -> str:
    """
    Colorizes a black and white image.

    Parameters
    ----------
    img : a PIL Image, or the string of the path of the image

    Returns
    -------
    Image
        the link to the colorized image path.
    """
    # Load and transform the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    # Colorize the image
    with torch.no_grad():
        colorized = model(image)

    # Convert the output to PIL image and save it
    colorized_image = transforms.ToPILImage()(colorized.squeeze(0))
    # Modify the file path based on the specified output format
    colorized_image_path = 'static/images/colorized_image.png'
    colorized_image.save(colorized_image_path, format=output_format.upper())

    return colorized_image_path

###################### PUZZLE ######################

# Sample puzzle dimensions
rows = 3
cols = 3

# Initiate a class for the puzzle pieces
class PuzzlePiece:
    """class for puzzle pieces

    Attributes :
        id : correct rank of the piece in the puzzle
        image : for display
        order : actual rank of the piece in the puzzle
        width : for display
        height : for display
    """
    def __init__(self, id: int, image: str, order: int, width: float, height: float):
        self.id = id
        self.image = image
        self.order = order
        self.width = width
        self.height = height

        # Compute row and col based on the shuffled order
        self.row = order // cols # Example : the row of the 4th piece in a 3x3
        # puzzle is 2
        self.col = order % cols # Example : the column of the 4th piece in a
        # 3x3 puzzle is 1



def create_puzzle(base_image_path: str, base_image_color_path: str, rows: int = rows, cols: int = cols) -> list :
    """function creating the puzzle

    Args:
        base_image_path (str): chemin d'accès à l'image en noir et blanc
        base_image_color_path (str): chemin d'accès à l'image en couleurs
        rows (int, optional): Number of raws. Defaults to rows.
        cols (int, optional): Number of columns. Defaults to cols.

    Returns:
        list: list of PuzzlePiece objects representing the full puzzle 
    """

    # 1. Black and White

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


    # 2. Color

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
                id=i + j * cols,
                image=f"/static/images/sliced_{j}_{i}.png",
                order=i + j * cols,
                width=int(image_width * scaling_factor / rows),
                height=int(image_height * scaling_factor / cols)
            ) for i in range(cols)
        ] for j in range(rows)
    ]

    # Flatten the list of lists into a single list
    flat_puzzle = [piece for sublist in puzzle for piece in sublist]

    # Randomize the order of puzzle pieces
    random.shuffle(flat_puzzle)

    # Assign new order values after shuffling
    for index, piece in enumerate(flat_puzzle):
        piece.order = index
        piece.row = index // cols
        piece.col = index % cols

    # Re-create the 2D puzzle structure
    shuffled_puzzle = [
        flat_puzzle[i * cols:(i + 1) * cols] for i in range(rows)
    ]

    return shuffled_puzzle

##### THE DIFFERENT ROUTES OF THE FLASK APP #####


# Menu
@app.route('/')
def index():
    return render_template('index.html')

# use the images path as an app route
@app.route('/sliding_pieces')
def sliding_pieces():
    session['previous_url'] = url_for('sliding_pieces')
    session['current_url'] = url_for('sliding_pieces')

    # get the path of the image saved in the session
    base_image_path = session.get('puzzle_image_path', 'static/images/puzzle.png')
    base_image_color_path = session.get('puzzle_image_color_path', 'static/images/puzzle_color.png')

    # create the puzzle with the image selected
    puzzle = create_puzzle(base_image_path, base_image_color_path, rows, cols)

    return render_template('sliding_pieces.html', puzzle=puzzle)


@app.route('/free_pieces')
def free_pieces():
    session['previous_url'] = url_for('free_pieces')
    session['current_url'] = url_for('free_pieces')

    # Use the same image paths as those defined in the session

    base_image_path = session.get('puzzle_image_path', 'static/images/puzzle.png')
    base_image_color_path = session.get('puzzle_image_color_path', 'static/images/puzzle_color.png')

    puzzle = create_puzzle(base_image_path, base_image_color_path, rows, cols)

    return render_template('free_pieces.html', puzzle=puzzle)

# Select image
@app.route('/select-image')
def select_image():
    image_files = os.listdir('static/images/examples')
    return render_template('select_image.html', images=image_files)


@app.route('/set-puzzle-image', methods=['POST'])
def set_puzzle_image():
    data = request.get_json()
    selected_image = data['image']

    # Path of the original selected image
    original_image_path = os.path.join('static/images/examples', selected_image)

    # Call the colorize_image function to colorize and save the image
    colorized_image_path = colorize_image(original_image_path, model)

    # Store the paths in the session
    session['puzzle_image_path'] = original_image_path
    session['puzzle_image_color_path'] = colorized_image_path

    # Retrieve the previous URL for redirection
    previous_url = session.get('previous_url', url_for('index'))

    # Return the URL for client-side redirection
    return jsonify({'redirect_url': previous_url})


# define file extensions allowed
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename) -> bool :
    # check whether the file extension is allowed
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = '../5000/static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image_route():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # create a new puzzle with the downloaded image
            session['puzzle_image_path'] = filepath # save the path in the login
            colorized_image_path = colorize_image(filepath, model) # colorize the image
            session['colorized_image_path'] = colorized_image_path # save the colorized image path in the login

            current_url = session.get('current_url', url_for('index'))
            return redirect(current_url)

    # if get, display the download page
    return render_template('index.html') # go back to the menu if an error occurs


if __name__ == '__main__':
    app.run(port=5000, debug=True)
