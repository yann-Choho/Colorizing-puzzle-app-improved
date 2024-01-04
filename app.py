# app.py


# Main code for the Flask app
# Each puzzle page (sliding / free) has its own :
    # .html to specify display
    # .js to specify interactive content 
    # The .css is common


# Requirements
from flask import Flask, request, session, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import random


app = Flask(__name__)
app.secret_key = 'secret_key'


import torch
from torchvision import transforms
from PIL import Image

from colorization_training import ColorizationNet

###################### COLORIZATION WITH DEEP LEARNING ######################

# Function to load the model
def load_model(model_path):
    model = ColorizationNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the model
model = load_model('model1.pth')  # TODO : put it in folder colorization model

# Function to colorize the image
def colorize_image(image_path, model, output_format='png'):
    """
    Colorizes a black and white image.

    Parameters
    ----------
    img : a PIL Image, or the string of the path of the image

    Returns
    -------
    Image
        the link to the colorisied image colorized_image_path.
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



def create_puzzle(base_image_path, base_image_color_path, rows=rows, cols=cols):
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

##### THE DIFFERENTE ROUTES OF THE FLASK APP #####


    
# Menu 
@app.route('/')
def index():
    return render_template('Index.html')

# Dans vos routes de puzzle, utilisez les chemins d'image de la session
@app.route('/sliding_pieces')
def sliding_pieces():
    session['previous_url'] = url_for('sliding_pieces')
    session['current_url'] = url_for('sliding_pieces')

    # Récupérer le chemin de l'image sélectionnée stockée dans la session
    base_image_path = session.get('puzzle_image_path', 'static/images/puzzle.png')
    base_image_color_path = session.get('puzzle_image_color_path', 'static/images/puzzle_color.png')
    
    # Générer le puzzle avec l'image sélectionnée
    puzzle = create_puzzle(base_image_path, base_image_color_path, rows, cols)
    
    return render_template('sliding_pieces.html', puzzle=puzzle)


@app.route('/free_pieces')
def free_pieces():
    session['previous_url'] = url_for('free_pieces')
    session['current_url'] = url_for('free_pieces')
    
    # Utilisez les mêmes chemins d'image que ceux définis dans la session

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

    # Chemin de l'image d'origine sélectionnée
    original_image_path = os.path.join('static/images/examples', selected_image)

    # Chemin de l'image colorisée qui sera créé
    colorized_image_path = 'static/images/colorized_image.png'

    # Appeler la fonction pour coloriser l'image et la sauvegarder
    colorize_image(original_image_path, colorized_image_path)

    # Stocker les chemins dans la session
    session['puzzle_image_path'] = original_image_path
    session['puzzle_image_color_path'] = colorized_image_path

    # Récupérer l'URL précédente pour la redirection
    previous_url = session.get('previous_url', url_for('index'))

    # Retourner l'URL pour la redirection côté client
    return jsonify({'redirect_url': previous_url})


# Ensemble des extensions de fichiers autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    # Vérifiez si l'extension du fichier est autorisée
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = 'static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image_route():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Créer un nouveau puzzle avec l'image téléchargée
            session['puzzle_image_path'] = filepath # Stocker le chemin dans la session
            colorized_image_path = colorize_image(filepath, model) # Coloriser l'image
            session['colorized_image_path'] = colorized_image_path # Stocker le chemin de l'image colorisée dans la session

            current_url = session.get('current_url', url_for('index'))
            return redirect(current_url)

    # Si GET, afficher la page de téléchargement
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(port=8000, debug=True)
