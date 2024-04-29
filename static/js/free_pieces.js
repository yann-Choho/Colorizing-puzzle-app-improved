// static/js/free_pieces.js


// Function to shuffle the order of puzzle pieces
function shufflePieces() {
    const container = document.querySelector('.puzzle-pieces-container');
    const pieces = Array.from(container.children);

    // Fisher-Yates shuffle algorithm
    for (let i = pieces.length; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        container.insertBefore(pieces[j], pieces[i]);
    }
}

function submitImageUploadForm() {
    // Vérifiez si un fichier a été sélectionné
    const input = document.getElementById('imageUpload');
    if (input.files.length > 0) {
        // Créez un FormData et ajoutez le fichier
        const formData = new FormData();
        formData.append('file', input.files[0]);

        // Utilisez fetch API pour envoyer le fichier
        fetch('/upload-image', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            // Vérifiez que la réponse du serveur est ok (statut HTTP 200-299)
            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.statusText);
            }
            // Recharger la page après l'upload réussi
            window.location.reload();
        })
        .catch(error => {
            // Gérez les erreurs de réseau ou les erreurs de parsing JSON ici
            console.error('Error:', error);
        });
    }
}


document.addEventListener('DOMContentLoaded', function () {

    // Attachez l'événement 'change' à l'input de type fichier
    const imageUpload = document.getElementById('imageUpload');
    imageUpload.addEventListener('change', submitImageUploadForm);

const shuffleButton = document.getElementById('shuffle-button');
    shuffleButton.addEventListener('click', shufflePieces);

    
    // Array to track the state of the puzzle grid (0 for empty, piece ID for filled)
    const puzzleGrid = [0, 0, 0, 0, 0, 0, 0, 0, 0];

    // Variable to store the selected puzzle piece
    let selectedPieceIndex = null;
    
         

    // Add event listeners to each grid cell
    const gridCells = document.querySelectorAll('.grid-cell');
    gridCells.forEach((cell, index) => {
        cell.addEventListener('click', function () {
            handleCellClick(index);
        });
    });

 // Add event listeners to each puzzle piece
    const puzzlePieces = document.querySelectorAll('.puzzle-piece');
    puzzlePieces.forEach((piece, index) => {
        piece.addEventListener('click', function () {
            handlePieceClick(index);
        });

        // Initialize puzzle pieces with their images and dimensions
        const imageUrl = `/static/images/sliced_${Math.floor(index / 3)}_${index % 3}.png`;
        
        // Create a temporary image to get the original dimensions
        const tempImage = new Image();
        tempImage.src = imageUrl;
        tempImage.onload = function () {
            const originalWidth = tempImage.width;
            const originalHeight = tempImage.height;

            // Calculate the new dimensions for puzzle pieces
            const minDimension = 100;
            const scaleFactor = Math.max(minDimension / originalWidth, minDimension / originalHeight);
            const newWidth = originalWidth * scaleFactor;
            const newHeight = originalHeight * scaleFactor;

            piece.style.backgroundImage = `url('${imageUrl}')`;
            piece.style.backgroundSize = `cover`;
            piece.style.width = `${newWidth}px`;
            piece.style.height = `${newHeight}px`;

            // Set the same dimensions for grid cells
            const gridCells = document.querySelectorAll('.grid-cell');
            gridCells.forEach((cell) => {
                cell.style.width = `${newWidth}px`;
                cell.style.height = `${newHeight}px`;
            });
        };
    });


      

// Function to handle a click on a puzzle piece
function handlePieceClick(pieceIndex, event) {
   

    // Check if the clicked piece is already on the grid
    if (puzzleGrid.includes(pieceIndex + 1)) {
        // If yes, move the piece back to the row of available pieces
        puzzleGrid[puzzleGrid.indexOf(pieceIndex + 1)] = 0;
        updatePuzzle();
    } else {
        // If no, mark the selected piece (do not hide it yet)
        selectedPieceIndex = pieceIndex;
        updatePuzzle();
    }
}


// Function to handle a click on a grid cell
function handleCellClick(cellIndex) {
    // Check if a puzzle piece is selected
    if (selectedPieceIndex !== null) {
        // Move the selected piece to the clicked cell
        puzzleGrid[cellIndex] = selectedPieceIndex + 1; // Use piece ID instead of 1
        // Clear the selected piece
        selectedPieceIndex = null;
        updatePuzzle();
        checkVictory()
    } else {
        puzzleGrid[cellIndex] = 0;
        updatePuzzle();
    }
}




// Function to update the puzzle display based on the grid state
function updatePuzzle() {
    gridCells.forEach((cell, index) => {
        // Update the background image of each cell based on the grid state
        const pieceIndex = puzzleGrid[index] - 1;
        if (pieceIndex >= 0 && pieceIndex < puzzlePieces.length) {
            const imageUrl = `/static/images/sliced_${Math.floor(pieceIndex / 3)}_${pieceIndex % 3}.png`;
            cell.style.backgroundImage = `url('${imageUrl}')`;
            cell.style.backgroundSize = 'cover';  // Add this line to set background size to cover
        } else {
            cell.style.backgroundImage = 'none';
        }
    });

    puzzlePieces.forEach((piece, index) => {
        // Display puzzle piece if it's not in the grid and not selected
        piece.style.display = puzzleGrid.includes(index + 1) ? 'none' : 'block';
    });
}


    // Initial update to show puzzle pieces and empty grid cells
    updatePuzzle();

    
    // Function to check if the puzzle is in the correct order and replace with colored pieces
    function checkVictory() {
        const correctOrder = Array.from({ length: puzzleGrid.length }, (_, index) => index + 1);
        
        if (JSON.stringify(puzzleGrid) === JSON.stringify(correctOrder)) {
            // Replace puzzle with full colored pieces
            gridCells.forEach((cell, index) =>{
                const imageUrl = `/static/images/sliced_color_${Math.floor(index / 3)}_${index % 3}.png`;
                cell.style.backgroundImage = `url('${imageUrl}')`;
                cell.style.backgroundSize = 'cover';
            });
        }
    }
    
    
});

function downloadColorizedImage() {
    // URL directe vers l'image colorisée stockée dans /static/images
    const downloadUrl = '/static/images/colorized_image.png';

    // Crée un élément 'a' pour déclencher le téléchargement
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = 'colorized.png'; // Nom du fichier à télécharger

    // Simule un clic sur le lien pour déclencher le téléchargement
    document.body.appendChild(link);
    link.click();

    // Nettoyage : Supprime le lien après le téléchargement
    document.body.removeChild(link);
}



