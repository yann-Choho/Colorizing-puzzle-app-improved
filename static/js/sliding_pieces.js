// static/js/sliding_pieces.js

// Declare puzzlePieces globally
let puzzlePieces;

// Function to initialize puzzlePieces
function initializePuzzlePieces() {
    puzzlePieces = document.querySelectorAll('.puzzle-piece');
}

// Call the initialization function
initializePuzzlePieces();


document.addEventListener('DOMContentLoaded', function () {

// The grid of the puzzle
const puzzleContainer = document.getElementById('puzzle-container');

// Check if all the non-empty pieces' id and order are consistent
function isVictory() {
    const orderedPieces = Array.from(puzzlePieces).sort((a, b) => a.order - b.order);
    const isCorrectOrder = orderedPieces.every((piece, index) => {
        const expectedOrder = parseInt(piece.dataset.order);
        const actualOrder = index;
        console.log(`Piece ${piece.id}: Expected Order: ${expectedOrder}, Actual Order: ${actualOrder}`);
        return expectedOrder === actualOrder;
    });

    console.log('isCorrectOrder:', isCorrectOrder);

    return isCorrectOrder;
}

// Specify what happens when a piece is (left) clicked
    puzzlePieces.forEach(piece => {
        piece.addEventListener('click', function () {
        // Check if the piece is adjacent to the empty piece
            const emptyCell = document.querySelector('.empty');
            if (isAdjacent(piece, emptyCell)) {
            // If adjacent, call the swapping function
                swapPieces(piece, emptyCell);
                // check if victory is achieved
                if (isVictory()) {
            // If yes, replace puzzle with full original picture
                        puzzlePieces.forEach(piece => {
        const row = parseInt(piece.dataset.row);
        const col = parseInt(piece.dataset.col);
        piece.style.backgroundImage = `url('/static/images/sliced_color_${row}_${col}.png')`;
        piece.style.backgroundSize = 'cover';
    });
                }
            }
        });
    });

    function isAdjacent(piece, empty) {
        const pieceRow = parseInt(piece.dataset.row);
        const pieceCol = parseInt(piece.dataset.col);
        const emptyRow = parseInt(empty.dataset.row);
        const emptyCol = parseInt(empty.dataset.col);

        return (
            (pieceRow === emptyRow && Math.abs(pieceCol - emptyCol) === 1) ||
            (pieceCol === emptyCol && Math.abs(pieceRow - emptyRow) === 1)
        );
    }

    function swapPieces(piece1, empty) {
        const order1 = piece1.style.order;
        const row1 = piece1.dataset.row;
        const col1 = piece1.dataset.col;
        piece1.style.order = empty.style.order;
        piece1.dataset.row = empty.dataset.row;
        piece1.dataset.col = empty.dataset.col;
        empty.dataset.row = row1;
        empty.dataset.col = col1;
        empty.style.order = order1;
        
        // Update dataset.order after swapping
        const tempOrder = piece1.dataset.order;
        piece1.dataset.order = empty.dataset.order;
        empty.dataset.order = tempOrder;
    }
    

});


    // Function called by the html button "Shuffle"
    function shufflePieces() {
        // Check if puzzlePieces is defined
        // Sélectionnez toutes les pièces du puzzle
        let puzzlePieces = document.querySelectorAll('.puzzle-piece');

        // Convertissez la NodeList en tableau pour faciliter le mélange
        let piecesArray = Array.from(puzzlePieces);

        // Mélangez les pièces en utilisant l'algorithme Fisher-Yates
        for (let i = piecesArray.length - 1; i > 0; i--) {
            let j = Math.floor(Math.random() * (i + 1)); // Nombre aléatoire entre 0 et i

            // Échangez les pièces
            [piecesArray[i].style.order, piecesArray[j].style.order] = [piecesArray[j].style.order, piecesArray[i].style.order];
        }
        // Special case : shuffle after a victory
        let downloadButton = document.getElementById('downloadButton');
        if (downloadButton) {
            downloadButton.style.display = 'none';
        }


// Fonction pour télécharger l'image
function downloadImage() {
    // URL directe vers l'image colorisée stockée dans /static/images
    const downloadUrl = '/static/images/colorized_image.png';

    // Crée un élément 'a' pour déclencher le téléchargement
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = 'colorized_image.png'; // Nom du fichier à télécharger

    // Simule un clic sur le lien pour déclencher le téléchargement
    document.body.appendChild(link);
    link.click();

    // Nettoyage : Supprime le lien après le téléchargement
    document.body.removeChild(link);
}

// Check if victory is achieved
if (isVictory()) {
    // Affichez l'image colorisée pour chaque pièce du puzzle
    puzzlePieces.forEach(piece => {
        const row = parseInt(piece.dataset.row);
        const col = parseInt(piece.dataset.col);
        piece.style.backgroundImage = `url('/static/images/sliced_color_${row}_${col}.png')`;
        piece.style.backgroundSize = 'cover';
    });

    // Afficher le bouton de téléchargement
    document.getElementById('downloadButton').style.display = 'block';
}

    // The main piece movement mechanism
    // but inside the ShufflePieces function
        function swapPieces(piece1, empty) {
            const order1 = piece1.style.order;
            const row1 = piece1.dataset.row;
            const col1 = piece1.dataset.col;
            piece1.style.order = empty.style.order;
            piece1.dataset.row = empty.dataset.row;
            piece1.dataset.col = empty.dataset.col;
            empty.dataset.row = row1;
            empty.dataset.col = col1;
            empty.style.order = order1;
            
            // Update dataset.order after swapping
            const tempOrder = piece1.dataset.order;
            piece1.dataset.order = empty.dataset.order;
            empty.dataset.order = tempOrder;
        }

    // Fisher-Yates shuffle algorithm
    for (let i = puzzlePieces.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        swapPieces(puzzlePieces[i], puzzlePieces[j]);
    }

   
}