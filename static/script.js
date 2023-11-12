// script.js

// Declare puzzlePieces globally
let puzzlePieces;

// Function to initialize puzzlePieces
function initializePuzzlePieces() {
    puzzlePieces = document.querySelectorAll('.puzzle-piece');
}

// Call the initialization function
initializePuzzlePieces();


document.addEventListener('DOMContentLoaded', function () {

    const puzzleContainer = document.getElementById('puzzle-container');
    //const puzzlePieces = document.querySelectorAll('.puzzle-piece');

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

    puzzlePieces.forEach(piece => {
        piece.addEventListener('click', function () {
            const emptyCell = document.querySelector('.empty');
            if (isAdjacent(piece, emptyCell)) {
                swapPieces(piece, emptyCell);

                if (isVictory()) {
                                    // Replace puzzle with full original picture
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



function shufflePieces() {
    // Check if puzzlePieces is defined
    if (!puzzlePieces) {
        console.error('Error: puzzlePieces is not defined.');
        return;
    }
    
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
    
     if (isVictory()) {
        // Reset any victory-related changes
        puzzlePieces.forEach(piece => {
            piece.style.backgroundImage = `url('/static/images/sliced_${piece.dataset.row}_${piece.dataset.col}.png')`;
            piece.style.backgroundSize = 'cover';

            // If piece has id=0, set it to be a blank cell
            if (piece.dataset.id === '0') {
                piece.style.backgroundImage = 'none'; // Set it to be empty
                // You can also set a background color or other styles to represent a blank cell
            }
        });
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

    // Fisher-Yates shuffle algorithm
    for (let i = puzzlePieces.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        swapPieces(puzzlePieces[i], puzzlePieces[j]);
    }

   
}