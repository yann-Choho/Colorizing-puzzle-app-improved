function selectImage(imageName) {
    // Envoyer le nom de l'image sélectionnée au serveur
    fetch('/set-puzzle-image', {
        method: 'POST',
        body: JSON.stringify({ image: imageName }),
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => {
        if (response.ok) {
            window.location.href = '/sliding_pieces'; // Redirige vers la page du puzzle
        }
    }).catch(error => {
        console.error('Error:', error);
    });
}