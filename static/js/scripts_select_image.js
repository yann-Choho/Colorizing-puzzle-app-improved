function selectImage(imageName) {
    fetch('/set-puzzle-image', {
        method: 'POST',
        body: JSON.stringify({ image: imageName }),
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => response.json())
    .then(data => {
        window.location.href = data.redirect_url; // Redirige vers la page précédente
    });
}
