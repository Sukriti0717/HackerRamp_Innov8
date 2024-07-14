document.addEventListener('DOMContentLoaded', function () {
    const productContainer = document.getElementById('product-container');
    const products = Array.from(document.querySelectorAll('.product-card'));
    const yayButton = document.getElementById('yay-button');
    const nayButton = document.getElementById('nay-button');

    let currentProductIndex = 0;

    // Hide all products initially except the first one
    products.forEach((product, index) => {
        if (index !== currentProductIndex) {
            product.style.display = 'none';
        }
    });

    function handleAction(action) {
        const product = products[currentProductIndex];

        if (action === 'interested') {
            product.style.transform = 'translateX(100%)';
            sendSwipeData(product.dataset.id, 'interested');
        } else if (action === 'not_interested') {
            product.style.transform = 'translateX(-100%)';
            sendSwipeData(product.dataset.id, 'not_interested');
        }

        setTimeout(() => {
            product.style.display = 'none';
            product.style.transform = 'none';

            currentProductIndex++;
            if (currentProductIndex >= products.length) {
                currentProductIndex = 0; // Loop back to the first product or handle end of products
            }

            const nextProduct = products[currentProductIndex];
            nextProduct.style.display = 'flex';
        }, 300); // Delay to allow swipe animation to complete
    }

    function sendSwipeData(productId, action) {
        fetch('/swipe', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ productId: productId, action: action })
        })
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
    }

    yayButton.addEventListener('click', () => handleAction('interested'));
    nayButton.addEventListener('click', () => handleAction('not_interested'));
});
