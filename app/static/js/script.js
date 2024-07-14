body {
    font-family: Arial, sans-serif;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background-color: #f0f0f0;
}

#product-container {
    width: 300px;
    height: 500px;
    position: relative;
    overflow: hidden;
}

.product-card {
    width: 100%;
    height: 100%;
    position: absolute;
    top: 0;
    left: 0;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    transition: transform 0.5s ease-in-out;
}

.product-card img {
    max-width: 100%;
    max-height: 60%;
    border-radius: 8px 8px 0 0;
}

.product-card h2 {
    margin: 16px 0;
}

#buttons-container {
    margin-top: 20px;
    display: flex;
    justify-content: space-between;
    width: 200px;
}

.swipe-button {
    padding: 10px 20px;
    font-size: 16px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

#nay-button {
    background-color: #ffcccc;
    color: #ff0000;
}

#nay-button:hover {
    background-color: #ff9999;
}

#yay-button {
    background-color: #ccffcc;
    color: #00ff00;
}

#yay-button:hover {
    background-color: #99ff99;
}
