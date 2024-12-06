let botIconDefault = `
    <div class="bot-icon">
        <i class="fa-solid fa-robot"></i>
    </div>
`;
let botImageSpecial = '/static/bot_image.jpg';
let currentBotIcon = botIconDefault;

function now() {
    const now = new Date();
  
    const pad = (num) => String(num).padStart(2, '0');
  
    const day = pad(now.getDate());
    const month = pad(now.getMonth() + 1); // Months are 0-based
    const year = now.getFullYear();
  
    const hours = pad(now.getHours());
    const minutes = pad(now.getMinutes());
    const seconds = pad(now.getSeconds());
  
    return `${day}/${month}/${year} ${hours}:${minutes}:${seconds}`;
}
  
function debug(message) {
    console.log(`[DEBUG] [${now()}] ${message}`)
}

function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") return;

    const outputDiv = document.getElementById("chat-output");

    // Adicionar mensagem do usuário com ícone Font Awesome
    outputDiv.innerHTML += `
        <div class="message-container" style="justify-content: flex-end;">
            <div class="user-message">${userInput}</div>
            <div class="user-icon">
                <i class="fa-solid fa-user"></i>
            </div>
        </div>
    `;

    document.getElementById("user-input").value = "";

    fetch('/response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        let botMessageHTML = `
            <div class="message-container" style="justify-content: flex-start;">
                ${botIconDefault}
                <div class="bot-message">${data.response}</div>
            </div>
        `;

        outputDiv.innerHTML += botMessageHTML;
        outputDiv.scrollTop = outputDiv.scrollHeight;
    })
    .catch(error => console.error("Erro:", error));
}

document.addEventListener("DOMContentLoaded", function () {
    const dropdown = document.getElementById("dropdown");
    const startChatButton = document.getElementById("start-chat");
    const modelSelectorContainer = document.getElementById("model-selector-container");
    const loadingContainer = document.getElementById("loading-container");
    const chatContainer = document.getElementById("chat-container");

    // Enable the button (to go to chat) after selecting a model.
    dropdown.addEventListener("change", () => {
        startChatButton.disabled = dropdown.value === "";
    });

    // Show the chat and hide the dropdown (model selector).
    startChatButton.addEventListener("click", () => {
        modelSelectorContainer.classList.add("hidden");
        loadingContainer.classList.remove("hidden");

        fetch('/select_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 'model_name': dropdown.options[dropdown.selectedIndex].innerText })
        })
        .then(response => response.json())
        .then(data => {
            loadingContainer.classList.add("hidden");
            chatContainer.classList.remove("hidden");
        })
        .catch(error => console.error("Erro:", error));
    });
});

document.addEventListener('keydown', (event) => {
    // Keybind: ALT+A. Send a request to the server to reset the chatbot knowledge.
    if (event.altKey === true && event.key == 'a') {
        debug('Sending a request for reset chatbot knowledge.')

        // Request to the server to reset the chatbot knowledge.
        fetch('/reset_chatbot_knowledge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        })
        .then(response => response.json())
        .then(data => {
            const outputDiv = document.getElementById("chat-output");
            outputDiv.innerHTML = '';
            debug('Chatbot knowledge has been reset.')
        })
        .catch(error => console.error("Error: ", error));
    }
});

// Add a event listener that when the `Enter` is pressed, the user message is sent.
document.getElementById("user-input").addEventListener("keypress", (event) => {
    if (event.key === "Enter")
        // Send the message to the server.
        sendMessage();
});