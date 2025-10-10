// BrandCompass.ai - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('BrandCompass.ai initialized');

    initializeAnimations();
    initializeTooltips();
});

function initializeAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);

    document.querySelectorAll('.card').forEach(card => {
        observer.observe(card);
    });
}

function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');

    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            showTooltip(this);
        });

        element.addEventListener('mouseleave', function() {
            hideTooltip();
        });
    });
}

function showTooltip(element) {
    const tooltipText = element.getAttribute('data-tooltip');
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = tooltipText;

    document.body.appendChild(tooltip);

    const rect = element.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
}

function hideTooltip() {
    const tooltips = document.querySelectorAll('.tooltip');
    tooltips.forEach(tooltip => tooltip.remove());
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.classList.add('show');
    }, 100);

    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

window.showNotification = showNotification;
window.debounce = debounce;

// Chat Widget
document.addEventListener('DOMContentLoaded', function() {
    const chatIcon = document.getElementById('chat-icon');
    const chatWidget = document.getElementById('chat-widget');
    const closeChat = document.getElementById('close-chat');
    const chatBody = document.getElementById('chat-body');
    const chatInput = document.getElementById('chat-input');
    const sendChat = document.getElementById('send-chat');
    const refreshChat = document.getElementById('refresh-chat');
    const chatBackdrop = document.getElementById('chat-backdrop');

    let chatHistory = [];

    chatIcon.addEventListener('click', () => {
        chatWidget.style.display = 'flex';
        chatIcon.style.display = 'none';
        chatBackdrop.classList.add('show');
    });

    closeChat.addEventListener('click', () => {
        chatWidget.style.display = 'none';
        chatIcon.style.display = 'flex';
        chatBackdrop.classList.remove('show');
    });

    refreshChat.addEventListener('click', () => {
        if (confirm('Clear chat history? This will start a fresh conversation.')) {
            clearChat();
        }
    });

    // Click on backdrop to close chat
    chatBackdrop.addEventListener('click', () => {
        chatWidget.style.display = 'none';
        chatIcon.style.display = 'flex';
        chatBackdrop.classList.remove('show');
    });

    sendChat.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        appendMessage(message, 'user');
        chatInput.value = '';

        // Show engaging loading message
        const loadingMessages = [
            '🔍 Analyzing your request...',
            '📊 Crunching the numbers...',
            '🤔 Thinking...',
            '📈 Fetching forecast data...',
            '⚡ Processing...',
            '🎯 Working on it...'
        ];
        const loadingMsg = loadingMessages[Math.floor(Math.random() * loadingMessages.length)];
        const loadingElement = appendLoadingMessage(loadingMsg);

        const selectedCountry = document.body.dataset.selectedCountry;

        fetch('/api/v1/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                message: message, 
                chat_history: chatHistory,
                selected_country: selectedCountry
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading message
            if (loadingElement && loadingElement.parentNode) {
                loadingElement.remove();
            }
            
            appendMessage(data.response, 'assistant');
            chatHistory.push({ role: 'user', content: message });
            chatHistory.push({ role: 'assistant', content: data.response });
        })
        .catch(error => {
            // Remove loading message
            if (loadingElement && loadingElement.parentNode) {
                loadingElement.remove();
            }
            
            appendMessage('Sorry, something went wrong. Please try again.', 'assistant');
            console.error('Error:', error);
        });
    }

    function appendLoadingMessage(message) {
        const loadingElement = document.createElement('div');
        loadingElement.classList.add('chat-loading');
        loadingElement.innerHTML = `
            <div class="spinner"></div>
            <span>${message}</span>
        `;
        chatBody.appendChild(loadingElement);
        chatBody.scrollTop = chatBody.scrollHeight;
        return loadingElement;
    }

    function appendMessage(message, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', sender + '-message');

        if (sender === 'assistant') {
            const converter = new showdown.Converter();
            messageElement.innerHTML = converter.makeHtml(message);
        } else {
            messageElement.textContent = message;
        }

        chatBody.appendChild(messageElement);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    function clearChat() {
        // Clear the chat history
        chatHistory = [];
        
        // Clear the chat body
        chatBody.innerHTML = '';
        
        // Add a system message
        const systemMessage = document.createElement('div');
        systemMessage.classList.add('chat-message', 'system-message');
        systemMessage.textContent = '🔄 Chat history cleared. Starting fresh conversation!';
        chatBody.appendChild(systemMessage);
        
        // Optional: notify backend to clear server-side memory
        fetch('/api/v1/chat/clear', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        }).catch(error => {
            console.log('Note: Server-side memory clear not implemented yet');
        });
    }
});
