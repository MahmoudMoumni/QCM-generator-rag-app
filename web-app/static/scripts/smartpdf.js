

document.getElementById("chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const chatBox = document.getElementById("chat-box");
    const formData = new FormData(e.target);
    const user_input = formData.get("user_input");

    // Add user message to UI
    const userBubble = document.createElement("div");
    userBubble.className = "chat-bubble user-msg";
    userBubble.textContent = user_input;
    chatBox.appendChild(userBubble);
    chatBox.scrollTop = chatBox.scrollHeight;

    const typingIndicator = document.getElementById("typing-indicator");
    
    function showTypingIndicator() {
      chatBox.appendChild(typingIndicator);  // move it to the bottom
      typingIndicator.style.display = "flex";
    }
    
    function hideTypingIndicator() {
      typingIndicator.style.display = "none";
    }

    // Clear input
    e.target.reset();
    showTypingIndicator(); // Always moves it to the bottom and shows it
    // Scroll to bottom after slight delay to ensure it's visible
    setTimeout(() => {
        chatBox.scrollTop = chatBox.scrollHeight;
    }, 50);
    // Send message to backend
    const response = await fetch("/chat", {
        method: "POST",
        body: JSON.stringify({ user_input }),
        headers: { "Content-Type": "application/json" }
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    const botBubble = document.createElement("div");
    botBubble.className = "chat-bubble bot-msg";
    chatBox.appendChild(botBubble);

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        botBubble.textContent += decoder.decode(value, { stream: true });
        chatBox.scrollTop = chatBox.scrollHeight;
    }
    hideTypingIndicator();
});

document.getElementById("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const status = document.getElementById("upload-status");
    const formData = new FormData(e.target);

    const response = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    const text = await response.text();
    status.textContent = text;
});


let pdfDoc = null;

async function loadPdfAndShowPage(pdfUrl, pageNum) {
  const loadingTask = pdfjsLib.getDocument(pdfUrl);
  pdfDoc = await loadingTask.promise;
  const page = await pdfDoc.getPage(pageNum);

  const viewport = page.getViewport({ scale: 1.5 });
  const canvas = document.getElementById("pdf-canvas");
  const context = canvas.getContext("2d");

  canvas.height = viewport.height;
  canvas.width = viewport.width;

  await page.render({ canvasContext: context, viewport }).promise;
}
