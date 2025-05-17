

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


document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("new_pdf");
    const uploadBtn = document.getElementById("uploadBtn");
    const fileSelect = document.getElementById("file_name");
    const statusDiv = document.getElementById("uploadStatus");

    // Enable Upload button only when a file is selected
    fileInput.addEventListener("change", () => {
        uploadBtn.disabled = !fileInput.files.length;
        statusDiv.classList.add("d-none"); // Hide previous status when new file selected
    });

    // Upload file on click
    uploadBtn.addEventListener("click", async () => {
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        uploadBtn.disabled = true;

        // Show "uploading..." status
        statusDiv.textContent =`⏳ Uploading "${file.name}"...`;
        statusDiv.className = "form-text text-info mt-1";
        statusDiv.classList.remove("d-none");

        try {
            const res = await fetch("/upload_file", {
                method: "POST",
                body: formData
            });

            const data = await res.json();

            if (data.status === "success") {
                fileInput.value = "";
                uploadBtn.disabled = true;

                // Update status
                statusDiv.textContent = `✅ File "${file.name}" uploaded and added to the list.`;
                statusDiv.className = "form-text text-success mt-1";

            } else {
                statusDiv.textContent = "❌ Upload failed: " + data.message;
                statusDiv.className = "form-text text-danger mt-1";
            }

        } catch (err) {
            statusDiv.textContent = "❌ Error: " + err.message;
            statusDiv.className = "form-text text-danger mt-1";
        }
    });
});
   