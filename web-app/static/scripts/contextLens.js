

    // Function to handle click event
    async function handleResultClick(doc_id, page, content) {
        // Decode the content text
        const decodedContent = decodeURIComponent(content);
        
        console.log("Document ID:", doc_id);
        console.log("Page:", page);
        console.log("Content:", decodedContent);
        try {
            const response = await fetch("/get-doc-url", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ doc_id: parseInt(doc_id) })
            });
    
            if (response.ok) {
                const data = await response.json();
                console.log(data);
                url=data["doc_url"];
                pageNum = parseInt(page)+1;//we add 1 because in books the count starts from 1
                pdfDoc=null;
                pdfjsLib.getDocument(url).promise.then((doc) => {
                    pdfDoc = doc;
                    document.getElementById('page-count').textContent = doc.numPages;
                    renderPage(pageNum);
                });
                console.log("success getting doc");
            } else {
                alert("Failed to get the document URL.");
            }
        } catch (error) {
            console.error("Error:", error);
        }
        
    }
document.addEventListener("DOMContentLoaded", function () {

    document.getElementById("search-form").addEventListener("submit", async (e) => {
        e.preventDefault();

        const submitButton = document.getElementById("search-submit-btn");
        const resultsDiv = document.getElementById("search_results_div");
    
        // Disable the button and show loading spinner
        submitButton.disabled = true;
        
        resultsDiv.innerHTML = `
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>Please wait while we search...</p>
        `;


        const formData = new FormData(e.target);
        const user_input = formData.get("user_input");
        try{
            // Send message to backend
            const response = await fetch("/search", {
                method: "POST",
                body: JSON.stringify({ user_input }),
                headers: { "Content-Type": "application/json" }
            });
            console.log(response)
            if (response.ok) {
                console.log("hello")
                const data = await response.json();
                console.log("Search Results:", data.search_results); // Display search results in console
                displayResults(data.search_results); // Call function to display results
            } else {
                resultsDiv.innerHTML = `<p class="text-danger">Error in search request.</p>`;
                alert("Error in search request.");
            }
        }catch (error) {
            resultsDiv.innerHTML = `<p class="text-danger">Failed to connect. Please try again.</p>`;
        } finally {
            // Re-enable the submit button
            submitButton.disabled = false;
        }

    });


    // Function to display search results dynamically
    function displayResults(results) {
        const resultsDiv = document.getElementById("search_results_div");
        resultsDiv.innerHTML = ""; // Clear previous results

        if (results.length === 0) {
            resultsDiv.innerHTML = "<p>No results found.</p>";
            return;
        }

        results.forEach(result => {
            const label = document.createElement("label");
            label.innerHTML = `
                <input type="checkbox" hidden>
                <button class="suggestion-btn" onclick="handleResultClick('${result.doc_id}', '${result.page}', '${encodeURIComponent(result.content)}')">
                    doc_id: ${result.doc_id} -- page: ${result.page} -- content: ${result.content}
                </button>
                <div class="expanded-text">${JSON.stringify(result)}</div>
            `;
            resultsDiv.appendChild(label);
        });
    }

    const fileInput = document.getElementById("new_pdf");
    const uploadBtn = document.getElementById("uploadBtn");
    const fileSelect = document.getElementById("file_name");
    const statusDiv = document.getElementById("uploadStatus");

    // Enable Upload button only when a file is selected
    fileInput.addEventListener("change", () => {
        console.log(fileInput.files.length)
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

let pdfDoc = null, pageNum = 1;

const canvas = document.getElementById('pdf-canvas');
const ctx = canvas.getContext('2d');



function renderPage(num) {
    pdfDoc.getPage(num).then((page) => {
        const viewport = page.getViewport({ scale: 1.5 });
        canvas.height = viewport.height;
        canvas.width = viewport.width;

        const renderContext = {
        canvasContext: ctx,
        viewport: viewport
        };

        page.render(renderContext);
        document.getElementById('page-num').textContent = num;
    });
}

document.getElementById('prev-page').addEventListener('click', () => {
if (pageNum <= 1) return;
pageNum--;
renderPage(pageNum);
});

document.getElementById('next-page').addEventListener('click', () => {
if (pageNum >= pdfDoc.numPages) return;
pageNum++;
renderPage(pageNum);
});

async function fetchDocumentUrl() {
    const docSelect = document.getElementById("file_name");
    const docId = docSelect.value;
    console.log(docId)

    if (!docId) return; // If no document selected

    try {
        const response = await fetch("/get-doc-url", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ doc_id: docId })
        });

        if (response.ok) {
            const data = await response.json();
            console.log(data);
            url=data["doc_url"];
            pageNum = 1;
            pdfDoc=null;
            pdfjsLib.getDocument(url).promise.then((doc) => {
                pdfDoc = doc;
                document.getElementById('page-count').textContent = doc.numPages;
                renderPage(pageNum);
            });
            console.log("success getting doc");
        } else {
            alert("Failed to get the document URL.");
        }
    } catch (error) {
        console.error("Error:", error);
    }
}