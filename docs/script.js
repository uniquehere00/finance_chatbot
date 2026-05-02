console.log("script.js v3 loaded");

const API_URL = "https://web-production-bf42a.up.railway.app";
let currentSessionId = null;
let isProcessing = false;

window.onload = async () => {
    console.log("Page loaded");
    await loadSampleButtons();
};

async function loadSampleButtons() {
    try {
        const res = await fetch(`${API_URL}/samples`);
        const data = await res.json();
        if (data.samples && data.samples.length > 0) {
            const row = document.getElementById("samples-row");
            row.innerHTML = "";
            data.samples.forEach(sample => {
                const btn = document.createElement("button");
                btn.className = "btn btn-sample";
                btn.textContent = `📈 ${sample.display_name}`;
                btn.onclick = () => loadSample(sample.filename);
                row.appendChild(btn);
            });
        }
    } catch (err) {
        console.log("Could not load samples:", err);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    document.getElementById("upload-box").classList.add("dragover");
}

function handleDragLeave(e) {
    document.getElementById("upload-box").classList.remove("dragover");
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById("upload-box").classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
}

function handleFileSelect(e) {
    console.log("handleFileSelect triggered");
    const file = e.target.files[0];
    console.log("File:", file?.name);
    if (file) uploadFile(file);
}

function resetFileInput() {
    document.getElementById("file-input").value = "";
}

async function uploadFile(file) {
    console.log("uploadFile called:", file.name);
    resetFileInput();

    if (!file.name.endsWith(".pdf")) {
        alert("Please upload a PDF file.");
        return;
    }

    showProcessing(`Uploading ${file.name}...`);
    setStep(1);

    const formData = new FormData();
    formData.append("file", file);

    try {
        console.log("Sending upload request...");
        const res = await fetch(`${API_URL}/upload`, {
            method: "POST",
            body: formData
        });

        console.log("Upload response status:", res.status);

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Upload failed");
        }

        const data = await res.json();
        console.log("Upload data:", data);
        setStep(2);

        await pollUntilReady(data.session_id, data.pdf_name);

    } catch (err) {
        console.error("Upload error:", err);
        hideProcessing();
        alert(`Error: ${err.message}`);
    }
}

async function loadSample(filename) {
    console.log("loadSample called:", filename);
    showProcessing(`Loading sample document...`);
    setStep(1);

    try {
        const res = await fetch(`${API_URL}/load-sample`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Failed to load sample");
        }

        const data = await res.json();
        console.log("Sample response:", data);
        setStep(2);

        await pollUntilReady(data.session_id, data.pdf_name);

    } catch (err) {
        console.error("Sample error:", err);
        hideProcessing();
        alert(`Error: ${err.message}`);
    }
}

async function pollUntilReady(sessionId, pdfName) {
    console.log("Polling session:", sessionId);

    const maxAttempts = 100;
    let attempts = 0;

    const messages = [
        "Extracting content from PDF...",
        "Detecting tables and text...",
        "Converting tables to natural language...",
        "Generating embeddings...",
        "Building search index...",
        "Almost ready..."
    ];

    while (attempts < maxAttempts) {
        await sleep(3000);
        attempts++;

        const msgIndex = Math.min(Math.floor(attempts / 4), messages.length - 1);
        document.getElementById("processing-title").textContent = messages[msgIndex];

        if (attempts > 2) setStep(3);
        if (attempts > 6) setStep(4);

        try {
            const res = await fetch(`${API_URL}/status/${sessionId}`);
            const status = await res.json();
            console.log(`Poll ${attempts}:`, status.status);

            if (status.status === "ready") {
                console.log("Ready! Starting chat...");
                await sleep(500);
                startChat(sessionId, status.pdf_name, status.chunks_created);
                return;
            }

            if (status.status === "error") {
                throw new Error(status.message || "Processing failed");
            }

        } catch (err) {
            if (err.message.includes("Processing failed")) throw err;
            console.log("Poll network error, retrying:", err.message);
        }
    }

    throw new Error("Processing timed out. Try a smaller PDF.");
}

function startChat(sessionId, pdfName, chunksCreated) {
    console.log("startChat called:", sessionId, pdfName, chunksCreated);
    currentSessionId = sessionId;

    document.getElementById("doc-name").textContent = pdfName;
    document.getElementById("doc-chunks").textContent =
        `${chunksCreated} chunks indexed`;

    document.getElementById("upload-screen").classList.remove("active");
    document.getElementById("upload-screen").classList.add("hidden");
    document.getElementById("chat-screen").classList.remove("hidden");
    document.getElementById("chat-screen").classList.add("active");

    document.getElementById("question-input").focus();
}

async function sendQuestion() {
    console.log("Sending with session:", currentSessionId);
    if (isProcessing) return;

    const input = document.getElementById("question-input");
    const question = input.value.trim();
    if (!question) return;

    if (!currentSessionId) {
        alert("No document loaded. Please upload a document first.");
        return;
    }

    input.value = "";
    autoResize(input);

    const chips = document.getElementById("suggestion-chips");
    if (chips) chips.style.display = "none";

    addMessage("user", question);
    const thinkingId = showThinking();
    isProcessing = true;
    document.getElementById("send-btn").disabled = true;

    try {
        const res = await fetch(`${API_URL}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: currentSessionId,
                question: question
            })
        });

        removeThinking(thinkingId);

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Failed to get answer");
        }

        const data = await res.json();
        addBotMessage(data.answer, data.sources);

    } catch (err) {
        removeThinking(thinkingId);
        addMessage("bot", `Sorry, something went wrong: ${err.message}`);
    } finally {
        isProcessing = false;
        document.getElementById("send-btn").disabled = false;
        document.getElementById("question-input").focus();
    }
}

function addMessage(role, text) {
    const container = document.getElementById("chat-messages");
    const msg = document.createElement("div");
    msg.className = `message ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = text;
    msg.appendChild(bubble);
    container.appendChild(msg);
    scrollToBottom();
}

function addBotMessage(answer, sources) {
    const container = document.getElementById("chat-messages");
    const msg = document.createElement("div");
    msg.className = "message bot";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = answer;
    msg.appendChild(bubble);

    if (sources && sources.length > 0) {
        const toggleBtn = document.createElement("button");
        toggleBtn.className = "sources-toggle";
        toggleBtn.textContent =
            `📎 ${sources.length} sources retrieved · Click to view`;

        const sourcesList = document.createElement("div");
        sourcesList.className = "sources-list";

        sources.forEach(source => {
            const item = document.createElement("div");
            item.className = "source-item";
            item.innerHTML = `
                <div class="source-header">
                    <span class="source-label">
                        Source ${source.rank} · ${source.source} · Page ${source.page || "N/A"}
                    </span>
                    <span class="source-score">
                        ${(source.similarity * 100).toFixed(0)}% match
                    </span>
                </div>
                <div class="source-preview">${source.preview}</div>
            `;
            sourcesList.appendChild(item);
        });

        toggleBtn.onclick = () => {
            sourcesList.classList.toggle("visible");
            toggleBtn.textContent = sourcesList.classList.contains("visible")
                ? `📎 ${sources.length} sources retrieved · Click to hide`
                : `📎 ${sources.length} sources retrieved · Click to view`;
        };

        msg.appendChild(toggleBtn);
        msg.appendChild(sourcesList);
    }

    container.appendChild(msg);
    scrollToBottom();
}

function showThinking() {
    const container = document.getElementById("chat-messages");
    const id = "thinking-" + Date.now();
    const msg = document.createElement("div");
    msg.className = "message bot";
    msg.id = id;
    const thinking = document.createElement("div");
    thinking.className = "thinking";
    thinking.innerHTML = "<span></span><span></span><span></span>";
    msg.appendChild(thinking);
    container.appendChild(msg);
    scrollToBottom();
    return id;
}

function removeThinking(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function askSuggestion(btn) {
    document.getElementById("question-input").value = btn.textContent.trim();
    sendQuestion();
}

function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
}

function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
}

function showProcessing(title) {
    document.querySelector(".upload-container").classList.add("hidden");
    document.getElementById("processing-state").classList.remove("hidden");
    document.getElementById("processing-title").textContent = title;
}

function hideProcessing() {
    document.querySelector(".upload-container").classList.remove("hidden");
    document.getElementById("processing-state").classList.add("hidden");
    resetSteps();
}

function setStep(n) {
    for (let i = 1; i <= 4; i++) {
        const step = document.getElementById(`step-${i}`);
        if (i < n) step.className = "step done";
        else if (i === n) step.className = "step active";
        else step.className = "step";
    }
}

function resetSteps() {
    for (let i = 1; i <= 4; i++) {
        document.getElementById(`step-${i}`).className = "step";
    }
}

function newConversation() {
    if (!currentSessionId) return;
    fetch(`${API_URL}/reset/${currentSessionId}`, { method: "POST" });
    document.getElementById("chat-messages").innerHTML = `
        <div class="welcome-message">
            <p>Conversation reset. Ask me anything about the document.</p>
            <div class="suggestion-chips" id="suggestion-chips">
                <button class="chip" onclick="askSuggestion(this)">What was the revenue?</button>
                <button class="chip" onclick="askSuggestion(this)">What is the operating margin?</button>
                <button class="chip" onclick="askSuggestion(this)">Summarize key highlights</button>
                <button class="chip" onclick="askSuggestion(this)">What was the YoY growth?</button>
            </div>
        </div>
    `;
}

function uploadNew() {
    if (currentSessionId) {
        fetch(`${API_URL}/session/${currentSessionId}`, { method: "DELETE" });
        currentSessionId = null;
    }
    document.getElementById("chat-screen").classList.add("hidden");
    document.getElementById("chat-screen").classList.remove("active");
    document.getElementById("upload-screen").classList.remove("hidden");
    document.getElementById("upload-screen").classList.add("active");
    hideProcessing();
    resetFileInput();
}

function scrollToBottom() {
    const container = document.getElementById("chat-messages");
    container.scrollTop = container.scrollHeight;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}