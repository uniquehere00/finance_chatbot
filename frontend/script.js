console.log("FinSight v5 loaded");

const API_URL = "https://web-production-bf42a.up.railway.app";
let currentSessionId = null;
let isProcessing = false;

// ===== INIT =====
window.onload = async () => {
    // Warm up backend silently
    fetch(`${API_URL}/health`).catch(() => {});
    await loadSampleButtons();
};

// ===== LOAD SAMPLE BUTTONS =====
async function loadSampleButtons() {
    try {
        const res = await fetch(`${API_URL}/samples`);
        const data = await res.json();
        if (data.samples && data.samples.length > 0) {
            const row = document.getElementById("samples-row");
            row.innerHTML = "";
            data.samples.forEach(sample => {
                const btn = document.createElement("button");
                btn.className = "sample-btn";
                btn.innerHTML = `
                    <span class="sample-icon">📊</span>
                    <div class="sample-info">
                        <span class="sample-name">${sample.display_name}</span>
                        <span class="sample-type">Earnings Release · ${sample.size_mb}MB</span>
                    </div>
                `;
                btn.onclick = () => loadSample(sample.filename);
                row.appendChild(btn);
            });
        }
    } catch (err) {
        console.log("Could not load samples:", err);
    }
}

// ===== DRAG AND DROP =====
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
    const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.pdf'));
    if (files.length > 0) uploadFiles(files);
    else alert("Please drop PDF files only.");
}

function handleFileSelect(e) {
    console.log("Files selected:", e.target.files.length);
    const files = Array.from(e.target.files);
    if (files.length > 0) uploadFiles(files);
}

function resetFileInput() {
    document.getElementById("file-input").value = "";
}

// ===== UPLOAD =====

async function uploadFiles(files) {
    console.log("Uploading", files.length, "files");
    resetFileInput();

    // Validate all are PDFs
    const nonPdfs = files.filter(f => !f.name.endsWith('.pdf'));
    if (nonPdfs.length > 0) {
        alert(`These are not PDFs: ${nonPdfs.map(f => f.name).join(', ')}`);
        return;
    }

    const label = files.length === 1
        ? `Uploading ${files[0].name}...`
        : `Uploading ${files.length} documents...`;

    showProcessing(label);
    setStep(1);

    const formData = new FormData();
    files.forEach(file => formData.append("files", file));

    try {
        const res = await fetch(`${API_URL}/upload`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Upload failed");
        }

        const data = await res.json();
        console.log("Upload response:", data);
        setStep(2);

        await pollUntilReady(data.session_id, data.pdf_names.join(", "));

    } catch (err) {
        console.error("Upload error:", err);
        hideProcessing();
        alert(`Error: ${err.message}`);
    }
}

// ===== LOAD SAMPLE =====
async function loadSample(filename) {
    console.log("Loading sample:", filename);
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

// ===== POLL UNTIL READY =====
async function pollUntilReady(sessionId, pdfName) {
    console.log("Polling session:", sessionId);

    const maxAttempts = 100;
    let attempts = 0;

    const messages = [
        "Reading PDF pages...",
        "Extracting text and tables...",
        "Converting tables to natural language...",
        "Generating semantic embeddings...",
        "Building FAISS search index...",
        "Almost ready..."
    ];

    while (attempts < maxAttempts) {
        await sleep(3000);
        attempts++;

        const msgIndex = Math.min(Math.floor(attempts / 3), messages.length - 1);
        document.getElementById("processing-title").textContent = messages[msgIndex];

        if (attempts > 2) setStep(3);
        if (attempts > 5) setStep(4);

        try {
            const res = await fetch(`${API_URL}/status/${sessionId}`);
            const status = await res.json();
            console.log(`Poll ${attempts}:`, status.status);

            if (status.status === "ready") {
                await sleep(500);
                hideProcessing();
                startChat(sessionId, status.pdf_name, status.chunks_created, status.pdf_count || 1);
                return;
            }

            if (status.status === "error") {
                throw new Error(status.message || "Processing failed");
            }

        } catch (err) {
            if (err.message !== "Processing failed") {
                console.log("Poll retry:", err.message);
                continue;
            }
            throw err;
        }
    }

    hideProcessing();
    alert("Processing timed out. Please try a smaller PDF.");
}

// ===== START CHAT =====
function startChat(sessionId, pdfName, chunksCreated, pdfCount) {
    console.log("Starting chat:", sessionId);
    currentSessionId = sessionId;

    // Show multiple docs info if applicable
    const displayName = pdfCount > 1
        ? `${pdfCount} documents loaded`
        : pdfName;

    document.getElementById("doc-name").textContent = displayName;
    document.getElementById("doc-chunks").textContent =
        `${chunksCreated} chunks indexed`;

    resetChatToWelcome(pdfCount);

    document.getElementById("upload-screen").classList.remove("active");
    document.getElementById("upload-screen").classList.add("hidden");
    document.getElementById("chat-screen").classList.remove("hidden");
    document.getElementById("chat-screen").classList.add("active");

    document.getElementById("question-input").focus();
}

// ===== RESET CHAT TO WELCOME =====
function resetChatToWelcome(pdfCount = 1) {
    const multiDocChips = pdfCount > 1
        ? `<button class="chip" onclick="askSuggestion(this)">Compare revenue across documents</button>
           <button class="chip" onclick="askSuggestion(this)">What are common risk factors?</button>`
        : `<button class="chip" onclick="askSuggestion(this)">What was the revenue?</button>
           <button class="chip" onclick="askSuggestion(this)">What is the operating margin?</button>`;

    document.getElementById("chat-messages").innerHTML = `
        <div class="chat-welcome" id="chat-welcome">
            <div class="welcome-icon">${pdfCount > 1 ? '📚' : '💬'}</div>
            <h3>${pdfCount > 1 ? `${pdfCount} documents ready` : 'Document ready'}</h3>
            <p>${pdfCount > 1
                ? 'Ask questions across all documents. I can compare and contrast information between them.'
                : 'Ask anything about the document. I\'ll answer with exact source citations.'
            }</p>
            <div class="suggestion-chips" id="suggestion-chips">
                ${multiDocChips}
                <button class="chip" onclick="askSuggestion(this)">Summarize key highlights</button>
                <button class="chip" onclick="askSuggestion(this)">What was the YoY growth?</button>
            </div>
        </div>
    `;
}

// ===== SEND QUESTION =====
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

    // Remove welcome state on first question
    const welcome = document.getElementById("chat-welcome");
    if (welcome) welcome.remove();

    input.value = "";
    autoResize(input);

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

// ===== ADD MESSAGES =====
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
        toggleBtn.innerHTML = `📎 ${sources.length} sources · click to view`;

        const sourcesList = document.createElement("div");
        sourcesList.className = "sources-list";

        sources.forEach(source => {
            const item = document.createElement("div");
            item.className = "source-item";
            item.innerHTML = `
                <div class="source-header">
                    <span class="source-label">Source ${source.rank} · Page ${source.page || "N/A"}</span>
                    <span class="source-score">${(source.similarity * 100).toFixed(0)}% match</span>
                </div>
                <div class="source-preview">${source.preview}</div>
            `;
            sourcesList.appendChild(item);
        });

        toggleBtn.onclick = () => {
            sourcesList.classList.toggle("visible");
            toggleBtn.innerHTML = sourcesList.classList.contains("visible")
                ? `📎 ${sources.length} sources · click to hide`
                : `📎 ${sources.length} sources · click to view`;
        };

        msg.appendChild(toggleBtn);
        msg.appendChild(sourcesList);
    }

    container.appendChild(msg);
    scrollToBottom();
}

// ===== THINKING =====
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

// ===== SUGGESTIONS =====
function askSuggestion(btn) {
    document.getElementById("question-input").value = btn.textContent.trim();
    sendQuestion();
}

// ===== KEYBOARD =====
function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
}

// ===== TEXTAREA RESIZE =====
function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

// ===== PROCESSING UI =====
function showProcessing(title) {
    document.getElementById("processing-state").classList.remove("hidden");
    document.getElementById("processing-title").textContent = title;
    resetSteps();
}

function hideProcessing() {
    document.getElementById("processing-state").classList.add("hidden");
    resetSteps();
}

function setStep(n) {
    for (let i = 1; i <= 4; i++) {
        const step = document.getElementById(`step-${i}`);
        if (i < n) step.className = "proc-step done";
        else if (i === n) step.className = "proc-step active";
        else step.className = "proc-step";
    }
}

function resetSteps() {
    for (let i = 1; i <= 4; i++) {
        document.getElementById(`step-${i}`).className = "proc-step";
    }
}

// ===== NAVIGATION — BUG FIXES HERE =====
function newConversation() {
    if (!currentSessionId) return;

    // Reset memory on server
    fetch(`${API_URL}/reset/${currentSessionId}`, { method: "POST" });

    // Reset chat to welcome — NOT "conversation reset" message
    resetChatToWelcome();
}

function uploadNew() {
    // Close current session on server
    if (currentSessionId) {
        fetch(`${API_URL}/session/${currentSessionId}`, { method: "DELETE" });
        currentSessionId = null;
    }

    // ===== FIX: Clear header info =====
    document.getElementById("doc-name").textContent = "";
    document.getElementById("doc-chunks").textContent = "";

    // Switch back to upload screen
    document.getElementById("chat-screen").classList.add("hidden");
    document.getElementById("chat-screen").classList.remove("active");
    document.getElementById("upload-screen").classList.remove("hidden");
    document.getElementById("upload-screen").classList.add("active");

    hideProcessing();
    resetFileInput();
}

// ===== UTILS =====
function scrollToBottom() {
    const container = document.getElementById("chat-messages");
    container.scrollTop = container.scrollHeight;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}