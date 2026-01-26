// API Configuration
const API_BASE = window.location.origin;

// State
let currentSessionId = null;
let isStreaming = false;
let documents = [];

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    checkHealth();
    loadDocuments();
});

// Initialize
function initializeApp() {
    console.log('Chat With Your Docs initialized');
}

// Event Listeners
function setupEventListeners() {
    // File upload
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // Message input
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');

    messageInput.addEventListener('input', () => {
        autoResize(messageInput);
        sendBtn.disabled = !messageInput.value.trim();
    });

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    // New chat
    document.getElementById('newChatBtn').addEventListener('click', newChat);

    // Example queries
    document.querySelectorAll('.example-query').forEach(btn => {
        btn.addEventListener('click', () => {
            messageInput.value = btn.textContent.replace(/"/g, '');
            sendBtn.disabled = false;
            messageInput.focus();
        });
    });
}

// Auto-resize textarea
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

// Check API health
async function checkHealth() {
    const indicator = document.getElementById('statusIndicator');
    const statusText = indicator.querySelector('.status-text');
    const statusDot = indicator.querySelector('.status-dot');

    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            statusText.textContent = 'Connected';
            statusDot.style.background = 'var(--success)';
        }
    } catch (error) {
        statusText.textContent = 'Disconnected';
        statusDot.style.background = 'var(--error)';
        statusDot.style.animation = 'none';
    }
}

// Load documents
async function loadDocuments() {
    const documentList = document.getElementById('documentList');

    try {
        const response = await fetch(`${API_BASE}/api/documents`);
        documents = await response.json();

        if (documents.length === 0) {
            documentList.innerHTML = '<div class="loading">No documents yet</div>';
            return;
        }

        documentList.innerHTML = documents.map(doc => `
            <div class="document-item">
                <div class="document-info">
                    <div class="document-name">${doc.file_name}</div>
                    <div class="document-meta">${doc.total_chunks} chunks</div>
                </div>
                <button class="document-delete" onclick="deleteDocument('${doc.doc_id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                </button>
            </div>
        `).join('');

        // Enable send button if documents exist
        document.getElementById('sendBtn').disabled = false;

    } catch (error) {
        console.error('Failed to load documents:', error);
        documentList.innerHTML = '<div class="loading">Failed to load</div>';
    }
}

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
}

// Upload file
async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    const documentList = document.getElementById('documentList');
    documentList.innerHTML = '<div class="loading">Uploading...</div>';

    try {
        const response = await fetch(`${API_BASE}/api/ingest`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        // Poll for completion
        await pollIngestionStatus(result.job_id);

        // Reload documents
        await loadDocuments();

        addSystemMessage(`✅ Document "${file.name}" processed successfully!`);

    } catch (error) {
        console.error('Upload failed:', error);
        addSystemMessage(`❌ Failed to upload document: ${error.message}`);
        loadDocuments();
    }
}

// Poll ingestion status
async function pollIngestionStatus(jobId) {
    const pipelineStatus = document.getElementById('pipelineStatus');

    while (true) {
        const response = await fetch(`${API_BASE}/api/ingest/status/${jobId}`);
        const status = await response.json();

        pipelineStatus.innerHTML = `
            <div class="pipeline-stage">
                <div class="stage-spinner"></div>
                <span>${status.message}</span>
            </div>
        `;

        if (status.status === 'completed') {
            pipelineStatus.innerHTML = '';
            break;
        }

        if (status.status === 'failed') {
            throw new Error(status.error || 'Processing failed');
        }

        await new Promise(resolve => setTimeout(resolve, 2000));
    }
}

// Delete document
async function deleteDocument(docId) {
    if (!confirm('Delete this document?')) return;

    try {
        await fetch(`${API_BASE}/api/documents/${docId}`, {
            method: 'DELETE'
        });

        await loadDocuments();
        addSystemMessage('Document deleted');

    } catch (error) {
        console.error('Delete failed:', error);
        addSystemMessage('Failed to delete document');
    }
}

// Send message with STREAMING
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const question = input.value.trim();

    if (!question || isStreaming) return;

    // Create session if it doesn't exist (for conversation context)
    if (!currentSessionId) {
        currentSessionId = 'session_' + Date.now();
        console.log('Created new session:', currentSessionId);
    }

    // Add user message
    addMessage(question, 'user');

    // Clear input
    input.value = '';
    input.style.height = 'auto';
    document.getElementById('sendBtn').disabled = true;
    isStreaming = true;

    // Show pipeline status
    const pipelineStatus = document.getElementById('pipelineStatus');
    pipelineStatus.innerHTML = `
        <div class="pipeline-stage">
            <div class="stage-spinner"></div>
            <span>Processing query...</span>
        </div>
    `;

    // Create assistant message container for streaming
    const messages = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <span class="streaming-cursor">▋</span>
        </div>
    `;
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;

    const contentDiv = messageDiv.querySelector('.message-content');
    let fullAnswer = '';
    let sources = [];

    try {
        // Get top K value from settings
        const topKInput = document.getElementById('topK');
        const topK = topKInput ? parseInt(topKInput.value) : 5;

        // Use EventSource for streaming with session ID and top K
        const sessionParam = currentSessionId ? `&session_id=${currentSessionId}` : '';
        const topKParam = `&top_k=${topK}`;
        const eventSource = new EventSource(
            `${API_BASE}/api/stream/query?question=${encodeURIComponent(question)}${sessionParam}${topKParam}`
        );

        // Handle sources event
        eventSource.addEventListener('sources', (event) => {
            const data = JSON.parse(event.data);
            sources = data.sources || [];
            console.log('Received sources:', sources);
        });

        eventSource.addEventListener('answer', (event) => {
            const data = JSON.parse(event.data);
            fullAnswer += data.chunk;
            // Render markdown
            const renderedContent = marked.parse(fullAnswer);
            contentDiv.innerHTML = renderedContent + '<span class="streaming-cursor">▋</span>';
            messages.scrollTop = messages.scrollHeight;
        });

        eventSource.addEventListener('stage', (event) => {
            const data = JSON.parse(event.data);
            pipelineStatus.innerHTML = `
                <div class="pipeline-stage">
                    <div class="stage-spinner"></div>
                    <span>${data.stage || 'Processing'}...</span>
                </div>
            `;
        });

        eventSource.addEventListener('complete', () => {
            const renderedContent = marked.parse(fullAnswer);
            let finalContent = renderedContent;

            // Add sources/citations if available
            if (sources && sources.length > 0) {
                const citationsHtml = `
                    <div class="message-sources">
                        <h4>📚 Sources</h4>
                        ${sources.map(s => `
                            <div class="source-item">
                                <strong>${s.file_name}</strong> (Page ${s.page}, Score: ${(s.score || 0).toFixed(3)})
                            </div>
                        `).join('')}
                    </div>
                `;
                finalContent += citationsHtml;
            }

            contentDiv.innerHTML = finalContent;
            eventSource.close();
            pipelineStatus.innerHTML = '';
            document.getElementById('sendBtn').disabled = false;
            isStreaming = false;
        });

        eventSource.addEventListener('error', (event) => {
            console.error('Streaming error:', event);
            if (fullAnswer) {
                const renderedContent = marked.parse(fullAnswer);
                contentDiv.innerHTML = renderedContent;
            } else {
                contentDiv.innerHTML = '❌ Failed to get response';
            }
            eventSource.close();
            pipelineStatus.innerHTML = '';
            document.getElementById('sendBtn').disabled = false;
            isStreaming = false;
        });

    } catch (error) {
        console.error('Query failed:', error);
        contentDiv.innerHTML = '❌ Failed to get response';
        pipelineStatus.innerHTML = '';
        document.getElementById('sendBtn').disabled = false;
        isStreaming = false;
    }
}

// Add message to chat
function addMessage(content, role, sources = null, confidence = null) {
    const messages = document.getElementById('messages');

    // Remove welcome message
    const welcome = messages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role}`;

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="message-sources">
                <h4>📚 Sources (${sources.length})</h4>
                ${sources.map(s => `
                    <div class="source-item">
                        ${s.citation} - Score: ${s.score.toFixed(3)}
                    </div>
                `).join('')}
                ${confidence ? `<span class="confidence-badge">Confidence: ${(confidence * 100).toFixed(1)}%</span>` : ''}
            </div>
        `;
    }

    // Render markdown for assistant messages, plain text for user
    let renderedContent = content;
    if (role === 'assistant') {
        renderedContent = marked.parse(content);
    } else {
        renderedContent = content.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    messageDiv.innerHTML = `
        <div class="message-content">
            ${renderedContent}
            ${sourcesHtml}
        </div>
    `;

    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

// Add system message
function addSystemMessage(content) {
    const messages = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-assistant';
    messageDiv.innerHTML = `
        <div class="message-content" style="background: var(--bg-primary); font-size: 0.875rem;">
            ${content}
        </div>
    `;
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

// New chat
function newChat() {
    currentSessionId = null;
    const messages = document.getElementById('messages');
    messages.innerHTML = `
        <div class="welcome-message">
            <h3>👋 New Conversation</h3>
            <p>Ask questions about your documents.</p>
        </div>
    `;
}
