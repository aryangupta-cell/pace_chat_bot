// Shared chat logic used by both the standalone chat page (index.html) and
// the floating chat bubble on the dashboard-embed page (dashboard.html).
// Call initChatWidget({chatEl, inputEl, sendBtnEl}) once the DOM is ready.
function initChatWidget({ chatEl, inputEl, sendBtnEl }) {
  if (!document.getElementById('pace-table-style')) {
    const style = document.createElement('style');
    style.id = 'pace-table-style';
    style.textContent = `
      .bubble .pace-table-wrap {
        max-width: 100%;
        overflow-x: auto;
        margin: 4px 0;
      }
      .bubble table.pace-table {
        border-collapse: collapse;
        width: 100%;
        table-layout: auto;
        font-size: 12.5px;
        margin: 0;
      }
      .bubble table.pace-table th, .bubble table.pace-table td {
        border: 1px solid #d7dce3;
        padding: 4px 6px;
        text-align: left;
        vertical-align: top;
      }
      .bubble table.pace-table th {
        background: #0a58ca;
        color: #fff;
        font-weight: 600;
        white-space: normal;
        min-width: 60px;
      }
      .bubble table.pace-table td {
        white-space: nowrap;
      }
      .bubble table.pace-table th:first-child,
      .bubble table.pace-table td:first-child {
        min-width: 24px;
      }
      .bubble table.pace-table tbody tr:nth-child(even) { background: #f2f5fa; }
      .pace-typing-dots {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 0;
      }
      .pace-typing-dots span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #9aa4b2;
        opacity: 0.4;
        animation: pace-typing-blink 1.2s infinite ease-in-out;
      }
      .pace-typing-dots span:nth-child(2) { animation-delay: 0.2s; }
      .pace-typing-dots span:nth-child(3) { animation-delay: 0.4s; }
      @keyframes pace-typing-blink {
        0%, 80%, 100% { opacity: 0.4; transform: scale(0.85); }
        40% { opacity: 1; transform: scale(1); }
      }
    `;
    document.head.appendChild(style);
  }

  const sessionId = (window.crypto && window.crypto.randomUUID)
    ? window.crypto.randomUUID()
    : 'sess-' + Math.random().toString(36).slice(2) + Date.now();

  function addMessage(text, who) {
    const div = document.createElement('div');
    div.className = 'msg ' + who;
    const bubble = document.createElement('span');
    bubble.className = 'bubble';
    // Server-formatted replies embed a pre-escaped <table class="pace-table">
    // for multi-row results (see app/main.py _render_table/_esc); everything
    // else is plain text. Only switch to innerHTML when a table is actually
    // present, and only for bot messages — user input always stays as
    // textContent so it can never be interpreted as HTML.
    if (who === 'bot' && typeof text === 'string' && text.indexOf('<table') !== -1) {
      // Wrap the server-emitted <table> in a scrollable container so the
      // table itself stays a real single <table> (header/body columns laid
      // out together, so they align) instead of being split into two
      // independently-sized tables the way a bare `display:block` table
      // with `thead/tbody{display:table}` would force.
      bubble.innerHTML = text.replace(
        /<table class="pace-table">[\s\S]*?<\/table>/g,
        (tableHtml) => `<div class="pace-table-wrap">${tableHtml}</div>`
      );
    } else {
      bubble.textContent = text;
    }
    div.appendChild(bubble);
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function showTyping() {
    const div = document.createElement('div');
    div.className = 'msg bot pace-typing-msg';
    const bubble = document.createElement('span');
    bubble.className = 'bubble';
    bubble.innerHTML = '<span class="pace-typing-dots"><span></span><span></span><span></span></span>';
    div.appendChild(bubble);
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
    return div;
  }

  async function send() {
    const text = inputEl.value.trim();
    if (!text) return;
    addMessage(text, 'user');
    inputEl.value = '';
    const typingEl = showTyping();
    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionId })
      });
      const data = await resp.json();
      typingEl.remove();
      addMessage(data.reply, 'bot');
    } catch (err) {
      typingEl.remove();
      addMessage('Error contacting server: ' + err, 'bot');
    }
  }

  sendBtnEl.addEventListener('click', send);
  inputEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });

  addMessage("Hi! Ask me about attendance or productive time (optionally mention a department and/or month).", 'bot');

  return { addMessage, send };
}
