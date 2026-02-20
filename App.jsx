import { useState, useEffect, useRef } from 'react'
import './index.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ============================
// Login / Signup Page
// ============================
function AuthPage({ onLogin }) {
  const [isSignup, setIsSignup] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const endpoint = isSignup ? '/signup' : '/login'
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Authentication failed')

      localStorage.setItem('token', data.token)
      localStorage.setItem('username', data.username)
      onLogin(data.username, data.token)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-content">
        <div className="auth-card">
          <div className="auth-header">
            <h1 className="auth-title">
              <span className="auth-logo-icon">📖</span> Urdu Story AI
            </h1>
            <p className="auth-subtitle">Professional Story Generation Platform</p>
          </div>

          {error && <div className="error-msg">{error}</div>}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label">Username</label>
              <input
                className="form-input"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                minLength={3}
                placeholder="Enter your username"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Password</label>
              <input
                className="form-input"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={4}
                placeholder="••••••••"
              />
            </div>

            <button className="btn-primary" type="submit" disabled={loading}>
              {loading ? 'Processing...' : (isSignup ? 'Create Account' : 'Sign In')}
            </button>
          </form>

          <div className="auth-switch">
            {isSignup ? 'Already have an account?' : "Don't have an account?"}
            <button className="auth-switch-btn" onClick={() => setIsSignup(!isSignup)}>
              {isSignup ? 'Sign In' : 'Sign Up'}
            </button>
          </div>
        </div>
      </div>
      <div className="auth-sidebar">
        {/* Background image handled by CSS */}
      </div>
    </div>
  )
}

// ============================
// Chat Sidebar
// ============================
function ChatSidebar({ chats, activeChat, onSelectChat, onNewChat, onDeleteChat, username, onLogout }) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="app-brand">
          <span>📖</span> Urdu Story AI
        </div>
        <button className="new-chat-btn" onClick={onNewChat}>
          <span>+</span> New Story
        </button>
      </div>

      <div className="chat-list">
        {chats.map(chat => (
          <div
            key={chat.id}
            className={`chat-list-item ${activeChat === chat.id ? 'active' : ''}`}
            onClick={() => onSelectChat(chat)}
          >
            <span className="chat-title">{chat.title || 'Untitled Story'}</span>
            <button
              className="delete-chat-btn"
              onClick={(e) => { e.stopPropagation(); onDeleteChat(chat.id) }}
              title="Delete"
            >
              ×
            </button>
          </div>
        ))}
      </div>

      <div className="user-profile">
        <div className="user-details">
          <div className="avatar-placeholder">{username[0]}</div>
          <span>{username}</span>
        </div>
        <button className="logout-icon-btn" onClick={onLogout} title="Sign Out">
          ➔
        </button>
      </div>
    </div>
  )
}

// ============================
// Main App
// ============================
function MainApp({ username, token, onLogout }) {
  const [chats, setChats] = useState([])
  const [activeChat, setActiveChat] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [maxWords, setMaxWords] = useState(150)
  const [temperature, setTemperature] = useState(0.7)
  const [nGram, setNGram] = useState(3)
  const [vocabSize, setVocabSize] = useState(250)
  const [toast, setToast] = useState('')
  const chatEndRef = useRef(null)

  useEffect(() => {
    loadChats()
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const loadChats = async () => {
    try {
      const res = await fetch(`${API_URL}/chats`, {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (res.ok) {
        const data = await res.json()
        setChats(data)
      }
    } catch (err) {
      console.error('Failed to load chats:', err)
    }
  }

  const handleNewChat = () => {
    setActiveChat(null)
    setMessages([])
    setInput('')
  }

  const handleSelectChat = (chat) => {
    setActiveChat(chat.id)
    try {
      const requestMessages = JSON.parse(chat.messages)
      setMessages(requestMessages)
    } catch (e) {
      console.error('Failed to parse chat messages', e)
      setMessages([])
    }
  }

  const handleDeleteChat = async (chatId) => {
    try {
      await fetch(`${API_URL}/chats/${chatId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      })
      const newChats = chats.filter(c => c.id !== chatId)
      setChats(newChats)
      if (activeChat === chatId) {
        handleNewChat()
      }
      showToast('Chat deleted')
    } catch (err) {
      console.error('Failed to delete chat:', err)
    }
  }

  const showToast = (msg) => {
    setToast(msg)
    setTimeout(() => setToast(''), 3000)
  }

  const handleSend = async () => {
    if (!input.trim() || isGenerating) return

    const userText = input.trim()
    setInput('')

    const newMessages = [...messages, { role: 'user', text: userText }]
    setMessages(newMessages)

    setIsGenerating(true)

    const aiMessageIndex = newMessages.length
    const messagesWithAi = [...newMessages, { role: 'ai', text: '', loading: true }]
    setMessages(messagesWithAi)

    let fullGeneratedText = ''

    try {
      const res = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prefix: userText,
          max_length: maxWords,
          temperature: temperature,
          repetition_penalty: 1.3,
          n_gram: nGram,
          vocab_size: vocabSize
        })
      })

      if (!res.ok) throw new Error('Generation failed')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.full_text) {
                fullGeneratedText = data.full_text
                setMessages(prev => {
                  const updated = [...prev]
                  if (updated[aiMessageIndex]) {
                    updated[aiMessageIndex] = { role: 'ai', text: data.full_text, loading: false }
                  }
                  return updated
                })
              }
            } catch (e) {
              // ignore malformed
            }
          }
        }
      }

      const finalMessages = [...newMessages, { role: 'ai', text: fullGeneratedText, loading: false }]

      const title = activeChat
        ? chats.find(c => c.id === activeChat)?.title
        : userText.slice(0, 50) || 'New Story'

      const saveRes = await fetch(`${API_URL}/chats`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          id: activeChat,
          title: title,
          messages: JSON.stringify(finalMessages)
        })
      })

      if (saveRes.ok) {
        const savedData = await saveRes.json()
        if (!activeChat) {
          setActiveChat(savedData.id)
          setChats(prev => [{
            id: savedData.id,
            title,
            messages: JSON.stringify(finalMessages),
            updated_at: new Date().toISOString()
          }, ...prev])
        } else {
          setChats(prev => {
            const others = prev.filter(c => c.id !== activeChat)
            const current = prev.find(c => c.id === activeChat)
            return [{ ...current, messages: JSON.stringify(finalMessages), updated_at: new Date().toISOString() }, ...others]
          })
        }
      }

    } catch (err) {
      console.error(err)
      setMessages(prev => {
        const updated = [...prev]
        if (updated[aiMessageIndex]) {
          updated[aiMessageIndex] = { role: 'ai', text: 'Error generating story.', loading: false, error: true }
        }
        return updated
      })
    } finally {
      setIsGenerating(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="app-container">
      <ChatSidebar
        chats={chats}
        activeChat={activeChat}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        username={username}
        onLogout={onLogout}
      />

      <div className="main-content">
        <div className="top-bar">
          <div className="model-selectors" style={{ display: 'flex', gap: '16px' }}>
            <div className="model-selector">
              <span>Model:</span>
              <select
                value={nGram}
                onChange={(e) => setNGram(parseInt(e.target.value))}
                style={{ background: 'transparent', color: 'inherit', border: 'none', marginLeft: '4px', cursor: 'pointer', outline: 'none' }}
              >
                <option value={3}>Trigram</option>
                <option value={5}>5-Gram</option>
                <option value={7}>7-Gram</option>
              </select>
            </div>
            <div className="model-selector">
              <span>Vocab:</span>
              <select
                value={vocabSize}
                onChange={(e) => setVocabSize(parseInt(e.target.value))}
                style={{ background: 'transparent', color: 'inherit', border: 'none', marginLeft: '4px', cursor: 'pointer', outline: 'none' }}
              >
                <option value={250}>250</option>
                <option value={500}>500</option>
                <option value={1000}>1,000</option>
                <option value={5000}>5,000</option>
              </select>
            </div>
          </div>
          <div className="temp-control-wrapper">
            <span className="temp-label">Creativity: {temperature}</span>
            <input
              type="range"
              min="0.1"
              max="1.5"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="temp-slider-input"
            />
          </div>
        </div>

        {/* ── CHILDREN'S CHAT SCREEN ── */}
        <div className="chat-screen">
          <div className="chat-area">
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">✨</div>
                <h2 className="empty-title">اردو کہانی شروع کریں</h2>
                <p>نیچے لکھیں اور کہانی شروع کریں!</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div key={idx} className="message-row">
                  <div className={`message-avatar ${msg.role === 'user' ? 'user-avatar-msg' : 'ai-avatar-msg'}`}>
                    {msg.role === 'user' ? '👤' : '🤖'}
                  </div>
                  <div className={`message-content ${msg.role === 'ai' ? 'urdu-text' : ''}`}>
                    <span className="message-role">{msg.role === 'user' ? 'You' : 'Story AI'}</span>
                    {msg.text}
                    {msg.loading && <span style={{ display: 'inline-block', marginLeft: '4px' }}>...</span>}
                  </div>
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="input-container">
            <div className="input-actions" style={{ marginBottom: '8px' }}>
              <div className="word-count-chip">
                <span>Length:</span>
                <input
                  type="number"
                  value={maxWords}
                  onChange={(e) => setMaxWords(Number(e.target.value))}
                  className="wc-input"
                  min="10" max="500"
                />
                <span>words</span>
              </div>
            </div>
            <div className="input-box-wrapper">
              <textarea
                className="chat-input"
                placeholder="Start your story here..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={isGenerating}
              />
              <div className="input-actions" style={{ justifyContent: 'flex-end', marginTop: 0 }}>
                <button
                  className="send-btn"
                  onClick={handleSend}
                  disabled={!input.trim() || isGenerating}
                >
                  ➔
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {toast && <div className="toast-notif">{toast}</div>}
    </div>
  )
}

// ============================
// Root App
// ============================
function App() {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const savedToken = localStorage.getItem('token')
    const savedUser = localStorage.getItem('username')
    if (savedToken && savedUser) {
      setToken(savedToken)
      setUser(savedUser)
    }
    setLoading(false)
  }, [])

  const handleLogin = (username, token) => {
    setUser(username)
    setToken(token)
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('username')
    setUser(null)
    setToken(null)
  }

  if (loading) return null

  if (!user || !token) {
    return <AuthPage onLogin={handleLogin} />
  }

  return <MainApp username={user} token={token} onLogout={handleLogout} />
}

export default App
