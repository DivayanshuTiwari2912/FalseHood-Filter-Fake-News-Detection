import React, { useState } from 'react';
import { FiUsers, FiMessageSquare, FiSend, FiClock, FiUser } from 'react-icons/fi';

// Collaboration panel for team fact checking
const CollaborationPanel = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [activationMessage, setActivationMessage] = useState(null);
  const [messages, setMessages] = useState([
    {
      id: 1,
      user: 'System',
      message: 'Welcome to the collaboration panel! Share your analysis and discuss with team members.',
      timestamp: new Date().toISOString()
    }
  ]);
  const [newMessage, setNewMessage] = useState('');
  const [username, setUsername] = useState('Guest User');
  const [showNameInput, setShowNameInput] = useState(false);
  
  // Format timestamp
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  // Handle sending a new message
  const handleSendMessage = (e) => {
    e.preventDefault();
    
    if (!newMessage.trim()) return;
    
    const message = {
      id: Date.now(),
      user: username,
      message: newMessage,
      timestamp: new Date().toISOString()
    };
    
    setMessages([...messages, message]);
    setNewMessage('');
    
    // In a real implementation, this would send the message to other users
    // via WebSockets or another real-time communication method
  };
  
  // Handle opening the panel for the first time
  const handleOpenPanel = () => {
    setIsOpen(true);
    
    // Show a welcome/activation message once
    if (!activationMessage) {
      setTimeout(() => {
        setActivationMessage({
          title: 'Team Collaboration Activated',
          message: 'Share insights and analyses with your team in real-time. This feature would connect to other team members in a real implementation.',
        });
      }, 500);
    }
  };
  
  // Handle username change
  const handleUsernameChange = (newName) => {
    if (newName.trim()) {
      setUsername(newName);
      setShowNameInput(false);
      
      // Add a system message about the name change
      const message = {
        id: Date.now(),
        user: 'System',
        message: `User changed name to "${newName}"`,
        timestamp: new Date().toISOString()
      };
      
      setMessages([...messages, message]);
    }
  };
  
  return (
    <div className="collaboration-panel-container">
      {/* Floating button to toggle panel */}
      <button 
        className={`collaboration-toggle-btn ${isOpen ? 'active' : ''}`}
        onClick={isOpen ? () => setIsOpen(false) : handleOpenPanel}
        title={isOpen ? "Close collaboration panel" : "Open collaboration panel"}
      >
        <FiUsers size={20} />
        <span className="collaboration-badge">
          {messages.length > 1 ? messages.length - 1 : ''}
        </span>
      </button>
      
      {/* Main panel */}
      {isOpen && (
        <div className="collaboration-panel">
          <div className="collaboration-header">
            <h5>
              <FiUsers className="me-2" />
              Team Collaboration
            </h5>
            <button 
              className="collaboration-close-btn"
              onClick={() => setIsOpen(false)}
            >
              &times;
            </button>
          </div>
          
          {/* Welcome message for first-time activation */}
          {activationMessage && (
            <div className="collaboration-activation-message">
              <h6>{activationMessage.title}</h6>
              <p>{activationMessage.message}</p>
              <button 
                className="btn btn-sm btn-outline-secondary"
                onClick={() => setActivationMessage(null)}
              >
                Got it
              </button>
            </div>
          )}
          
          {/* Username display/editor */}
          <div className="collaboration-user">
            <div className="d-flex align-items-center justify-content-between p-2 border-bottom">
              <div className="d-flex align-items-center">
                <FiUser className="me-2 text-primary" />
                {showNameInput ? (
                  <form onSubmit={(e) => {
                    e.preventDefault();
                    handleUsernameChange(e.target.username.value);
                  }}>
                    <div className="input-group input-group-sm">
                      <input 
                        type="text" 
                        className="form-control form-control-sm" 
                        name="username"
                        defaultValue={username}
                        autoFocus
                      />
                      <button type="submit" className="btn btn-sm btn-primary">Save</button>
                    </div>
                  </form>
                ) : (
                  <>
                    <span>{username}</span>
                    <button 
                      className="btn btn-sm text-primary ms-2"
                      onClick={() => setShowNameInput(true)}
                      style={{ padding: '0', background: 'none', border: 'none' }}
                    >
                      (edit)
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
          
          {/* Messages area */}
          <div className="collaboration-messages">
            {messages.map((msg) => (
              <div 
                key={msg.id} 
                className={`message ${msg.user === 'System' ? 'system-message' : 
                                     msg.user === username ? 'user-message' : 'other-message'}`}
              >
                <div className="message-header">
                  <strong>{msg.user}</strong>
                  <small className="message-time">
                    <FiClock className="me-1" size={12} />
                    {formatTime(msg.timestamp)}
                  </small>
                </div>
                <div className="message-body">
                  {msg.message}
                </div>
              </div>
            ))}
          </div>
          
          {/* Message input */}
          <form className="collaboration-input" onSubmit={handleSendMessage}>
            <div className="input-group">
              <input
                type="text"
                className="form-control"
                placeholder="Type your message..."
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
              />
              <button 
                type="submit" 
                className="btn btn-primary"
                disabled={!newMessage.trim()}
              >
                <FiSend />
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};

export default CollaborationPanel;