import React, { useState, useEffect, useRef } from 'react';
import { Brain, Cpu, Wifi, WifiOff, Eye, Server, Radio, Sparkles } from 'lucide-react';
import VoiceOrb from './components/VoiceOrb';
import SystemTelemetry from './components/SystemTelemetry';
import ScreenPerception from './components/ScreenPerception';
import ConsoleLogs from './components/ConsoleLogs';
import UserKnowledge from './components/UserKnowledge';

const WS_URL = 'ws://localhost:8000/ws';

export default function App() {
  const [activeTab, setActiveTab] = useState('command');
  const [isConnected, setIsConnected] = useState(false);
  const [telemetry, setTelemetry] = useState(null);
  const [specs, setSpecs] = useState(null);
  const [memory, setMemory] = useState({});
  const [screenFrame, setScreenFrame] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState([]);
  const [selectedModel, setSelectedModel] = useState('qwen2.5-coder:3b');
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    };
  }, []);

  const speakInBrowser = (text) => {
    if (window.speechSynthesis && text) {
      try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;

        utterance.onstart = () => setIsSpeaking(true);
        utterance.onend = () => setIsSpeaking(false);
        utterance.onerror = () => setIsSpeaking(false);

        window.speechSynthesis.speak(utterance);
      } catch (e) {
        console.error("Browser TTS Error:", e);
      }
    }
  };

  const connectWebSocket = () => {
    try {
      if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        return;
      }

      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        setIsConnected(true);
        console.log("Connected to JARVIS Core WebSocket");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'init') {
            setSpecs(data.specs);
            setMemory(data.memory || {});
          } else if (data.type === 'telemetry_tick') {
            setTelemetry(data.telemetry);
            if (data.screen_frame) setScreenFrame(data.screen_frame);
            if (!window.speechSynthesis || !window.speechSynthesis.speaking) {
              setIsSpeaking(data.is_speaking || false);
            }
          } else if (data.type === 'status') {
            setIsProcessing(true);
            setLogs((prev) => [...prev, { type: 'status', message: data.message }]);
          } else if (data.type === 'thought') {
            setLogs((prev) => [...prev, { type: 'thought', text: data.text }]);
          } else if (data.type === 'tool_exec') {
            setLogs((prev) => [...prev, { type: 'tool_exec', tool: data.tool, input: data.input, output: data.output }]);
          } else if (data.type === 'response') {
            setIsProcessing(false);
            setLogs((prev) => [...prev, { type: 'response', text: data.text }]);
            // Trigger browser voice synthesis out loud!
            speakInBrowser(data.text);
          } else if (data.type === 'error') {
            setIsProcessing(false);
            setLogs((prev) => [...prev, { type: 'error', message: data.message }]);
          }
        } catch (err) {
          console.error("WS Parse error:", err);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        reconnectTimerRef.current = setTimeout(connectWebSocket, 1500);
      };

      ws.onerror = (error) => {
        console.error("WS Error:", error);
        setIsConnected(false);
      };

      wsRef.current = ws;
    } catch (err) {
      console.error("WS Exception:", err);
    }
  };

  const handleSendCommand = (commandText) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      setIsProcessing(true);
      wsRef.current.send(JSON.stringify({
        type: 'text_command',
        command: commandText,
        model: selectedModel,
        speak: true
      }));
    } else {
      connectWebSocket();
      setLogs((prev) => [...prev, { type: 'status', message: "Reconnecting to JARVIS Core... Please re-send command in a moment." }]);
    }
  };

  const handleTriggerGridOverlay = async () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return new Promise((resolve) => {
        const handler = (e) => {
          const data = JSON.parse(e.data);
          if (data.type === 'grid_overlay') {
            wsRef.current.removeEventListener('message', handler);
            resolve(data);
          }
        };
        wsRef.current.addEventListener('message', handler);
        wsRef.current.send(JSON.stringify({ type: 'quick_action', action: 'screenshot_grid' }));
      });
    }
    return null;
  };

  const handleStopSpeech = () => {
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'quick_action', action: 'stop_speech' }));
    }
  };

  const handleUpdateMemory = (key, value) => {
    fetch('http://localhost:8000/api/memory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key, value })
    })
      .then(res => res.json())
      .then(data => {
        setMemory(prev => ({ ...prev, [key]: value }));
      })
      .catch(err => console.error("Memory update error:", err));
  };

  return (
    <div className="app-container">
      {/* Top Header Navigation */}
      <header className="glass-card header-bar">
        <div className="flex-gap-3">
          <div style={{
            width: '48px', height: '48px', borderRadius: '16px',
            background: 'linear-gradient(135deg, #06b6d4, #3b82f6, #8b5cf6)',
            display: 'flex', items: 'center', justifyContent: 'center',
            boxShadow: '0 0 20px rgba(6, 182, 212, 0.4)'
          }}>
            <Brain size={28} style={{ color: '#fff' }} />
          </div>
          <div>
            <div className="flex-gap-2">
              <h1 style={{
                fontSize: '1.5rem', fontWeight: 900, letterSpacing: '0.05em',
                background: 'linear-gradient(90deg, #22d3ee, #60a5fa, #c084fc)',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent'
              }}>
                JARVIS AI BRAIN
              </h1>
              <span style={{
                fontSize: '0.65rem', fontFamily: 'JetBrains Mono', fontWeight: 700,
                background: 'rgba(6, 182, 212, 0.15)', color: '#22d3ee',
                border: '1px solid rgba(6, 182, 212, 0.4)', padding: '2px 8px', borderRadius: '999px'
              }}>
                v1.0 LOCAL
              </span>
            </div>
            <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '2px' }}>
              Autonomous PC Vision, Voice & Tool Execution Core
            </p>
          </div>
        </div>

        {/* Tab Controls */}
        <div style={{
          display: 'flex', alignItems: 'center', background: 'rgba(3, 7, 18, 0.9)',
          padding: '6px', borderRadius: '16px', border: '1px solid rgba(51, 65, 85, 0.6)'
        }}>
          <button 
            onClick={() => setActiveTab('command')}
            className={`tab-btn ${activeTab === 'command' ? 'active' : ''}`}
          >
            <Radio size={16} />
            <span>Voice & Command</span>
          </button>
          <button 
            onClick={() => setActiveTab('vision')}
            className={`tab-btn ${activeTab === 'vision' ? 'active' : ''}`}
          >
            <Eye size={16} />
            <span>Screen Vision</span>
          </button>
          <button 
            onClick={() => setActiveTab('system')}
            className={`tab-btn ${activeTab === 'system' ? 'active' : ''}`}
          >
            <Server size={16} />
            <span>Telemetry & Memory</span>
          </button>
        </div>

        {/* Status Indicator */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px',
          borderRadius: '12px', fontSize: '0.75rem', fontFamily: 'JetBrains Mono', fontWeight: 700,
          background: isConnected ? 'rgba(6, 78, 59, 0.5)' : 'rgba(136, 19, 55, 0.5)',
          border: isConnected ? '1px solid rgba(16, 185, 129, 0.5)' : '1px solid rgba(244, 63, 94, 0.5)',
          color: isConnected ? '#34d399' : '#fb7185'
        }}>
          {isConnected ? <Wifi size={16} /> : <WifiOff size={16} />}
          <span>{isConnected ? 'CORE ONLINE' : 'RECONNECTING...'}</span>
        </div>
      </header>

      {/* Tab 1: Voice & Command Center */}
      {activeTab === 'command' && (
        <div className="grid-two-col">
          <VoiceOrb 
            isSpeaking={isSpeaking}
            isProcessing={isProcessing}
            onSendVoiceCommand={handleSendCommand}
            selectedModel={selectedModel}
            onSelectModel={setSelectedModel}
            onStopSpeech={handleStopSpeech}
          />
          <ConsoleLogs 
            logs={logs}
            onSendCommand={handleSendCommand}
            isProcessing={isProcessing}
          />
        </div>
      )}

      {/* Tab 2: Desktop Screen Vision */}
      {activeTab === 'vision' && (
        <div className="grid-two-col">
          <ScreenPerception 
            screenFrame={screenFrame}
            onTriggerGrid={handleTriggerGridOverlay}
          />
          <ConsoleLogs 
            logs={logs}
            onSendCommand={handleSendCommand}
            isProcessing={isProcessing}
          />
        </div>
      )}

      {/* Tab 3: Telemetry & Memory */}
      {activeTab === 'system' && (
        <div className="grid-two-col">
          <SystemTelemetry telemetry={telemetry} specs={specs} />
          <UserKnowledge 
            memory={memory}
            onUpdateMemory={handleUpdateMemory}
          />
        </div>
      )}
    </div>
  );
}
