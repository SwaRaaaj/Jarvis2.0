import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Volume2, VolumeX, Cpu, Sparkles, Radio, Zap, Infinity } from 'lucide-react';

export default function VoiceOrb({ 
  isSpeaking, 
  isProcessing, 
  onSendVoiceCommand, 
  selectedModel, 
  onSelectModel, 
  onStopSpeech 
}) {
  const [isListening, setIsListening] = useState(false);
  const [continuousMode, setContinuousMode] = useState(true); // Default to Hands-Free Continuous Listening
  const [transcript, setTranscript] = useState('');
  const [ttsMuted, setTtsMuted] = useState(false);
  const recognitionRef = useRef(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const rec = new SpeechRecognition();
      rec.continuous = false;
      rec.interimResults = true;
      rec.lang = 'en-US';

      rec.onresult = (event) => {
        let currentTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript;
        }
        setTranscript(currentTranscript);

        if (event.results[0].isFinal) {
          const finalUtterance = currentTranscript.trim();
          setTranscript('');
          if (finalUtterance) {
            onSendVoiceCommand(finalUtterance);
          }
        }
      };

      rec.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        if (event.error !== 'no-speech') {
          setIsListening(false);
        }
      };

      rec.onend = () => {
        setIsListening(false);
        // Hands-Free Auto Restart: If continuous mode is enabled and JARVIS is not currently speaking/processing
        if (continuousMode && !isProcessing && !isSpeaking) {
          setTimeout(() => {
            try {
              rec.start();
              setIsListening(true);
            } catch (e) {}
          }, 400);
        }
      };

      recognitionRef.current = rec;

      // Auto start continuous listening on mount
      if (continuousMode) {
        try {
          rec.start();
          setIsListening(true);
        } catch (e) {}
      }
    }
  }, [continuousMode]);

  // Restart listening after JARVIS finishes processing & speaking if in continuous mode
  useEffect(() => {
    if (continuousMode && !isProcessing && !isSpeaking && recognitionRef.current && !isListening) {
      const timer = setTimeout(() => {
        try {
          recognitionRef.current.start();
          setIsListening(true);
        } catch (e) {}
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [isProcessing, isSpeaking, continuousMode]);

  const toggleListening = () => {
    if (!recognitionRef.current) {
      alert("Speech recognition is not supported in this browser. Please use Google Chrome or Edge.");
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      setTranscript('');
      try {
        recognitionRef.current.start();
        setIsListening(true);
      } catch (e) {}
    }
  };

  const toggleContinuousMode = () => {
    const nextVal = !continuousMode;
    setContinuousMode(nextVal);
    if (!nextVal && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  const quickCommands = [
    { label: "Open Notepad", cmd: "Open Notepad and type 'JARVIS AI Brain Online'", icon: "📝" },
    { label: "Launch Chrome", cmd: "Open Google Chrome browser", icon: "🌐" },
    { label: "Hardware Stats", cmd: "What is my CPU and RAM usage?", icon: "📊" },
    { label: "Take Screenshot", cmd: "Take a screenshot of my screen", icon: "📷" },
    { label: "Open Terminal", cmd: "Open Command Prompt terminal", icon: "⚡" }
  ];

  const getStatusBadge = () => {
    if (isListening) return { text: "HANDS-FREE LISTENING ACTIVE...", bg: "bg-emerald-500/20 text-emerald-300 border-emerald-500/50" };
    if (isProcessing) return { text: "THINKING & EXECUTING TASK...", bg: "bg-cyan-500/20 text-cyan-300 border-cyan-500/50" };
    if (isSpeaking) return { text: "JARVIS SPEAKING OUT LOUD...", bg: "bg-purple-500/20 text-purple-300 border-purple-500/50" };
    return { text: "MICROPHONE PAUSED", bg: "bg-slate-800 text-slate-400 border-slate-700" };
  };

  const status = getStatusBadge();

  return (
    <div className="glass-card p-6 md:p-8 flex flex-col items-center justify-center text-center relative overflow-hidden">
      {/* Top Controls Header */}
      <div className="w-full flex items-center justify-between mb-6 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <div className={`w-2.5 h-2.5 rounded-full ${isListening ? 'bg-emerald-400 animate-ping' : 'bg-slate-500'}`} />
          <span className="text-xs font-bold tracking-wider text-slate-300 uppercase">
            {continuousMode ? 'Hands-Free Voice Mode' : 'Tap-To-Speak Mode'}
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* Hands Free Toggle */}
          <button 
            onClick={toggleContinuousMode}
            className={`btn-secondary text-xs flex items-center gap-1.5 ${
              continuousMode ? 'border-emerald-400 text-emerald-300 bg-emerald-950/40' : ''
            }`}
            title="Toggle Hands-Free Continuous Voice Mode"
          >
            <Infinity size={15} className={continuousMode ? 'animate-spin text-emerald-400' : ''} />
            <span>{continuousMode ? 'Hands-Free: ON' : 'Hands-Free: OFF'}</span>
          </button>

          {/* Model Switcher */}
          <div className="flex items-center gap-2 bg-slate-900/90 border border-slate-700/80 px-3 py-1.5 rounded-xl">
            <Cpu size={14} className="text-cyan-400" />
            <select 
              value={selectedModel} 
              onChange={(e) => onSelectModel(e.target.value)}
              className="bg-transparent text-xs text-cyan-300 font-semibold outline-none cursor-pointer"
            >
              <option value="qwen2.5-coder:3b" className="bg-slate-900 text-slate-200">qwen2.5-coder:3b (Fast Actions)</option>
              <option value="gemma3:4b" className="bg-slate-900 text-slate-200">gemma3:4b (Reasoning)</option>
            </select>
          </div>

          {/* Mute TTS */}
          <button 
            onClick={onStopSpeech}
            title={isSpeaking ? "Stop Speech Output" : "Voice Output Mute/Unmute"}
            className={`p-2 rounded-xl border transition ${
              isSpeaking 
                ? 'bg-purple-500/20 border-purple-500/50 text-purple-300 animate-pulse' 
                : 'bg-slate-900 border-slate-700 text-slate-400 hover:text-slate-200'
            }`}
          >
            {isSpeaking ? <Volume2 size={18} /> : <VolumeX size={18} />}
          </button>
        </div>
      </div>

      {/* Hero Interactive Orb */}
      <div className="my-6 relative flex items-center justify-center">
        <div 
          onClick={toggleListening}
          className={`hero-orb ${isListening ? 'listening' : isSpeaking ? 'speaking' : ''}`}
          title="Click to toggle Mic manually"
        >
          <div className="hero-orb-ring-1" />
          <div className="hero-orb-ring-2" />

          {isListening ? (
            <Mic size={54} className="text-emerald-100 animate-pulse" />
          ) : isProcessing ? (
            <Sparkles size={54} className="text-cyan-200 animate-spin" />
          ) : isSpeaking ? (
            <div className="flex items-end gap-1.5 h-8">
              <div className="wave-bar" style={{ animationDelay: '0s' }} />
              <div className="wave-bar" style={{ animationDelay: '0.2s' }} />
              <div className="wave-bar" style={{ animationDelay: '0.4s' }} />
              <div className="wave-bar" style={{ animationDelay: '0.1s' }} />
            </div>
          ) : (
            <MicOff size={50} className="text-slate-400 opacity-60" />
          )}
        </div>
      </div>

      {/* State Badge & Dynamic Transcript */}
      <div className="space-y-2 mb-6">
        <div className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full border text-xs font-bold tracking-widest uppercase ${status.bg}`}>
          <Radio size={14} className="animate-pulse" />
          <span>{status.text}</span>
        </div>

        <p className="text-sm font-medium text-slate-200 max-w-lg min-h-[28px] flex items-center justify-center">
          {transcript ? (
            <span className="text-emerald-300 italic font-mono font-bold text-base">"{transcript}"</span>
          ) : (
            <span className="text-slate-400 text-xs">
              {continuousMode 
                ? "🎙️ Hands-Free Mode Active: Speak anytime! JARVIS is listening automatically."
                : "Tap the Orb to speak a command manually."}
            </span>
          )}
        </p>
      </div>

      {/* Quick Action Shortcuts */}
      <div className="w-full pt-4 border-t border-slate-800/80">
        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest block mb-3">Quick Voice Shortcuts</span>
        <div className="flex flex-wrap items-center justify-center gap-2">
          {quickCommands.map((item, idx) => (
            <button
              key={idx}
              onClick={() => onSendVoiceCommand(item.cmd)}
              disabled={isProcessing}
              className="bg-slate-900/80 hover:bg-slate-800 border border-slate-700/80 hover:border-cyan-500/50 text-slate-200 hover:text-cyan-300 text-xs font-semibold px-3.5 py-2 rounded-xl transition flex items-center gap-2 disabled:opacity-50"
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
