import React, { useState, useRef, useEffect } from 'react';
import { Send, CheckCircle2, AlertCircle, Wrench, Brain, User, MessageSquare } from 'lucide-react';

export default function ConsoleLogs({ logs, onSendCommand, isProcessing }) {
  const [inputText, setInputText] = useState('');
  const logsEndRef = useRef(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputText.trim() && !isProcessing) {
      onSendCommand(inputText.trim());
      setInputText('');
    }
  };

  return (
    <div className="glass-card p-6 flex flex-col h-full min-h-[460px]">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <MessageSquare size={18} />
          </div>
          <div>
            <h3 className="font-bold text-sm text-slate-100">Live Agent Console</h3>
            <p className="text-[11px] text-slate-400">Real-time thought chain & PC task execution history</p>
          </div>
        </div>
        <span className="text-xs font-mono font-semibold text-cyan-400 bg-slate-900 border border-slate-800 px-3 py-1 rounded-full">
          {logs.length} Events
        </span>
      </div>

      {/* Messages Stream */}
      <div className="flex-1 overflow-y-auto space-y-3 pr-2 mb-4 max-h-[380px]">
        {logs.length === 0 ? (
          <div className="h-full min-h-[220px] flex flex-col items-center justify-center text-center p-6 text-slate-500 text-xs">
            <Brain size={36} className="mb-2 text-slate-600 opacity-60" />
            <p className="font-semibold text-slate-400">JARVIS Console Ready</p>
            <p className="max-w-xs mt-1 text-slate-500">Speak or type a command above. JARVIS will display its thought process and tool execution here in real time.</p>
          </div>
        ) : (
          logs.map((log, index) => (
            <div key={index} className="transition-all">
              {log.type === 'status' && (
                <div className="flex items-center gap-2 text-xs font-medium text-cyan-400 bg-cyan-950/40 border border-cyan-800/40 px-3 py-2 rounded-xl">
                  <Brain size={14} className="animate-spin text-cyan-300" />
                  <span>{log.message}</span>
                </div>
              )}

              {log.type === 'thought' && (
                <div className="bg-slate-900/90 border border-purple-500/30 rounded-2xl p-4 space-y-2">
                  <div className="flex items-center gap-2 text-xs font-bold text-purple-400 uppercase tracking-wider">
                    <Brain size={14} />
                    <span>Thought Reasoning</span>
                  </div>
                  <p className="text-xs text-slate-200 leading-relaxed font-sans whitespace-pre-wrap">{log.text}</p>
                </div>
              )}

              {log.type === 'tool_exec' && (
                <div className="bg-emerald-950/30 border border-emerald-500/30 rounded-2xl p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-xs font-bold text-emerald-400 uppercase tracking-wider">
                      <Wrench size={14} />
                      <span>PC Tool Executed: {log.tool}</span>
                    </div>
                    <span className="text-[10px] font-mono bg-emerald-900/60 text-emerald-300 px-2 py-0.5 rounded">SUCCESS</span>
                  </div>
                  <div className="text-xs font-mono text-slate-300 bg-slate-950/60 p-2.5 rounded-xl space-y-1">
                    <div><span className="text-slate-500">Input:</span> <span className="text-cyan-300">{JSON.stringify(log.input)}</span></div>
                    <div><span className="text-slate-500">Output:</span> <span className="text-emerald-300">{JSON.stringify(log.output)}</span></div>
                  </div>
                </div>
              )}

              {log.type === 'response' && (
                <div className="bg-slate-900 border border-cyan-500/40 rounded-2xl p-4 space-y-2 shadow-lg shadow-cyan-500/5">
                  <div className="flex items-center gap-2 text-xs font-bold text-cyan-300 uppercase tracking-wider">
                    <CheckCircle2 size={16} className="text-cyan-400" />
                    <span>JARVIS Response</span>
                  </div>
                  <p className="text-sm font-medium text-slate-100 leading-relaxed">{log.text}</p>
                </div>
              )}

              {log.type === 'error' && (
                <div className="bg-rose-950/40 border border-rose-500/40 text-rose-300 text-xs p-3 rounded-xl flex items-center gap-2">
                  <AlertCircle size={16} className="text-rose-400 shrink-0" />
                  <span>{log.message}</span>
                </div>
              )}
            </div>
          ))
        )}
        <div ref={logsEndRef} />
      </div>

      {/* Command Input Box */}
      <form onSubmit={handleSubmit} className="flex items-center gap-2 pt-2 border-t border-slate-800">
        <input 
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type any PC command e.g. 'Open Notepad', 'Check my RAM', 'Take screenshot'..."
          disabled={isProcessing}
          className="flex-1 bg-slate-900/90 border border-slate-700/80 rounded-xl px-4 py-3 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition font-sans"
        />
        <button 
          type="submit"
          disabled={!inputText.trim() || isProcessing}
          className="btn-primary py-3 text-xs font-bold disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Send size={15} />
          <span>Send</span>
        </button>
      </form>
    </div>
  );
}
