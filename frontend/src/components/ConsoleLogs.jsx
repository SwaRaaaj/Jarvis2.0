import React, { useState, useRef, useEffect } from 'react';
import { Send, CheckCircle2, AlertCircle, Wrench, Brain, User, MessageSquare, ListChecks } from 'lucide-react';

// Full class strings, never interpolated. Tailwind's compiler scans source text for complete class
// names, so a constructed string like `bg-${tone}-950/30` produces no CSS at all.
const TOOL_TONES = {
  success: {
    box: 'bg-emerald-950/30 border-emerald-500/30',
    label: 'text-emerald-400',
    badge: 'bg-emerald-900/60 text-emerald-300',
  },
  failed: {
    box: 'bg-rose-950/30 border-rose-500/30',
    label: 'text-rose-400',
    badge: 'bg-rose-900/60 text-rose-300',
  },
  offTarget: {
    box: 'bg-amber-950/30 border-amber-500/30',
    label: 'text-amber-400',
    badge: 'bg-amber-900/60 text-amber-300',
  },
};

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

              {log.type === 'tool_exec' && (() => {
                // The status badge used to be hardcoded to SUCCESS, so a failed or misdirected
                // action looked identical to a successful one.
                const ok = log.output?.status === 'success';
                const offTarget = log.output?.on_target === false;
                const tone = TOOL_TONES[offTarget ? 'offTarget' : ok ? 'success' : 'failed'];
                return (
                  <div className={`${tone.box} border rounded-2xl p-4 space-y-2`}>
                    <div className="flex items-center justify-between">
                      <div className={`flex items-center gap-2 text-xs font-bold ${tone.label} uppercase tracking-wider`}>
                        <Wrench size={14} />
                        <span>PC Tool Executed: {log.tool}</span>
                      </div>
                      <span className={`text-[10px] font-mono ${tone.badge} px-2 py-0.5 rounded`}>
                        {offTarget ? 'OFF-TARGET' : ok ? 'SUCCESS' : 'FAILED'}
                      </span>
                    </div>
                    {log.output?.matched_name && (
                      <div className="text-xs text-slate-300">
                        <span className="text-slate-500">Hit:</span>{' '}
                        <span className="text-cyan-300 font-medium">{log.output.matched_name}</span>
                        {log.output?.grounding?.method && (
                          <span className="text-slate-500"> (grounded via {log.output.grounding.method})</span>
                        )}
                      </div>
                    )}
                    {offTarget && (
                      <p className="text-[11px] text-amber-300/90 leading-relaxed">{log.output.scope_reason}</p>
                    )}
                    <div className="text-xs font-mono text-slate-300 bg-slate-950/60 p-2.5 rounded-xl space-y-1">
                      <div><span className="text-slate-500">Input:</span> <span className="text-cyan-300">{JSON.stringify(log.input)}</span></div>
                      <div><span className="text-slate-500">Output:</span> <span className="text-slate-300">{JSON.stringify(log.output)}</span></div>
                    </div>
                  </div>
                );
              })()}

              {log.type === 'detail' && (
                <details className="bg-slate-900/70 border border-slate-700/60 rounded-2xl p-4 group">
                  <summary className="flex items-center gap-2 text-xs font-bold text-slate-300 uppercase tracking-wider cursor-pointer select-none">
                    <ListChecks size={14} className="text-slate-400" />
                    <span>Full Breakdown</span>
                    {log.data?.steps_total != null && (
                      <span className="ml-auto text-[10px] font-mono text-slate-400 normal-case">
                        {log.data.steps_done}/{log.data.steps_total} steps &middot; {log.data.elapsed_seconds}s
                        {log.data.stats?.llm_calls != null && ` · ${log.data.stats.llm_calls} model calls`}
                      </span>
                    )}
                  </summary>
                  <pre className="mt-3 text-[11px] leading-relaxed text-slate-300 whitespace-pre-wrap font-mono overflow-x-auto">{log.text}</pre>
                </details>
              )}

              {log.type === 'response' && (
                <div className="bg-slate-900 border border-cyan-500/40 rounded-2xl p-4 space-y-2 shadow-lg shadow-cyan-500/5">
                  <div className="flex items-center gap-2 text-xs font-bold text-cyan-300 uppercase tracking-wider">
                    <CheckCircle2 size={16} className="text-cyan-400" />
                    <span>JARVIS Response</span>
                    {log.stats?.llm_calls != null && (
                      <span className="ml-auto text-[10px] font-mono text-slate-500 normal-case">
                        {log.stats.llm_calls} model calls &middot; {log.stats.tree_walks} screen scans
                        {log.stats.walks_avoided > 0 && ` (${log.stats.walks_avoided} cached)`}
                      </span>
                    )}
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
