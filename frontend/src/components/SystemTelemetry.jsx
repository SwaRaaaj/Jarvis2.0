import React from 'react';
import { Cpu, HardDrive, Activity, Layout, Server, Zap, Layers } from 'lucide-react';

export default function SystemTelemetry({ telemetry, specs }) {
  if (!telemetry) {
    return (
      <div className="glass-card p-8 flex items-center justify-center text-slate-400 text-xs">
        <Activity className="animate-spin mr-2 text-cyan-400" size={20} /> Loading System Telemetry Stream...
      </div>
    );
  }

  const activeWin = telemetry.active_window || {};

  return (
    <div className="glass-card p-6 space-y-6">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-400">
            <Server size={18} />
          </div>
          <div>
            <h3 className="font-bold text-sm text-slate-100">PC Telemetry & Specs</h3>
            <p className="text-[11px] text-slate-400">Live hardware monitoring & active app process</p>
          </div>
        </div>
        <span className="text-xs font-mono font-bold text-cyan-300 bg-cyan-950/60 border border-cyan-800/60 px-3 py-1 rounded-full">
          {specs?.hostname || 'LOCAL-PC'}
        </span>
      </div>

      {/* Main Hardware Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* CPU Meter */}
        <div className="bg-slate-900/80 p-4 rounded-2xl border border-slate-800 space-y-2">
          <div className="flex justify-between items-center text-xs">
            <span className="text-slate-400 font-semibold flex items-center gap-1.5">
              <Cpu size={16} className="text-cyan-400" /> CPU Load
            </span>
            <span className="font-mono text-cyan-300 font-extrabold text-sm">{telemetry.cpu_percent}%</span>
          </div>
          <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden p-0.5 border border-slate-800">
            <div 
              className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, telemetry.cpu_percent)}%` }}
            />
          </div>
          <p className="text-[11px] text-slate-400 pt-0.5">{specs?.processor ? specs.processor.split('@')[0] : 'Multi-Core CPU'}</p>
        </div>

        {/* RAM Meter */}
        <div className="bg-slate-900/80 p-4 rounded-2xl border border-slate-800 space-y-2">
          <div className="flex justify-between items-center text-xs">
            <span className="text-slate-400 font-semibold flex items-center gap-1.5">
              <Activity size={16} className="text-purple-400" /> RAM Memory
            </span>
            <span className="font-mono text-purple-300 font-extrabold text-sm">{telemetry.ram_percent}%</span>
          </div>
          <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden p-0.5 border border-slate-800">
            <div 
              className="bg-gradient-to-r from-purple-500 to-pink-500 h-full rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, telemetry.ram_percent)}%` }}
            />
          </div>
          <p className="text-[11px] text-slate-400 pt-0.5">{telemetry.ram_used_gb} GB used of {specs?.total_ram_gb || '--'} GB</p>
        </div>
      </div>

      {/* Active Foreground Application Card */}
      <div className="bg-slate-900/90 p-4 rounded-2xl border border-cyan-500/30 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400 font-semibold flex items-center gap-1.5">
            <Layout size={16} className="text-emerald-400" /> Current Active Focused App
          </span>
          <span className="text-[10px] font-mono bg-emerald-950 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded-full font-bold">
            PID: {activeWin.pid || 'N/A'}
          </span>
        </div>
        <p className="text-sm font-bold text-slate-100 truncate">{activeWin.title || 'Desktop'}</p>
        <div className="flex items-center justify-between text-xs text-slate-400 font-mono pt-1">
          <span>Process: <strong className="text-cyan-300">{activeWin.app || 'explorer.exe'}</strong></span>
          {activeWin.width && <span>Res: {activeWin.width}x{activeWin.height}</span>}
        </div>
      </div>

      {/* Top Active Running Processes */}
      <div className="space-y-2">
        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-1.5">
          <Layers size={14} className="text-cyan-400" /> Top System Processes
        </h4>
        <div className="space-y-2 max-h-40 overflow-y-auto pr-1">
          {(telemetry.top_processes || []).map((proc, idx) => (
            <div key={idx} className="flex justify-between items-center text-xs bg-slate-900/60 p-2.5 rounded-xl border border-slate-800/80">
              <span className="font-mono text-slate-200 font-medium truncate max-w-[180px]">{proc.name}</span>
              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="text-cyan-400 font-semibold">CPU: {proc.cpu}%</span>
                <span className="text-purple-400 font-semibold">RAM: {proc.mem}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
