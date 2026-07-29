import React, { useState } from 'react';
import { Eye, Grid, Maximize2, ShieldCheck } from 'lucide-react';

export default function ScreenPerception({ screenFrame, onTriggerGrid }) {
  const [showGridOverlay, setShowGridOverlay] = useState(false);
  const [gridOverlayImage, setGridOverlayImage] = useState(null);

  const handleToggleGrid = async () => {
    if (!showGridOverlay) {
      const data = await onTriggerGrid();
      if (data && data.grid) {
        setGridOverlayImage(data.grid.image);
      }
      setShowGridOverlay(true);
    } else {
      setShowGridOverlay(false);
      setGridOverlayImage(null);
    }
  };

  return (
    <div className="glass-card p-6 flex flex-col justify-between space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Eye size={18} />
          </div>
          <div>
            <h3 className="font-bold text-sm text-slate-100">Live Desktop Vision Stream</h3>
            <p className="text-[11px] text-slate-400">Real-time screen perception feed for visual actions</p>
          </div>
        </div>

        <button 
          onClick={handleToggleGrid}
          className={`btn-secondary text-xs flex items-center gap-1.5 ${
            showGridOverlay ? 'border-cyan-400 text-cyan-300 bg-cyan-950/40' : ''
          }`}
        >
          <Grid size={14} />
          <span>{showGridOverlay ? 'Hide Grid' : 'Coordinate Grid'}</span>
        </button>
      </div>

      {/* Screen Frame Container */}
      <div className="relative rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 min-h-[260px] flex items-center justify-center shadow-inner">
        {showGridOverlay && gridOverlayImage ? (
          <img 
            src={gridOverlayImage} 
            alt="Coordinate Grid Desktop Frame" 
            className="w-full h-auto max-h-[380px] object-contain rounded-xl" 
          />
        ) : screenFrame ? (
          <img 
            src={screenFrame} 
            alt="Live Desktop Frame Stream" 
            className="w-full h-auto max-h-[380px] object-contain rounded-xl" 
          />
        ) : (
          <div className="flex flex-col items-center text-slate-500 text-xs p-8">
            <Eye size={36} className="mb-2 text-slate-600 animate-pulse" />
            <span className="font-semibold text-slate-400">Connecting Desktop Perception Stream...</span>
          </div>
        )}

        {/* Live HUD Badge */}
        <div className="absolute top-3 right-3 bg-slate-950/80 backdrop-blur border border-emerald-500/40 text-emerald-400 text-[10px] font-mono px-3 py-1 rounded-full flex items-center gap-1.5 font-bold shadow-lg">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
          <span>VISION ACTIVE (~2 FPS)</span>
        </div>
      </div>

      <div className="flex items-center justify-between text-xs text-slate-400 font-mono pt-1">
        <span className="flex items-center gap-1"><ShieldCheck size={14} className="text-cyan-400" /> Target Resolution: Full PC Desktop</span>
        <span className="text-cyan-300">Action Grid: 100px Scale</span>
      </div>
    </div>
  );
}
