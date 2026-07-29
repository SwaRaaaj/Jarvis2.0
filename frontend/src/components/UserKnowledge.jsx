import React, { useState } from 'react';
import { Database, Plus, Edit2, Save, ShieldAlert, Key } from 'lucide-react';

export default function UserKnowledge({ memory, onUpdateMemory }) {
  const [editingKey, setEditingKey] = useState('');
  const [newValue, setNewValue] = useState('');
  const [newKey, setNewKey] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);

  const handleSave = (key) => {
    if (key && newValue !== undefined) {
      onUpdateMemory(key, newValue);
      setEditingKey('');
    }
  };

  const handleAddNew = (e) => {
    e.preventDefault();
    if (newKey.trim() && newValue.trim()) {
      onUpdateMemory(newKey.trim(), newValue.trim());
      setNewKey('');
      setNewValue('');
      setShowAddModal(false);
    }
  };

  return (
    <div className="glass-card p-6 space-y-6">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-purple-500/10 border border-purple-500/30 flex items-center justify-center text-purple-400">
            <Database size={18} />
          </div>
          <div>
            <h3 className="font-bold text-sm text-slate-100">JARVIS Memory Vault</h3>
            <p className="text-[11px] text-slate-400">Persistent user facts & PC setup parameters</p>
          </div>
        </div>

        <button 
          onClick={() => setShowAddModal(!showAddModal)}
          className="btn-secondary text-xs flex items-center gap-1"
        >
          <Plus size={14} /> Add Fact
        </button>
      </div>

      {showAddModal && (
        <form onSubmit={handleAddNew} className="bg-slate-900 p-4 rounded-2xl border border-purple-500/40 space-y-3">
          <h4 className="text-xs font-bold text-purple-300">Add New Custom Memory Fact</h4>
          <input 
            type="text"
            placeholder="Fact Title (e.g. favorite_browser)"
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 text-xs text-slate-200 px-3 py-2 rounded-xl outline-none"
          />
          <input 
            type="text"
            placeholder="Fact Value (e.g. Google Chrome)"
            value={newValue}
            onChange={(e) => setNewValue(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 text-xs text-slate-200 px-3 py-2 rounded-xl outline-none"
          />
          <div className="flex justify-end gap-2 pt-1">
            <button 
              type="button" 
              onClick={() => setShowAddModal(false)}
              className="text-xs text-slate-400 px-3 py-1.5"
            >
              Cancel
            </button>
            <button 
              type="submit"
              className="btn-primary text-xs py-1.5 px-4"
            >
              Save Fact
            </button>
          </div>
        </form>
      )}

      {/* Memory Fact Cards */}
      <div className="space-y-2.5 max-h-[300px] overflow-y-auto pr-1">
        {Object.entries(memory || {}).map(([key, val]) => (
          <div key={key} className="bg-slate-900/60 p-3 rounded-2xl border border-slate-800 flex items-center justify-between text-xs hover:border-slate-700 transition">
            <div className="space-y-1 max-w-[80%]">
              <span className="font-mono text-cyan-400 font-bold uppercase tracking-wider text-[11px] block">{key}</span>
              {editingKey === key ? (
                <input 
                  type="text" 
                  value={newValue} 
                  onChange={(e) => setNewValue(e.target.value)}
                  className="bg-slate-950 border border-cyan-500 text-slate-200 px-3 py-1 rounded-lg w-full outline-none font-sans"
                />
              ) : (
                <span className="text-slate-200 font-medium block leading-relaxed">{val}</span>
              )}
            </div>
            {editingKey === key ? (
              <button 
                onClick={() => handleSave(key)}
                className="text-emerald-400 hover:text-emerald-300 p-2"
              >
                <Save size={16} />
              </button>
            ) : (
              <button 
                onClick={() => { setEditingKey(key); setNewValue(val); }}
                className="text-slate-500 hover:text-cyan-300 p-2 transition"
              >
                <Edit2 size={15} />
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Emergency Stop Key Card */}
      <div className="bg-rose-950/20 border border-rose-500/30 p-3.5 rounded-2xl flex items-center gap-3 text-xs text-rose-300">
        <ShieldAlert size={20} className="shrink-0 text-rose-400" />
        <span className="leading-relaxed">
          Emergency Stop: Press <strong className="font-mono underline text-white font-bold">ESC</strong> key or move mouse to top-left corner anytime to halt OS actions.
        </span>
      </div>
    </div>
  );
}
