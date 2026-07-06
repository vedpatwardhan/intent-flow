import React, { useState } from 'react';
import { ShieldAlert, Send, Sliders, Play, RotateCcw, HelpCircle } from 'lucide-react';

export default function ControlPanel({ onUserCommand, onComboStocChange, onTriggerAttack }) {
  const [textPrompt, setTextPrompt] = useState('pinch the red block and lift it');
  const [attackActive, setAttackActive] = useState(false);
  const [timesteps, setTimesteps] = useState({
    torso: 0.0,
    arm: 0.0,
    hand: 0.0,
    vision: 0.0
  });

  const handleSubmitPrompt = (e) => {
    e.preventDefault();
    if (!textPrompt.trim()) return;
    onUserCommand({
      type: 'text_command',
      prompt: textPrompt
    });
  };

  const handleSliderChange = (group, value) => {
    const newVal = parseFloat(value);
    setTimesteps(prev => ({ ...prev, [group]: newVal }));
    onComboStocChange(group, newVal);
  };

  const toggleAttack = () => {
    const nextState = !attackActive;
    setAttackActive(nextState);
    onTriggerAttack(nextState);
  };

  const triggerIkPhase = (phase) => {
    onUserCommand({
      type: 'ik_command',
      phase: phase
    });
  };

  const triggerReset = () => {
    onUserCommand({
      type: 'reset'
    });
  };

  const triggerWildRandomize = () => {
    onUserCommand({
      type: 'wild_randomize'
    });
  };

  return (
    <div className="glass-panel flex flex-col gap-5 h-full overflow-y-auto">
      <div>
        <h2 className="font-semibold text-lg text-neutral-100 flex items-center gap-2">
          <Sliders className="text-cyan-400" size={20} />
          Control Center
        </h2>
        <p className="text-xs text-neutral-500">Parameter tuning & physical state intervention</p>
      </div>

      {/* Task input */}
      <form onSubmit={handleSubmitPrompt} className="flex flex-col gap-2">
        <label className="text-xs text-neutral-400 font-semibold uppercase tracking-wider">Goal Command (CLIP)</label>
        <div className="flex gap-2">
          <input
            type="text"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            className="flex-grow bg-neutral-900 border border-neutral-800 focus:border-cyan-500 rounded px-3 py-2 text-sm outline-none text-white transition"
            placeholder="Type task prompt..."
          />
          <button type="submit" className="bg-cyan-500 hover:bg-cyan-600 text-black px-3 py-2 rounded transition flex items-center justify-center">
            <Send size={16} />
          </button>
        </div>
      </form>

      <hr className="border-neutral-850" />

      {/* Actual MuJoCo IK Phase Controllers */}
      <div className="flex flex-col gap-2">
        <label className="text-xs text-neutral-400 font-semibold uppercase tracking-wider flex items-center gap-1">
          <Play size={14} className="text-green-400" />
          Interactive IK Pickup Phases
        </label>
        <div className="grid grid-cols-2 gap-2 mt-1">
          <button
            onClick={triggerReset}
            className="py-2 px-3 text-xs bg-neutral-900 hover:bg-neutral-800 border border-neutral-800 hover:border-neutral-700 rounded transition flex items-center justify-center gap-1 text-neutral-200"
          >
            <RotateCcw size={12} />
            Reset Env
          </button>
          <button
            onClick={triggerWildRandomize}
            className="py-2 px-3 text-xs bg-neutral-900 hover:bg-neutral-800 border border-neutral-800 hover:border-neutral-700 rounded transition flex items-center justify-center gap-1 text-neutral-200"
          >
            <HelpCircle size={12} />
            Wild Randomize
          </button>
          <button
            onClick={() => triggerIkPhase(0)}
            className="py-2 px-3 text-xs bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/20 hover:border-cyan-500/40 text-cyan-400 rounded transition text-left"
          >
            Phase 0: Reach
          </button>
          <button
            onClick={() => triggerIkPhase(1)}
            className="py-2 px-3 text-xs bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/20 hover:border-cyan-500/40 text-cyan-400 rounded transition text-left"
          >
            Phase 1: Descent
          </button>
          <button
            onClick={() => triggerIkPhase(2)}
            className="py-2 px-3 text-xs bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/20 hover:border-cyan-500/40 text-cyan-400 rounded transition text-left"
          >
            Phase 2: Grasp
          </button>
          <button
            onClick={() => triggerIkPhase(3)}
            className="py-2 px-3 text-xs bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/20 hover:border-cyan-500/40 text-cyan-400 rounded transition text-left"
          >
            Phase 3: Lift
          </button>
        </div>
      </div>

      <hr className="border-neutral-850" />

      {/* ComboStoc sliders */}
      <div className="flex flex-col gap-3">
        <label className="text-xs text-neutral-400 font-semibold uppercase tracking-wider">ComboStoc Asynchronous Noise ($t$)</label>
        {Object.entries(timesteps).map(([group, val]) => (
          <div key={group} className="flex flex-col gap-1">
            <div className="flex justify-between text-xs text-neutral-300">
              <span className="capitalize">{group} Timeline</span>
              <span className="font-mono text-cyan-400">{val.toFixed(2)}s</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={val}
              onChange={(e) => handleSliderChange(group, e.target.value)}
              className="w-full h-1 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
          </div>
        ))}
      </div>

      <hr className="border-neutral-850" />

      {/* BadWorld Attack trigger */}
      <div className="flex flex-col gap-2">
        <label className="text-xs text-neutral-400 font-semibold uppercase tracking-wider">Adversarial Playbook</label>
        <button
          onClick={toggleAttack}
          className={`flex items-center justify-center gap-2 py-3 rounded font-bold border transition ${
            attackActive 
              ? 'bg-red-500 text-white border-red-400 glow-red pulse-active' 
              : 'bg-red-500/10 text-red-400 border-red-500/20 hover:bg-red-500/20'
          }`}
        >
          <ShieldAlert size={18} />
          {attackActive ? 'BadWorld Attack Engaged' : 'Inject BadWorld Attack'}
        </button>
      </div>
    </div>
  );
}
