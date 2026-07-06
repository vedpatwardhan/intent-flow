import React, { useState } from 'react';
import { ShieldAlert, Send, Sliders, FileCode } from 'lucide-react';

export default function ControlPanel({ onUserCommand, onComboStocChange, onTriggerAttack }) {
  const [textPrompt, setTextPrompt] = useState('pinch the green block and lift it');
  const [attackActive, setAttackActive] = useState(false);
  const [timesteps, setTimesteps] = useState({
    torso: 0,
    arm: 0,
    hand: 0,
    vision: 0
  });
  
  const [xmlContent, setXmlContent] = useState(
`<mujoco model="humanoid_pinch">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="block" pos="0.4 0 0.1">
      <geom size="0.02 0.02 0.02" type="box"/>
    </body>
  </worldbody>
</mujoco>`);

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

  return (
    <div className="glass-panel flex flex-col gap-5 h-full overflow-y-auto">
      <div>
        <h2 className="font-semibold text-lg text-neutral-100 flex items-center gap-2">
          <Sliders className="text-cyan-400" size={20} />
          Control Center
        </h2>
        <p className="text-xs text-neutral-500">Parameter tuning & state intervention</p>
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

      <hr className="border-neutral-850" />

      {/* MuJoCo XML editor */}
      <div className="flex flex-col gap-2 flex-grow">
        <label className="text-xs text-neutral-400 font-semibold uppercase tracking-wider flex items-center gap-1">
          <FileCode size={14} />
          MuJoCo XML (Interactive Mock)
        </label>
        <textarea
          value={xmlContent}
          onChange={(e) => setXmlContent(e.target.value)}
          className="w-full flex-grow bg-neutral-950 border border-neutral-900 focus:border-cyan-500 rounded p-2 text-xs font-mono text-neutral-400 outline-none resize-none transition"
          rows={6}
        />
      </div>
    </div>
  );
}
