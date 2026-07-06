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
    <div className="panel h-full">
      <div className="panel-header">
        <h2 className="panel-title">
          <Sliders className="text-cyan-400" size={18} />
          Control Center
        </h2>
        <p className="panel-subtitle">Parameter tuning & physical state intervention</p>
      </div>

      <div className="panel-content panel-content-flex">
        {/* Goal Command input */}
        <form onSubmit={handleSubmitPrompt} className="form-group">
          <label className="form-label">Goal Command (CLIP)</label>
          <div className="flex-row-center gap-8">
            <input
              type="text"
              value={textPrompt}
              onChange={(e) => setTextPrompt(e.target.value)}
              className="input-text flex-grow"
              placeholder="Type task prompt..."
            />
            <button type="submit" className="btn-send">
              <Send size={14} />
            </button>
          </div>
        </form>

        <hr className="separator" />

        {/* Interactive IK Phase controllers */}
        <div className="form-group">
          <label className="form-label form-label-flex">
            <Play size={12} className="text-green-400" />
            Interactive IK Pickup Phases
          </label>
          <div className="grid-phases">
            <button onClick={triggerReset} className="btn-phase btn-phase-action">
              <RotateCcw size={12} />
              Reset Env
            </button>
            <button onClick={triggerWildRandomize} className="btn-phase btn-phase-action">
              <HelpCircle size={12} />
              Wild Rand
            </button>
            <button onClick={() => triggerIkPhase(0)} className="btn-phase">
              Phase 0: Reach
            </button>
            <button onClick={() => triggerIkPhase(1)} className="btn-phase">
              Phase 1: Descent
            </button>
            <button onClick={() => triggerIkPhase(2)} className="btn-phase">
              Phase 2: Grasp
            </button>
            <button onClick={() => triggerIkPhase(3)} className="btn-phase">
              Phase 3: Lift
            </button>
          </div>
        </div>

        <hr className="separator" />

        {/* ComboStoc timeline noise sliders */}
        <div className="form-group gap-12">
          <label className="form-label">ComboStoc Asynchronous Noise ($t$)</label>
          {Object.entries(timesteps).map(([group, val]) => (
            <div key={group} className="slider-group">
              <div className="slider-header">
                <span className="slider-name">{group} Timeline</span>
                <span className="slider-val">{val.toFixed(2)}s</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={val}
                onChange={(e) => handleSliderChange(group, e.target.value)}
                className="slider-input"
              />
            </div>
          ))}
        </div>

        <hr className="separator" />

        {/* BadWorld Attack trigger */}
        <div className="form-group">
          <label className="form-label">Adversarial Playbook</label>
          <button
            onClick={toggleAttack}
            className={`btn-adversary ${attackActive ? 'active' : 'inactive'}`}
          >
            <ShieldAlert size={16} />
            {attackActive ? 'BadWorld Attack Engaged' : 'Inject BadWorld Attack'}
          </button>
        </div>
      </div>
    </div>
  );
}
