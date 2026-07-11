import React, { useState } from 'react';
import { ShieldAlert, Send, Sliders, Play, RotateCcw, HelpCircle } from 'lucide-react';

export default function ControlPanel({
  onUserCommand,
  onComboStocChange,
  onTriggerAttack,
  isTraining,
  trainingProgress,
  trainingStatus,
}) {
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
              disabled={isTraining}
            />
            <button type="submit" className="btn-send" disabled={isTraining}>
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
            <button onClick={triggerReset} className="btn-phase btn-phase-action" disabled={isTraining}>
              <RotateCcw size={12} />
              Reset Env
            </button>
            <button onClick={triggerWildRandomize} className="btn-phase btn-phase-action" disabled={isTraining}>
              <HelpCircle size={12} />
              Wild Rand
            </button>
            <button onClick={() => triggerIkPhase(0)} className="btn-phase" disabled={isTraining}>
              Phase 0: Reach
            </button>
            <button onClick={() => triggerIkPhase(1)} className="btn-phase" disabled={isTraining}>
              Phase 1: Descent
            </button>
            <button onClick={() => triggerIkPhase(2)} className="btn-phase" disabled={isTraining}>
              Phase 2: Grasp
            </button>
            <button onClick={() => triggerIkPhase(3)} className="btn-phase" disabled={isTraining}>
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
                disabled={isTraining}
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
            disabled={isTraining}
          >
            <ShieldAlert size={16} />
            {attackActive ? 'BadWorld Attack Engaged' : 'Inject BadWorld Attack'}
          </button>
        </div>

        <hr className="separator" />

        {/* Stage 3 Training Trigger */}
        <div className="form-group">
          <label className="form-label">Stage 3 Training Sandbox</label>
          {isTraining ? (
            <div className="w-full flex flex-col gap-2">
              <div className="flex justify-between text-xs text-cyan-400">
                <span>{trainingStatus}</span>
                <span>{Math.round(trainingProgress * 100)}%</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-cyan-500 h-full transition-all duration-300"
                  style={{ width: `${trainingProgress * 100}%` }}
                ></div>
              </div>
            </div>
          ) : (
            <button
              onClick={() => onUserCommand({ type: 'start_training' })}
              className="btn-adversary bg-cyan-600 hover:bg-cyan-500 text-white flex items-center justify-center gap-2 py-2 px-4 rounded w-full transition-all"
            >
              <Sliders size={16} />
              Start Stage 3 Training
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
