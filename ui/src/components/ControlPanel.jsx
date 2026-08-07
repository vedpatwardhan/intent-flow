import React, { useState } from 'react';
import { Sliders, Play, RotateCcw, HelpCircle, Send, Plus, Trash2, Shuffle } from 'lucide-react';

const COMPACT_WIRE_JOINTS = [
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
  "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint", "L_index_proximal_joint", "L_middle_proximal_joint", "L_ring_proximal_joint", "L_pinky_proximal_joint",
  "head_pitch_joint", "head_roll_joint", "head_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
  "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint", "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
  "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"
];

export default function ControlPanel({
  onUserCommand,
  onComboStocChange,
  onTriggerAttack,
  isTraining,
  trainingProgress,
  trainingStatus,
}) {
  const [inputText, setInputText] = useState('pinch the red block and lift it');
  const [checkpoints, setCheckpoints] = useState([
    'run_119/stage3_epoch_10.pt',
    'run_119/stage3_epoch_08.pt',
    'run_118/stage3_rl_final.pt',
    'stage2_sft.pt'
  ]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('run_119/stage3_epoch_10.pt');
  const [noiseScale, setNoiseScale] = useState(0.08);

  const [selectedJointIdx, setSelectedJointIdx] = useState(16); // right_shoulder_pitch_joint
  const [activeSliders, setActiveSliders] = useState([
    { idx: 16, name: 'right_shoulder_pitch_joint', val: 0.0 },
    { idx: 17, name: 'right_shoulder_roll_joint', val: 0.0 },
    { idx: 29, name: 'waist_yaw_joint', val: 0.0 },
    { idx: 30, name: 'waist_pitch_joint', val: 0.0 },
    { idx: 31, name: 'waist_roll_joint', val: 0.0 },
  ]);

  const handleTextSubmit = (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    onUserCommand({ type: 'text_command', prompt: inputText });
  };

  const addSlider = () => {
    if (activeSliders.some(s => s.idx === selectedJointIdx)) return;
    const jointName = COMPACT_WIRE_JOINTS[selectedJointIdx] || `joint_${selectedJointIdx}`;
    setActiveSliders(prev => [...prev, { idx: selectedJointIdx, name: jointName, val: 0.0 }]);
  };

  const removeSlider = (jointIdx) => {
    setActiveSliders(prev => prev.filter(s => s.idx !== jointIdx));
  };

  const handleJointSliderChange = (jointIdx, valStr) => {
    const val = parseFloat(valStr);
    setActiveSliders(prev => prev.map(s => s.idx === jointIdx ? { ...s, val } : s));
    onUserCommand({ type: 'set_joint', index: jointIdx, value: val });
  };

  const triggerReset = () => {
    onUserCommand({ type: 'reset' });
  };

  const triggerWildRandomize = () => {
    onUserCommand({ type: 'wild_randomize' });
  };

  const triggerExecuteCheckpoint = () => {
    onUserCommand({
      type: 'execute_checkpoint',
      checkpoint_name: selectedCheckpoint,
      step_nft_scale: noiseScale
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', height: '100%', boxSizing: 'border-box' }}>
      {/* Top Config & CLIP Prompt Card */}
      <div className="panel" style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <h2 className="panel-title" style={{ fontSize: '13px', margin: 0, display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Sliders className="text-cyan-400" size={15} />
          Diagnostics & Teleop
        </h2>

        <form onSubmit={handleTextSubmit} style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '2px' }}>
          <span className="form-label" style={{ fontSize: '9px', color: '#64748b' }}>CLIP Target:</span>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            style={{
              flexGrow: 1,
              background: '#09090d',
              border: '1px solid var(--border-glass)',
              borderRadius: '4px',
              padding: '3px 6px',
              color: '#fff',
              fontSize: '11px',
              outline: 'none',
              minWidth: 0
            }}
          />
          <button
            type="submit"
            style={{
              background: 'var(--accent-cyan-dim)',
              border: '1px solid rgba(6, 182, 212, 0.3)',
              color: 'var(--accent-cyan)',
              padding: '3px 8px',
              borderRadius: '4px',
              fontSize: '10px',
              fontWeight: 600,
              cursor: 'pointer',
              whiteSpace: 'nowrap'
            }}
          >
            Update
          </button>
        </form>

        <hr className="separator" style={{ margin: '4px 0' }} />

        {/* Checkpoint Model Selection */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <span className="form-label" style={{ fontSize: '9px', color: '#64748b' }}>Colab Model Checkpoint:</span>
          <select
            value={selectedCheckpoint}
            onChange={(e) => setSelectedCheckpoint(e.target.value)}
            disabled={isTraining}
            style={{
              width: '100%',
              background: '#09090d',
              border: '1px solid var(--border-glass)',
              borderRadius: '4px',
              padding: '4px 8px',
              color: 'var(--accent-cyan)',
              fontSize: '11px',
              fontFamily: 'monospace',
              outline: 'none'
            }}
          >
            {checkpoints.map(ckpt => (
              <option key={ckpt} value={ckpt}>{ckpt}</option>
            ))}
          </select>
        </div>

        {/* Denoising Randomization Noise Scale */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '2px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="form-label" style={{ fontSize: '9px', color: '#64748b' }}>Denoising Noise Scale (step_nft_scale):</span>
            <span style={{ fontSize: '10px', fontFamily: 'monospace', color: 'var(--accent-cyan)' }}>{noiseScale.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.00"
            max="0.50"
            step="0.01"
            value={noiseScale}
            onChange={(e) => setNoiseScale(parseFloat(e.target.value))}
            className="slider-input"
            style={{ width: '100%' }}
            disabled={isTraining}
          />
        </div>
      </div>

      {/* Joint Management & Teleoperation Sliders Panel */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', padding: '12px', overflow: 'hidden', flexGrow: 1, boxSizing: 'border-box' }}>
        <h3 className="form-label" style={{ margin: '0 0 8px 0', fontSize: '10px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Joint Management & State Audit
        </h3>

        {/* Add Joint selector row */}
        <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
          <select
            value={selectedJointIdx}
            onChange={(e) => setSelectedJointIdx(parseInt(e.target.value))}
            disabled={isTraining}
            style={{
              flexGrow: 1,
              maxWidth: '210px',
              background: '#09090d',
              border: '1px solid var(--border-glass)',
              borderRadius: '4px',
              padding: '4px 8px',
              color: '#fff',
              fontSize: '11px',
              fontFamily: 'monospace',
              outline: 'none',
              textOverflow: 'ellipsis',
              overflow: 'hidden',
              whiteSpace: 'nowrap'
            }}
          >
            {COMPACT_WIRE_JOINTS.map((name, idx) => (
              <option key={idx} value={idx}>
                [{idx}] {name}
              </option>
            ))}
          </select>
          <button
            onClick={addSlider}
            disabled={isTraining}
            style={{
              background: 'var(--accent-cyan-dim)',
              border: '1px solid rgba(6, 182, 212, 0.3)',
              color: 'var(--accent-cyan)',
              padding: '4px 10px',
              borderRadius: '4px',
              fontSize: '11px',
              fontWeight: 600,
              cursor: isTraining ? 'default' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '2px',
              whiteSpace: 'nowrap'
            }}
          >
            <Plus size={12} /> Add
          </button>
        </div>

        {/* Sliders scrolling list */}
        <div style={{ flexGrow: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px', paddingRight: '2px' }}>
          {activeSliders.map(slider => (
            <div key={slider.idx} style={{ background: 'rgba(255, 255, 255, 0.015)', border: '1px solid var(--border-glass)', borderRadius: '6px', padding: '6px 8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '10px', fontFamily: 'monospace', color: '#94a3b8', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', maxWidth: '180px' }} title={slider.name}>
                  [{slider.idx}] {slider.name}
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '10px', fontFamily: 'monospace', color: 'var(--accent-cyan)', fontWeight: 700 }}>
                    {slider.val.toFixed(2)}
                  </span>
                  <button
                    onClick={() => removeSlider(slider.idx)}
                    disabled={isTraining}
                    style={{ background: 'none', border: 'none', color: 'var(--accent-red)', cursor: 'pointer', padding: '2px', display: 'flex', alignItems: 'center' }}
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              </div>
              <input
                type="range"
                min="-1.0"
                max="1.0"
                step="0.05"
                value={slider.val}
                onChange={(e) => handleJointSliderChange(slider.idx, e.target.value)}
                disabled={isTraining}
                style={{
                  width: '100%',
                  height: '3px',
                  background: '#1e293b',
                  borderRadius: '2px',
                  outline: 'none',
                  appearance: 'none',
                  cursor: 'pointer'
                }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Integration & Execution Actions Card */}
      <div className="panel" style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <h3 className="form-label" style={{ margin: 0, fontSize: '10px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Execution & Simulation Actions
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
          <button
            onClick={triggerWildRandomize}
            className="btn-phase btn-phase-action"
            disabled={isTraining}
            style={{ padding: '6px', fontSize: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px', color: 'var(--accent-amber)' }}
          >
            <Shuffle size={10} />
            Randomize
          </button>
          <button
            onClick={triggerReset}
            className="btn-phase btn-phase-action"
            disabled={isTraining}
            style={{ padding: '6px', fontSize: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px', color: 'var(--accent-red)' }}
          >
            <RotateCcw size={10} />
            Home All
          </button>
        </div>

        <button
          onClick={triggerExecuteCheckpoint}
          disabled={isTraining}
          style={{
            background: '#09090d',
            border: '1px solid rgba(34, 197, 94, 0.4)',
            color: '#4ade80',
            padding: '6px 10px',
            borderRadius: '4px',
            fontSize: '10px',
            fontWeight: 700,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '6px',
            width: '100%',
            marginTop: '2px',
            transition: 'all 0.2s'
          }}
        >
          <Play size={11} /> Execute Checkpoint Trajectory
        </button>
      </div>
    </div>
  );
}
