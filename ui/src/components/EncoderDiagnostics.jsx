import React, { useState } from 'react';
import { Target, Image as ImageIcon, Eye, Grid, Move, RotateCcw, Shuffle, Sparkles, Plus, Trash2, Crosshair } from 'lucide-react';

const COMPACT_WIRE_JOINTS = [
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
  "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint", "L_index_proximal_joint", "L_middle_proximal_joint", "L_ring_proximal_joint", "L_pinky_proximal_joint",
  "head_pitch_joint", "head_roll_joint", "head_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
  "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint", "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
  "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"
];

const CAMERAS = [
  { id: 'world_center', name: 'Center' },
  { id: 'world_top', name: 'Top' },
  { id: 'world_left', name: 'Left' },
  { id: 'world_right', name: 'Right' },
  { id: 'world_wrist', name: 'Wrist' }
];

export default function EncoderDiagnostics({
  frame,
  frames,
  dinoAttn,
  clipSim,
  samMask,
  pointCloud,
  vggtTracks,
  activeCam,
  onCameraChange,
  onInteraction
}) {
  const [inputText, setInputText] = useState('cube block');
  const [selectedJointIdx, setSelectedJointIdx] = useState(16); // Default right_shoulder_pitch
  const [activeSliders, setActiveSliders] = useState([16, 17, 29, 30, 31]);
  const [jointValues, setJointValues] = useState({});
  
  // Decoupled click markers indexed by camera ID
  const [segmentMarkers, setSegmentMarkers] = useState({});
  const [trackMarkers, setTrackMarkers] = useState({});
  const [goalMarkers, setGoalMarkers] = useState({});

  const handleTextSubmit = (e) => {
    e.preventDefault();
    onInteraction({
      type: 'text_prompt',
      text: inputText
    });
  };

  const addSlider = () => {
    if (!activeSliders.includes(selectedJointIdx)) {
      setActiveSliders(prev => [...prev, selectedJointIdx].sort((a, b) => a - b));
    }
  };

  const removeSlider = (idx) => {
    setActiveSliders(prev => prev.filter(item => item !== idx));
  };

  const handleSliderChange = (idx, value) => {
    const valFloat = parseFloat(value);
    setJointValues(prev => ({ ...prev, [idx]: valFloat }));
    onInteraction({
      type: 'set_joint',
      index: idx,
      value: valFloat
    });
  };

  const triggerIKPhase = (phase) => {
    onInteraction({
      type: 'ik_command',
      phase
    });
  };

  const triggerReset = () => {
    onInteraction({ type: 'reset' });
    setJointValues({});
    setSegmentMarkers({});
    setTrackMarkers({});
    setGoalMarkers({});
  };

  const triggerRandomize = () => {
    onInteraction({ type: 'wild_randomize' });
    setJointValues({});
    setSegmentMarkers({});
    setTrackMarkers({});
    setGoalMarkers({});
  };

  const handleCameraChange = (camId) => {
    onCameraChange(camId);
    onInteraction({
      type: 'select_camera',
      camera: camId
    });
  };

  // Click captures mapped directly to individual functions and decoupled by camera ID
  const handleSegmentClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 224);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 224);
    setSegmentMarkers(prev => ({ ...prev, [activeCam]: { x, y } }));
    onInteraction({ type: 'original_click', x, y });
  };

  const handleTrackClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 224);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 224);
    setTrackMarkers(prev => ({ ...prev, [activeCam]: { x, y } }));
    onInteraction({ type: 'track_click', x, y });
  };

  const handleGoalClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 224);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 224);
    setGoalMarkers(prev => ({ ...prev, [activeCam]: { x, y } }));
    onInteraction({ type: 'goal_click', x, y });
  };

  const handleClearSelections = () => {
    setSegmentMarkers(prev => ({ ...prev, [activeCam]: null }));
    setTrackMarkers(prev => ({ ...prev, [activeCam]: null }));
    setGoalMarkers(prev => ({ ...prev, [activeCam]: null }));
    onInteraction({ type: 'clear_selections' });
  };

  const renderHeatmapOverlay = (dataMatrix, colorMap) => {
    if (!dataMatrix || dataMatrix.length === 0) {
      return (
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'rgba(0, 0, 0, 0.65)',
          color: '#94a3b8',
          fontSize: '10px',
          fontFamily: 'monospace'
        }}>
          Waiting for GPU features...
        </div>
      );
    }

    const grid = [];
    const patchSize = 240 / 14;

    for (let r = 0; r < 14; r++) {
      for (let c = 0; c < 14; c++) {
        const val = dataMatrix[r]?.[c] || 0.0;
        if (val > 0.05) {
          let fillStyle = '';
          if (colorMap === 'inferno') {
            fillStyle = `rgba(249, 115, 22, ${val * 0.85})`;
          } else {
            fillStyle = `rgba(6, 182, 212, ${val * 0.85})`;
          }

          grid.push(
            <rect
              key={`${r}-${c}`}
              x={c * patchSize}
              y={r * patchSize}
              width={patchSize}
              height={patchSize}
              fill={fillStyle}
            />
          );
        }
      }
    }

    return (
      <svg viewBox="0 0 240 240" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
        {grid}
      </svg>
    );
  };

  const renderVggtTracks = () => {
    if (!vggtTracks || vggtTracks.length === 0) {
      return (
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'rgba(0, 0, 0, 0.65)',
          color: '#94a3b8',
          fontSize: '10px',
          fontFamily: 'monospace'
        }}>
          Move objects in simulation to trace tracks...
        </div>
      );
    }

    return (
      <svg viewBox="0 0 240 240" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
        {vggtTracks.map((pt, idx) => {
          const x = pt[0] * 240;
          const y = pt[1] * 240;
          return (
            <g key={idx}>
              <circle cx={x} cy={y} r="3" fill="var(--accent-cyan)" />
              <line x1={x} y1={y} x2={x - 5} y2={y - 5} stroke="var(--accent-cyan)" strokeWidth="1.5" opacity="0.6" />
            </g>
          );
        })}
      </svg>
    );
  };

  const renderPointNextCloud = () => {
    if (!pointCloud || pointCloud.length === 0) {
      return (
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'rgba(0, 0, 0, 0.65)',
          color: '#94a3b8',
          fontSize: '10px',
          fontFamily: 'monospace'
        }}>
          Click segment viewport above to trigger point cloud...
        </div>
      );
    }

    const projectedPoints = pointCloud.map((pt, idx) => {
      const px = pt[0] * 80;
      const py = pt[1] * 80;
      const pz = pt[2] * 40;

      const screenX = 120 + (px - py) * 0.8;
      const screenY = 120 + (px + py) * 0.4 - pz;

      const activation = pt[3] || 0.0;
      const pointColor = activation > 0.6 ? '#facc15' : activation > 0.3 ? '#ef4444' : '#3b82f6';

      return (
        <circle
          key={idx}
          cx={screenX}
          cy={screenY}
          r="2.5"
          fill={pointColor}
          opacity="0.9"
        />
      );
    });

    return (
      <svg viewBox="0 0 240 240" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', background: '#020204' }}>
        {projectedPoints}
      </svg>
    );
  };

  const renderClickReticle = (marker, reticleColor) => {
    if (!marker) return null;
    const xRatio = (marker.x / 224) * 240;
    const yRatio = (marker.y / 224) * 240;

    return (
      <svg viewBox="0 0 240 240" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
        <circle cx={xRatio} cy={yRatio} r="6" fill="none" stroke={reticleColor} strokeWidth="1.5" />
        <line x1={xRatio - 12} y1={yRatio} x2={xRatio + 12} y2={yRatio} stroke={reticleColor} strokeWidth="1" />
        <line x1={xRatio} y1={yRatio - 12} x2={xRatio} y2={yRatio + 12} stroke={reticleColor} strokeWidth="1" />
      </svg>
    );
  };

  const activeFrame = frames?.[activeCam] || frame;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%', boxSizing: 'border-box', padding: '10px 16px' }}>
      
      {/* Split Workspace Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '12px', flexGrow: 1, minHeight: 0 }}>
        
        {/* Left Column: Squeezed Header Banner + Joint Sliders */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', height: '100%', overflow: 'hidden' }}>
          
          {/* Header Panel squeezed to Left Column Width (320px) */}
          <div className="panel" style={{ padding: '8px 12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <h2 className="panel-title" style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
              <Eye size={14} className="text-cyan-400" />
              Diagnostics & Teleop
            </h2>
            <form onSubmit={handleTextSubmit} style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
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
          </div>

          {/* Joint Management Sliders card */}
          <div className="panel" style={{ display: 'flex', flexDirection: 'column', padding: '12px', overflow: 'hidden', flexGrow: 1, boxSizing: 'border-box' }}>
            <div className="panel-header" style={{ marginBottom: '6px' }}>
              <span className="panel-title" style={{ fontSize: '11px' }}>Joint Management & State Audit</span>
            </div>

            {/* Add Joint selector row with truncation */}
            <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
              <select
                value={selectedJointIdx}
                onChange={(e) => setSelectedJointIdx(parseInt(e.target.value))}
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
                style={{
                  background: 'var(--accent-cyan-dim)',
                  border: '1px solid rgba(6, 182, 212, 0.3)',
                  color: 'var(--accent-cyan)',
                  padding: '4px 10px',
                  borderRadius: '4px',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
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
              {activeSliders.map((idx) => {
                const currentVal = jointValues[idx] || 0.0;
                return (
                  <div key={idx} style={{
                    background: 'rgba(255, 255, 255, 0.015)',
                    border: '1px solid var(--border-glass)',
                    borderRadius: '6px',
                    padding: '6px 8px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '4px'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '10px', fontFamily: 'monospace', color: '#94a3b8', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', maxWidth: '180px' }} title={COMPACT_WIRE_JOINTS[idx]}>
                        [{idx}] {COMPACT_WIRE_JOINTS[idx]}
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ fontSize: '10px', fontFamily: 'monospace', color: 'var(--accent-cyan)', fontWeight: 700 }}>
                          {currentVal.toFixed(2)}
                        </span>
                        <button
                          onClick={() => removeSlider(idx)}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: 'var(--accent-red)',
                            cursor: 'pointer',
                            padding: '2px',
                            display: 'flex',
                            alignItems: 'center'
                          }}
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
                      value={currentVal}
                      onChange={(e) => handleSliderChange(idx, e.target.value)}
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
                );
              })}
            </div>

            {/* Quick Actions Footer */}
            <div style={{
              borderTop: '1px solid var(--border-glass)',
              paddingTop: '8px',
              marginTop: '8px',
              display: 'flex',
              flexDirection: 'column',
              gap: '6px'
            }}>
              <div style={{ fontSize: '9px', color: '#64748b', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Sparkles size={11} className="text-cyan-400" />
                INTEGRATION SHIELD ACTIONS
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '4px' }}>
                {['Approach', 'Descend', 'Grasp', 'Lift'].map((phaseName, i) => (
                  <button
                    key={i}
                    onClick={() => triggerIKPhase(i)}
                    className="btn-phase btn-phase-action"
                    style={{ padding: '6px', fontSize: '9px', textAlign: 'center', justifyContent: 'center' }}
                  >
                    {phaseName}
                  </button>
                ))}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', borderTop: '1px solid rgba(255,255,255,0.03)', paddingTop: '6px' }}>
                <button
                  onClick={triggerRandomize}
                  className="btn-phase btn-phase-action"
                  style={{ padding: '6px', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent-amber)' }}
                >
                  <Shuffle size={10} />
                  Randomize
                </button>
                <button
                  onClick={triggerReset}
                  className="btn-phase btn-phase-action"
                  style={{ padding: '6px', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent-red)' }}
                >
                  <RotateCcw size={10} />
                  Home All
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Camera View Selector & Visual Overlays Grid */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'hidden', height: '100%' }}>
          
          {/* Live Camera Selection Row forced to horizontal layout */}
          <div className="panel" style={{
            padding: '8px 12px',
            display: 'flex',
            flexDirection: 'row',
            gap: '8px',
            overflowX: 'auto',
            flexShrink: 0,
            width: '100%',
            boxSizing: 'border-box'
          }}>
            {CAMERAS.map((cam) => {
              const isSelected = activeCam === cam.id;
              const hasFrame = frames && frames[cam.id];
              return (
                <div
                  key={cam.id}
                  onClick={() => handleCameraChange(cam.id)}
                  style={{
                    display: 'flex',
                    flexDirection: 'row',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '6px 12px',
                    borderRadius: '6px',
                    border: `1px solid ${isSelected ? 'var(--accent-cyan)' : 'var(--border-glass)'}`,
                    background: isSelected ? 'var(--accent-cyan-dim)' : 'rgba(255,255,255,0.01)',
                    cursor: 'pointer',
                    minWidth: '110px',
                    transition: 'all 0.2s',
                    flexShrink: 0
                  }}
                >
                  {hasFrame ? (
                    <img src={frames[cam.id]} alt={cam.name} style={{ width: '32px', height: '32px', borderRadius: '4px', objectFit: 'cover' }} />
                  ) : (
                    <div style={{ width: '32px', height: '32px', borderRadius: '4px', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '8px', color: '#475569' }}>
                      Off
                    </div>
                  )}
                  <span style={{ fontSize: '11px', fontWeight: 600, color: isSelected ? 'var(--accent-cyan)' : '#94a3b8' }}>
                    {cam.name}
                  </span>
                </div>
              );
            })}
            <button
              onClick={handleClearSelections}
              style={{
                background: 'var(--accent-red-dim)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                color: 'var(--accent-red)',
                padding: '6px 12px',
                borderRadius: '6px',
                fontSize: '11px',
                fontWeight: 600,
                cursor: 'pointer',
                marginLeft: 'auto',
                alignSelf: 'center',
                transition: 'all 0.2s'
              }}
            >
              Clear Selections
            </button>
          </div>

          {/* Dedicated Row 1: 3 Interactive Viewports stretching to match Row 2 card widths */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '12px',
            flexShrink: 0
          }}>
            
            {/* Viewport 1: SAM Interactive Segmenter */}
            <div className="panel" style={{ padding: '10px', display: 'flex', flexDirection: 'column', gap: '6px', boxSizing: 'border-box' }}>
              <div className="panel-header" style={{ marginBottom: '2px' }}>
                <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: 'var(--accent-red)', fontWeight: 600 }}>
                  <Crosshair size={12} />
                  1. Segment Viewport (SAM)
                </span>
              </div>
              <div
                className="diagnostics-viewport"
                onClick={handleSegmentClick}
                style={{ position: 'relative', width: '240px', height: '240px', margin: '0 auto', background: '#000', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-glass)', cursor: 'crosshair' }}
              >
                {activeFrame && <img src={activeFrame} alt="camera" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} />}
                {renderClickReticle(segmentMarkers[activeCam], 'var(--accent-red)')}
              </div>
              <div style={{ fontSize: '8px', color: '#64748b', textAlign: 'center', fontFamily: 'monospace' }}>
                Click to segment / generate point cloud.
              </div>
            </div>

            {/* Viewport 2: VGGT Interactive Tracker */}
            <div className="panel" style={{ padding: '10px', display: 'flex', flexDirection: 'column', gap: '6px', boxSizing: 'border-box' }}>
              <div className="panel-header" style={{ marginBottom: '2px' }}>
                <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: 'var(--accent-cyan)', fontWeight: 600 }}>
                  <Crosshair size={12} />
                  2. Track Viewport (VGGT)
                </span>
              </div>
              <div
                className="diagnostics-viewport"
                onClick={handleTrackClick}
                style={{ position: 'relative', width: '240px', height: '240px', margin: '0 auto', background: '#000', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-glass)', cursor: 'crosshair' }}
              >
                {activeFrame && <img src={activeFrame} alt="camera" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} />}
                {renderClickReticle(trackMarkers[activeCam], 'var(--accent-cyan)')}
              </div>
              <div style={{ fontSize: '8px', color: '#64748b', textAlign: 'center', fontFamily: 'monospace' }}>
                Click to seed continuous visual tracks.
              </div>
            </div>

            {/* Viewport 3: Flow Interactive Goal Selector */}
            <div className="panel" style={{ padding: '10px', display: 'flex', flexDirection: 'column', gap: '6px', boxSizing: 'border-box' }}>
              <div className="panel-header" style={{ marginBottom: '2px' }}>
                <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: 'var(--accent-green)', fontWeight: 600 }}>
                  <Crosshair size={12} />
                  3. Goal Viewport (Flow)
                </span>
              </div>
              <div
                className="diagnostics-viewport"
                onClick={handleGoalClick}
                style={{ position: 'relative', width: '240px', height: '240px', margin: '0 auto', background: '#000', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-glass)', cursor: 'crosshair' }}
              >
                {activeFrame && <img src={activeFrame} alt="camera" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} />}
                {renderClickReticle(goalMarkers[activeCam], 'var(--accent-green)')}
              </div>
              <div style={{ fontSize: '8px', color: '#64748b', textAlign: 'center', fontFamily: 'monospace' }}>
                Click to anchor policy trajectory goals.
              </div>
            </div>

          </div>

          {/* Dedicated Row 2: Visual Overlays Grid (Slightly Enlarged to 240px squares) */}
          <div style={{ overflowY: 'auto', flexGrow: 1, paddingRight: '2px' }}>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '12px'
            }}>
              
              {/* DINOv3 Attn Map */}
              <div className="panel" style={{ padding: '8px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div className="panel-header" style={{ marginBottom: '2px' }}>
                  <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: 'var(--accent-cyan)' }}>
                    <Target size={11} />
                    DINOv3 Spatial Attention
                  </span>
                </div>
                <div
                  className="diagnostics-viewport"
                  style={{ position: 'relative', width: '240px', height: '240px', margin: '0 auto', background: '#000', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-glass)' }}
                >
                  {activeFrame && <img src={activeFrame} alt="camera" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} />}
                  {renderHeatmapOverlay(dinoAttn, 'inferno')}
                </div>
                <div style={{ fontSize: '9px', color: '#64748b', textAlign: 'center', fontFamily: 'monospace' }}>
                  Attention overlays.
                </div>
              </div>

              {/* CLIP Attention Heatmap */}
              <div className="panel" style={{ padding: '8px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div className="panel-header" style={{ marginBottom: '2px' }}>
                  <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: 'var(--accent-cyan)' }}>
                    <ImageIcon size={11} />
                    CLIP Cosine Similarity
                  </span>
                </div>
                <div
                  className="diagnostics-viewport"
                  style={{ position: 'relative', width: '240px', height: '240px', margin: '0 auto', background: '#000', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-glass)' }}
                >
                  {activeFrame && <img src={activeFrame} alt="camera" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} />}
                  {renderHeatmapOverlay(clipSim, 'viridis')}
                </div>
                <div style={{ fontSize: '9px', color: '#64748b', textAlign: 'center', fontFamily: 'monospace' }}>
                  Token: <code style={{ color: 'var(--accent-cyan)' }}>"{inputText}"</code>
                </div>
              </div>

              {/* SAM Instance Segmentation */}
              <div className="panel" style={{ padding: '8px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div className="panel-header" style={{ marginBottom: '2px' }}>
                  <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: 'var(--accent-green)' }}>
                    <Grid size={11} />
                    SAM Segmentation Mask
                  </span>
                </div>
                <div
                  className="diagnostics-viewport"
                  style={{ position: 'relative', width: '240px', height: '240px', margin: '0 auto', background: '#000', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-glass)' }}
                >
                  {activeFrame && <img src={activeFrame} alt="camera" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} />}
                  {samMask ? (
                    <img src={samMask} alt="sam mask" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', mixBlendMode: 'screen', opacity: 0.65 }} />
                  ) : (
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.65)', color: '#94a3b8', fontSize: '9px', textAlign: 'center', padding: '10px' }}>
                      Click segment viewport above to segment.
                    </div>
                  )}
                </div>
                <div style={{ fontSize: '9px', color: '#64748b', textAlign: 'center', fontFamily: 'monospace' }}>
                  Boundary masks.
                </div>
              </div>

              {/* VGGT Point Tracks */}
              <div className="panel" style={{ padding: '8px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div className="panel-header" style={{ marginBottom: '2px' }}>
                  <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: 'var(--accent-amber)' }}>
                    <Move size={11} />
                    VGGT Trajectory Tracks
                  </span>
                </div>
                <div
                  className="diagnostics-viewport"
                  style={{ position: 'relative', width: '240px', height: '240px', margin: '0 auto', background: '#000', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-glass)', opacity: 0.85 }}
                >
                  {activeFrame && <img src={activeFrame} alt="camera" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} />}
                  {renderVggtTracks()}
                </div>
                <div style={{ fontSize: '9px', color: '#64748b', textAlign: 'center', fontFamily: 'monospace' }}>
                  Vector tracks.
                </div>
              </div>

              {/* PointNeXt 3D Cloud */}
              <div className="panel" style={{ padding: '8px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div className="panel-header" style={{ marginBottom: '2px' }}>
                  <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: 'var(--accent-cyan)' }}>
                    <Target size={11} />
                    PointNeXt 3D Cloud
                  </span>
                </div>
                <div className="diagnostics-viewport" style={{ position: 'relative', width: '240px', height: '240px', margin: '0 auto', background: '#000', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-glass)' }}>
                  {renderPointNextCloud()}
                </div>
                <div style={{ fontSize: '9px', color: '#64748b', textAlign: 'center', fontFamily: 'monospace' }}>
                  Projected 3D cloud.
                </div>
              </div>

            </div>
          </div>
          
        </div>
        
      </div>
    </div>
  );
}
