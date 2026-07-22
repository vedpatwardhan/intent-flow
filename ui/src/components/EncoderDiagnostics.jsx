import React, { useState } from 'react';
import { Target, Image as ImageIcon, Eye, Grid, Move, RotateCcw, Shuffle, Sparkles, Plus, Trash2, Crosshair, Camera } from 'lucide-react';
import UnifiedWorkspace from './UnifiedWorkspace';
import CriticalSubspace from './CriticalSubspace';

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

const HeatmapCanvas = ({ dataMatrix, colorMap }) => {
  const canvasRef = React.useRef(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 240, 240);

    if (!dataMatrix || dataMatrix.length === 0) return;

    const rows = dataMatrix.length;
    const cols = dataMatrix[0]?.length || 0;
    if (cols === 0) return;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = cols;
    tempCanvas.height = rows;
    const tempCtx = tempCanvas.getContext('2d');
    const imgData = tempCtx.createImageData(cols, rows);

    const clamp = (num, min, max) => Math.min(Math.max(num, min), max);
    const getJetRGB = (v) => {
      const r = clamp(Math.min(4 * v - 1.5, -4 * v + 4.5), 0, 1) * 255;
      const g = clamp(Math.min(4 * v - 0.5, -4 * v + 3.5), 0, 1) * 255;
      const b = clamp(Math.min(4 * v + 0.5, -4 * v + 2.5), 0, 1) * 255;
      return [Math.round(r), Math.round(g), Math.round(b)];
    };

    // Find max value in matrix to normalize raw metric magnitudes (like VGGT)
    let maxVal = 1e-8;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (dataMatrix[r]?.[c] > maxVal) {
          maxVal = dataMatrix[r][c];
        }
      }
    }

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        let val = dataMatrix[r]?.[c] || 0.0;
        // If this is the high-res 224x224 motion field, normalize to [0, 1] range for visual contrast
        if (rows === 224) {
          val = val / maxVal;
        }
        const idx = (r * cols + c) * 4;
        if (val > 0.05) {
          const [red, green, blue] = getJetRGB(val);
          imgData.data[idx] = red;
          imgData.data[idx + 1] = green;
          imgData.data[idx + 2] = blue;
          imgData.data[idx + 3] = Math.round(val * 0.45 * 255);
        } else {
          imgData.data[idx] = 0;
          imgData.data[idx + 1] = 0;
          imgData.data[idx + 2] = 0;
          imgData.data[idx + 3] = 0;
        }
      }
    }
    tempCtx.putImageData(imgData, 0, 0);

    // Upscale to 240x240 with bilinear filtering
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tempCanvas, 0, 0, cols, rows, 0, 0, 240, 240);
  }, [dataMatrix, colorMap]);

  return (
    <canvas
      ref={canvasRef}
      width={240}
      height={240}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        borderRadius: '6px'
      }}
    />
  );
};

export default function EncoderDiagnostics({
  frame,
  frames,
  dinoAttn,
  clipSim,
  samMask,
  pointCloud,
  motionField,
  activeCam,
  onCameraChange,
  onInteraction,
  taskIsolatedFeatures,
  isTraining,
  trainingProgress,
  trainingStatus
}) {
  const [inputText, setInputText] = useState('right hand to the red cube');
  const [selectedJointIdx, setSelectedJointIdx] = useState(16); // Default right_shoulder_pitch
  const [activeSliders, setActiveSliders] = useState([16, 17, 29, 30, 31]);
  const [jointValues, setJointValues] = useState({});

  // Independent click markers for each mode
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

  // Click captures mapped directly to individual functions
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
    return <HeatmapCanvas dataMatrix={dataMatrix} colorMap={colorMap} />;
  };

  const renderVggtTracks = () => {
    if (!motionField || motionField.length === 0) {
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
          Move objects in simulation to trace motion...
        </div>
      );
    }
    return <HeatmapCanvas dataMatrix={motionField} colorMap="cyan" />;
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

    // 1. Exact Plotly Camera Transformation Matrix
    // Match custom_camera: eye=[0.1, 0.5, 2.0], up=[0, 1, 0]
    const eye = [0.1, 0.5, 2.0];
    const eyeNorm = Math.sqrt(eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]);
    const forward = [-eye[0] / eyeNorm, -eye[1] / eyeNorm, -eye[2] / eyeNorm];
    const up_initial = [0.0, 1.0, 0.0];

    // Deriving orthogonal system (Gram-Schmidt)
    const right_raw = [-forward[2], 0.0, forward[0]];
    const rightNorm = Math.sqrt(right_raw[0] * right_raw[0] + right_raw[2] * right_raw[2]);
    const right = [right_raw[0] / rightNorm, 0.0, right_raw[2] / rightNorm];

    const up = [
      right[1] * forward[2] - right[2] * forward[1],
      right[2] * forward[0] - right[0] * forward[2],
      right[0] * forward[1] - right[1] * forward[0]
    ];

    // 2. Camera Space Transformation and Projection
    const pointsWithDepth = pointCloud.map((pt) => {
      const px = pt[0];
      const py = pt[1];
      const pz = pt[2];

      const x_cam = px * right[0] + py * right[1] + pz * right[2];
      const y_cam = px * up[0] + py * up[1] + pz * up[2];
      const z_cam = px * forward[0] + py * forward[1] + pz * forward[2] + 2.0;

      // Perspective focal scaling (Zoom factor)
      const focal = 2.4;
      const x_proj = (x_cam * focal) / z_cam;
      const y_proj = (y_cam * focal) / z_cam;

      // Map to 240x240 local viewport layout
      const screenX = ((x_proj + 1.0) / 2.0) * 240;
      const screenY = ((1.0 - y_proj) / 2.0) * 240;

      let pointColor = '#3b82f6';
      if (pt.length >= 6) {
        const r = Math.round(pt[3] * 255);
        const g = Math.round(pt[4] * 255);
        const b = Math.round(pt[5] * 255);
        pointColor = `rgb(${r},${g},${b})`;
      } else {
        const activation = pt[3] || 0.0;
        pointColor = activation > 0.6 ? '#facc15' : activation > 0.3 ? '#ef4444' : '#3b82f6';
      }

      return { screenX, screenY, z_cam, pointColor };
    });

    // 3. Painter's Algorithm: Sort by depth (z_cam) descending to draw back-to-front
    pointsWithDepth.sort((a, b) => b.z_cam - a.z_cam);

    const projectedPoints = pointsWithDepth.map((p, idx) => {
      // Clip circles to stay inside quadrant boundaries
      if (p.screenX < 0 || p.screenX > 240 || p.screenY < 0 || p.screenY > 240) {
        return null;
      }
      return (
        <circle
          key={idx}
          cx={p.screenX}
          cy={p.screenY}
          r="1.8"
          fill={p.pointColor}
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

  const triggerRecordExemplar = (name) => {
    if (onInteraction) {
      onInteraction({
        type: 'record_exemplar',
        name: name
      });
    }
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
            <hr className="separator" style={{ margin: '6px 0' }} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <span className="form-label" style={{ fontSize: '9px', color: '#64748b' }}>Stage 3 Training Sandbox:</span>
              {isTraining ? (
                <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--accent-cyan)' }}>
                    <span>{trainingStatus}</span>
                    <span>{Math.round(trainingProgress * 100)}%</span>
                  </div>
                  <div style={{ width: '100%', background: '#09090d', borderRadius: '4px', height: '6px', overflow: 'hidden', border: '1px solid var(--border-glass)' }}>
                    <div
                      style={{ background: 'var(--accent-cyan)', height: '100%', width: `${trainingProgress * 100}%`, transition: 'all 0.3s' }}
                    ></div>
                  </div>
                </div>
              ) : (
                <button
                  onClick={() => onInteraction({ type: 'start_training' })}
                  style={{
                    background: 'var(--accent-cyan-dim)',
                    border: '1px solid rgba(6, 182, 212, 0.3)',
                    color: 'var(--accent-cyan)',
                    padding: '6px 12px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    width: '100%',
                    transition: 'all 0.2s',
                    textAlign: 'center'
                  }}
                >
                  Start Stage 3 Training
                </button>
              )}
            </div>
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
                            cursor: isTraining ? 'default' : 'pointer',
                            opacity: isTraining ? 0.3 : 1,
                            padding: '2px',
                            display: 'flex',
                            alignItems: 'center'
                          }}
                          disabled={isTraining}
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
                    disabled={isTraining}
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
                  disabled={isTraining}
                  style={{ padding: '6px', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent-amber)' }}
                >
                  <Shuffle size={10} />
                  Randomize
                </button>
                <button
                  onClick={triggerReset}
                  className="btn-phase btn-phase-action"
                  disabled={isTraining}
                  style={{ padding: '6px', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent-red)' }}
                >
                  <RotateCcw size={10} />
                  Home All
                </button>
              </div>
            </div>
          </div>

          {/* Third Block: Exemplar Recorder & Goal Alignment Audit */}
          <div className="panel" style={{ display: 'flex', flexDirection: 'column', padding: '10px 12px', boxSizing: 'border-box' }}>
            <div className="panel-header" style={{ marginBottom: '6px' }}>
              <span className="panel-title" style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Camera size={13} className="text-cyan-400" />
                Exemplar Recorder & Goal Alignment Audit
              </span>
            </div>
            <p style={{ fontSize: '9px', color: '#64748b', margin: '0 0 8px 0' }}>
              Record observation footprints (images, proprioception, tactile) for Stage 3 latent distance auditing across 4 operational phases.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px' }}>
              {['phase_1', 'phase_2', 'phase_3', 'phase_4'].map((phaseName, idx) => (
                <button
                  key={phaseName}
                  onClick={() => triggerRecordExemplar(phaseName)}
                  disabled={isTraining}
                  style={{
                    background: '#09090d',
                    border: '1px solid rgba(6, 182, 212, 0.3)',
                    color: 'var(--accent-cyan)',
                    padding: '6px 8px',
                    borderRadius: '4px',
                    fontSize: '10px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '4px',
                    transition: 'all 0.2s'
                  }}
                >
                  <Camera size={11} />
                  Phase {idx + 1}
                </button>
              ))}
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

          {/* Single Grid Row: Unified Workspace + Vertical Modality Dashboard */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '420px 1fr',
            gap: '16px',
            flexGrow: 1,
            minHeight: 0,
            overflow: 'hidden'
          }}>
            <UnifiedWorkspace
              frames={frames}
              activeCam={activeCam}
              onInteraction={onInteraction}
              samMask={samMask}
            />
            <CriticalSubspace
              frame={activeFrame}
              isolatedFeatures={taskIsolatedFeatures}
              dinoAttn={dinoAttn}
              clipSim={clipSim}
              vggTracks={motionField}
            />
          </div>

        </div>

      </div>
    </div>
  );
}
