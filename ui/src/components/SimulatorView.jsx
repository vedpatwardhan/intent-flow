import React, { useRef, useState, useEffect } from 'react';
import { Focus, Navigation, Camera, Loader2 } from 'lucide-react';

export default function SimulatorView({ frames, candidateResults, onInteraction, connectionStatus, isExecuting }) {
  const [activeCam, setActiveCam] = useState('world_center');
  const [stepIdx, setStepIdx] = useState(0);

  const cameras = [
    { id: 'world_center', name: 'Center' },
    { id: 'world_top', name: 'Top' },
    { id: 'world_left', name: 'Left' },
    { id: 'world_right', name: 'Right' },
    { id: 'world_wrist', name: 'Wrist' }
  ];

  const handleCameraChange = (camId) => {
    setActiveCam(camId);
  };

  const topCandidates = candidateResults?.top_candidates || Array.from({ length: 8 }, (_, i) => ({
    rank: i + 1,
    candidate_idx: i,
    mean_phys_dist: 0.0,
    frames: null,
    frame_sequences: null
  }));

  // 3 full iterations limit with 1-second pause between iterations
  const seq = candidateResults?.top_candidates?.[0]?.frame_sequences?.[activeCam];
  const hasSequences = seq && seq.length > 0;

  // Reset animation state whenever new candidate evaluation results arrive
  useEffect(() => {
    setStepIdx(0);
  }, [candidateResults]);

  // Run candidate animation loop: 5 FPS sequence playback with a 1-second pause between the 3 cycles
  useEffect(() => {
    if (!hasSequences) return;

    let timeoutId;
    let intervalId;
    let iteration = 0;

    const playSequence = () => {
      let currentStep = 0;
      setStepIdx(0);

      intervalId = setInterval(() => {
        currentStep += 1;
        if (currentStep < 3) {
          setStepIdx(currentStep);
        } else {
          clearInterval(intervalId);
          iteration += 1;
          if (iteration < 3) {
            // Wait 1 second before starting next playback iteration
            timeoutId = setTimeout(playSequence, 1000);
          }
        }
      }, 600); // 5 FPS playback
    };

    playSequence();

    return () => {
      clearInterval(intervalId);
      clearTimeout(timeoutId);
    };
  }, [candidateResults, activeCam, hasSequences]);

  return (
    <div className="panel" style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div className="panel-header" style={{ borderBottom: '1px solid var(--border-glass)', paddingBottom: '10px', marginBottom: '8px', flexShrink: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 className="panel-title flex-row-center gap-8">
              <Camera className="text-cyan-400" size={18} />
              <span>Command Center Viewport</span>
            </h2>
            <p className="panel-subtitle">Top 8 evaluated action candidates ranked by mean physical distance</p>
          </div>
          {isExecuting && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                background: 'rgba(34, 197, 94, 0.12)',
                border: '1px solid rgba(34, 197, 94, 0.35)',
                color: '#4ade80',
                padding: '4px 10px',
                borderRadius: '6px',
                fontSize: '11px',
                fontWeight: 600,
                fontFamily: 'monospace',
                letterSpacing: '0.3px'
              }}
            >
              <Loader2 className="animate-spin text-green-400" size={14} />
              <span>Sampling Action Candidates...</span>
            </div>
          )}
        </div>

        {/* Camera Selector Tabs styled identically to Encoder Diagnostics */}
        <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: '8px', marginTop: '10px', overflowX: 'auto', paddingBottom: '2px' }}>
          {cameras.map((cam) => {
            const isSelected = activeCam === cam.id;
            const camFrame = frames && frames[cam.id];

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
                  minWidth: '100px',
                  transition: 'all 0.2s',
                  flexShrink: 0
                }}
              >
                {camFrame ? (
                  <img
                    src={camFrame.startsWith('data:') || camFrame.startsWith('blob:') ? camFrame : `data:image/jpeg;base64,${camFrame}`}
                    alt={cam.name}
                    style={{ width: '28px', height: '28px', borderRadius: '4px', objectFit: 'cover' }}
                  />
                ) : (
                  <div style={{ width: '28px', height: '28px', borderRadius: '4px', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '8px', color: '#475569' }}>
                    Off
                  </div>
                )}
                <span style={{ fontSize: '11px', fontWeight: 600, color: isSelected ? 'var(--accent-cyan)' : '#94a3b8' }}>
                  {cam.name}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="panel-content" style={{ flexGrow: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', paddingRight: 0 }}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gridTemplateRows: 'repeat(2, 1fr)',
            gap: '8px',
            flexGrow: 1,
            minHeight: 0,
            height: '100%'
          }}
        >
          {topCandidates.map((cand, idx) => {
            const seq = cand.frame_sequences && cand.frame_sequences[activeCam];
            const frameSrc = seq && seq.length > 0
              ? seq[stepIdx % seq.length]
              : (cand.frames && cand.frames[activeCam] ? cand.frames[activeCam] : (frames && frames[activeCam] ? frames[activeCam] : null));
            const isRankOne = idx === 0;

            return (
              <div
                key={idx}
                style={{
                  background: 'rgba(255, 255, 255, 0.015)',
                  border: isRankOne ? '1px solid rgba(34, 197, 94, 0.4)' : '1px solid var(--border-glass)',
                  borderRadius: '6px',
                  padding: '6px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px',
                  position: 'relative',
                  minHeight: 0,
                  overflow: 'hidden'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
                  <span style={{ fontSize: '10px', fontWeight: 600, fontFamily: 'monospace', color: '#f8fafc' }}>
                    Candidate #{idx + 1}
                  </span>
                  <span
                    style={{
                      fontSize: '8px',
                      fontWeight: 700,
                      fontFamily: 'monospace',
                      color: isRankOne ? '#4ade80' : 'var(--accent-cyan)',
                      background: isRankOne ? 'rgba(34, 197, 94, 0.15)' : 'var(--accent-cyan-dim)',
                      border: isRankOne ? '1px solid rgba(34, 197, 94, 0.3)' : '1px solid rgba(6, 182, 212, 0.2)',
                      padding: '1px 5px',
                      borderRadius: '4px'
                    }}
                  >
                    {isRankOne ? `⭐ Best (d=${cand.mean_phys_dist}m)` : `Rank #${idx + 1} (${cand.mean_phys_dist}m)`}
                  </span>
                </div>

                <div style={{ flex: 1, minHeight: 0, width: '100%', background: '#000', border: '1px solid #1a1a24', borderRadius: '4px', overflow: 'hidden', position: 'relative' }}>
                  {frameSrc ? (
                    <img
                      src={frameSrc.startsWith('data:') || frameSrc.startsWith('blob:') ? frameSrc : `data:image/jpeg;base64,${frameSrc}`}
                      alt={`Candidate ${idx + 1}`}
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    />
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#475569', fontSize: '10px' }}>
                      Waiting for trajectory...
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
