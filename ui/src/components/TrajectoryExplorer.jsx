import React, { useState, useEffect, useRef } from 'react';
import { Search, Play, Pause, Film, RefreshCw, Layers } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function TrajectoryExplorer() {
  const [rollouts, setRollouts] = useState([]);
  const [selectedRollout, setSelectedRollout] = useState('');
  const [steps, setSteps] = useState([]);
  const [selectedStep, setSelectedStep] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isPlayingAll, setIsPlayingAll] = useState(false);
  const [isLoadingRollouts, setIsLoadingRollouts] = useState(true);
  const [isLoadingSteps, setIsLoadingSteps] = useState(false);

  const videoRefs = useRef([]);

  // 1. Fetch available rollout runs on mount
  useEffect(() => {
    fetchRollouts();
  }, []);

  const fetchRollouts = async () => {
    setIsLoadingRollouts(true);
    try {
      const res = await fetch(`${API_BASE}/api/rollouts`);
      if (res.ok) {
        const data = await res.json();
        setRollouts(data);
        if (data.length > 0) {
          setSelectedRollout(data[0].id);
        }
      }
    } catch (err) {
      console.error('Failed to fetch rollouts:', err);
    } finally {
      setIsLoadingRollouts(false);
    }
  };

  // 2. Fetch steps whenever selectedRollout changes
  useEffect(() => {
    if (!selectedRollout) return;

    const fetchSteps = async () => {
      setIsLoadingSteps(true);
      setIsPlayingAll(false);
      try {
        const res = await fetch(`${API_BASE}/api/rollouts/${selectedRollout}/steps`);
        if (res.ok) {
          const data = await res.json();
          setSteps(data);
          if (data.length > 0) {
            setSelectedStep(data[0]);
          } else {
            setSelectedStep(null);
          }
        }
      } catch (err) {
        console.error(`Failed to fetch steps for ${selectedRollout}:`, err);
        setSteps([]);
        setSelectedStep(null);
      } finally {
        setIsLoadingSteps(false);
      }
    };

    fetchSteps();
  }, [selectedRollout]);

  // Reset video ref array whenever step changes
  useEffect(() => {
    videoRefs.current = [];
    setIsPlayingAll(false);
  }, [selectedStep]);

  // Handle global play / pause toggle
  const togglePlayAll = () => {
    const nextState = !isPlayingAll;
    setIsPlayingAll(nextState);

    videoRefs.current.forEach(v => {
      if (v) {
        if (nextState) {
          if (v.ended || v.currentTime >= (v.duration || 0)) {
            v.currentTime = 0;
          }
          v.play().catch(() => {});
        } else {
          v.pause();
        }
      }
    });
  };

  const handleVideoEnded = () => {
    const allEnded = videoRefs.current.every(v => !v || v.ended);
    if (allEnded) {
      setIsPlayingAll(false);
    }
  };

  // Filter steps by searchQuery
  const filteredSteps = steps.filter(s =>
    s.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.step_id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="full-page-layout">
      <div className="sidebar-layout">
        {/* Left Sidebar: Controls & Step Selection */}
        <div className="list-sidebar">
          {/* Rollout Run Selector */}
          <div className="form-group">
            <label className="form-label">Select Rollout Run</label>
            <select
              value={selectedRollout}
              onChange={(e) => setSelectedRollout(e.target.value)}
              className="input-text"
              style={{ width: '100%' }}
            >
              {isLoadingRollouts ? (
                <option>Loading runs...</option>
              ) : rollouts.length === 0 ? (
                <option>No rollouts found</option>
              ) : (
                rollouts.map(r => (
                  <option key={r.id} value={r.id}>
                    {r.label} ({r.id})
                  </option>
                ))
              )}
            </select>
          </div>

          <hr className="separator" />

          {/* Search Rollout Steps */}
          <div className="form-group">
            <label className="form-label">Search Rollout Steps</label>
            <div className="flex-row-center gap-8" style={{ background: '#0f1015', border: '1px solid #1e293b', borderRadius: '6px', padding: '8px 12px' }}>
              <Search size={14} style={{ color: '#64748b' }} />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{ flexGrow: 1, border: 'none', padding: 0, background: 'none', fontSize: '13px', color: '#fff', outline: 'none' }}
                placeholder="Filter by epoch or step..."
              />
            </div>
          </div>

          {/* Steps List */}
          <div className="form-group flex-grow" style={{ overflow: 'hidden', marginTop: '4px' }}>
            <div className="flex-row-center justify-between" style={{ marginBottom: '8px' }}>
              <span className="form-label" style={{ fontSize: '10px' }}>Matched Steps ({filteredSteps.length})</span>
              <span style={{ fontSize: '10px', fontFamily: 'monospace', color: '#64748b' }}>{selectedRollout}</span>
            </div>

            <div className="panel-content-flex" style={{ overflowY: 'auto', height: 'calc(100% - 24px)', paddingRight: '2px' }}>
              {isLoadingSteps ? (
                <div style={{ fontSize: '11px', color: '#64748b', textAlign: 'center', padding: '20px' }}>Loading steps...</div>
              ) : filteredSteps.length === 0 ? (
                <div style={{ fontSize: '11px', color: '#64748b', textAlign: 'center', padding: '20px' }}>No steps found</div>
              ) : (
                filteredSteps.map((s) => (
                  <div
                    key={s.step_id}
                    onClick={() => setSelectedStep(s)}
                    className={`list-item-card ${selectedStep?.step_id === s.step_id ? 'selected' : ''}`}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                      <span style={{ fontSize: '13px', fontWeight: 600, color: selectedStep?.step_id === s.step_id ? '#fff' : '#f8fafc' }}>
                        {s.label}
                      </span>
                      <span
                        style={{
                          fontSize: '9px',
                          fontFamily: 'monospace',
                          color: 'var(--accent-cyan)',
                          background: 'var(--accent-cyan-dim)',
                          border: '1px solid rgba(6, 182, 212, 0.2)',
                          padding: '2px 6px',
                          borderRadius: '4px'
                        }}
                      >
                        16 Tracks
                      </span>
                    </div>
                    <div style={{ fontSize: '10px', color: '#64748b', fontFamily: 'monospace' }}>
                      {s.step_id}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Right Main Panel */}
        <div className="panel" style={{ height: '100%', overflow: 'hidden' }}>
          <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <h2 className="panel-title">
                <Film size={16} style={{ color: 'var(--accent-cyan)' }} />
                <span>{selectedRollout ? selectedRollout : 'Rollout Explorer'} {selectedStep ? `• ${selectedStep.label.toUpperCase()}` : ''}</span>
              </h2>
              <p className="panel-subtitle">
                {selectedStep ? (
                  <>Visualizing 16 candidate trajectory rollouts for step checkpoint <code style={{ color: 'var(--accent-cyan)', fontFamily: 'monospace' }}>{selectedStep.step_id}</code></>
                ) : (
                  'Select a rollout run and step from the sidebar to inspect candidate videos'
                )}
              </p>
            </div>

            <div className="flex-row-center gap-8">
              <button
                onClick={fetchRollouts}
                className="btn-phase btn-phase-action"
                style={{ padding: '6px 12px', fontSize: '11px' }}
                title="Refresh Rollouts"
              >
                <RefreshCw size={12} className={isLoadingRollouts ? 'animate-spin' : ''} />
                <span>Refresh</span>
              </button>

              <button
                onClick={togglePlayAll}
                disabled={!selectedStep || selectedStep.tracks.length === 0}
                className="btn-phase btn-phase-action"
                style={{
                  padding: '6px 14px',
                  fontSize: '11px',
                  background: isPlayingAll ? 'var(--accent-red-dim)' : 'var(--accent-green-dim)',
                  borderColor: isPlayingAll ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)',
                  color: isPlayingAll ? 'var(--accent-red)' : 'var(--accent-green)',
                  fontWeight: 600
                }}
              >
                {isPlayingAll ? <Pause size={12} /> : <Play size={12} />}
                <span>{isPlayingAll ? 'Pause All Tracks' : 'Play All Tracks'}</span>
              </button>
            </div>
          </div>

          <div className="panel-content" style={{ overflowY: 'auto', height: 'calc(100% - 60px)' }}>
            {selectedStep ? (
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(4, 1fr)',
                  gap: '12px',
                  paddingRight: '4px'
                }}
              >
                {selectedStep.tracks.map((track, idx) => (
                  <div
                    key={track.track_id}
                    style={{
                      background: 'rgba(255, 255, 255, 0.02)',
                      border: '1px solid rgba(255, 255, 255, 0.04)',
                      borderRadius: '8px',
                      padding: '10px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '8px'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '12px', fontWeight: 600, fontFamily: 'monospace', color: '#f8fafc' }}>
                        {track.track_id.toUpperCase()}
                      </span>
                      <span
                        style={{
                          fontSize: '9px',
                          fontWeight: 600,
                          fontFamily: 'monospace',
                          color: idx === 0 ? 'var(--accent-cyan)' : '#64748b',
                          background: idx === 0 ? 'var(--accent-cyan-dim)' : 'rgba(255, 255, 255, 0.02)',
                          border: idx === 0 ? '1px solid rgba(6, 182, 212, 0.2)' : '1px solid rgba(255, 255, 255, 0.04)',
                          padding: '2px 6px',
                          borderRadius: '4px'
                        }}
                      >
                        {idx === 0 ? 'Main Track' : `Candidate ${idx}`}
                      </span>
                    </div>

                    <div style={{ width: '100%', aspectRatio: '16/9', background: '#000', border: '1px solid #1a1a24', borderRadius: '6px', overflow: 'hidden' }}>
                      <video
                        ref={el => (videoRefs.current[idx] = el)}
                        src={track.url.startsWith('http') ? track.url : `${API_BASE}${track.url}`}
                        controls
                        muted
                        playsInline
                        onEnded={handleVideoEnded}
                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#64748b', gap: '8px' }}>
                <Layers size={32} />
                <span>Select a rollout step from the sidebar to inspect candidate trajectory videos</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
