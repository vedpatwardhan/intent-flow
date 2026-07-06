import React, { useState, useEffect } from 'react';
import { Search, Play, CheckCircle, XCircle, BarChart2 } from 'lucide-react';

export default function TrajectoryExplorer() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTrajectory, setSelectedTrajectory] = useState(null);
  const [playingFrame, setPlayingFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Mock list of trajectories spanning Tasks and Skills
  const trajectories = [
    {
      id: 'traj_001',
      task: 'Pinch and lift cube',
      skill: 'pinch_cube',
      successRate: 0.92,
      difficulty: 'Easy',
      status: 'success',
      framesCount: 15,
      date: '2026-07-06 18:22'
    },
    {
      id: 'traj_002',
      task: 'Pinch and lift cube',
      skill: 'lift_cube',
      successRate: 0.88,
      difficulty: 'Easy',
      status: 'success',
      framesCount: 15,
      date: '2026-07-06 18:25'
    },
    {
      id: 'traj_003',
      task: 'Reach and grasp target',
      skill: 'reach_cube',
      successRate: 0.55,
      difficulty: 'Medium',
      status: 'failure',
      framesCount: 12,
      date: '2026-07-06 18:30'
    },
    {
      id: 'traj_004',
      task: 'Place cube in drawer',
      skill: 'place_cube',
      successRate: 0.22,
      difficulty: 'Hard',
      status: 'failure',
      framesCount: 18,
      date: '2026-07-06 18:35'
    },
    {
      id: 'traj_005',
      task: 'Open drawer handle',
      skill: 'reach_drawer',
      successRate: 0.95,
      difficulty: 'Easy',
      status: 'success',
      framesCount: 10,
      date: '2026-07-06 18:40'
    }
  ];

  // Filter trajectories based on query
  const filteredTrajectories = trajectories.filter(t => 
    t.task.toLowerCase().includes(searchQuery.toLowerCase()) ||
    t.skill.toLowerCase().includes(searchQuery.toLowerCase()) ||
    t.difficulty.toLowerCase().includes(searchQuery.toLowerCase())
  );

  useEffect(() => {
    if (filteredTrajectories.length > 0 && !selectedTrajectory) {
      setSelectedTrajectory(filteredTrajectories[0]);
    }
  }, [filteredTrajectories, selectedTrajectory]);

  // Handle mock video frame playback loop
  useEffect(() => {
    let timer;
    if (isPlaying && selectedTrajectory) {
      timer = setInterval(() => {
        setPlayingFrame(prev => (prev + 1) % selectedTrajectory.framesCount);
      }, 100);
    }
    return () => clearInterval(timer);
  }, [isPlaying, selectedTrajectory]);

  const selectTrajectory = (t) => {
    setSelectedTrajectory(t);
    setPlayingFrame(0);
    setIsPlaying(false);
  };

  const togglePlayback = () => {
    setIsPlaying(!isPlaying);
  };

  // Helper to draw a mock 2D frame animation based on the active frame index and status
  const renderPlaybackFrame = () => {
    if (!selectedTrajectory) return null;
    
    const maxFrames = selectedTrajectory.framesCount;
    const progress = playingFrame / maxFrames;
    const isSuccess = selectedTrajectory.status === 'success';

    // SVG coordinates for a simple robot hand moving toward a block
    const blockColor = isSuccess ? '#22c55e' : '#ef4444';
    // Starting position of hand
    const startX = 60;
    const startY = 160;
    // Target position of block
    const targetX = 240;
    const targetY = isSuccess && progress > 0.6 ? 160 - (progress - 0.6) * 100 : 160;

    // Kinematic interpolation of gripper
    const gripperX = startX + progress * (targetX - startX);
    const gripperY = startY + Math.sin(progress * Math.PI) * 40;

    return (
      <svg className="w-full h-full" style={{ background: '#09090d' }}>
        {/* Table top */}
        <line x1="20" y1="180" x2="300" y2="180" stroke="#1e293b" strokeWidth="3" />
        
        {/* Block */}
        <rect 
          x={targetX - 10} 
          y={targetY - 10} 
          width="20" 
          height="20" 
          fill={blockColor} 
          rx="2"
        />

        {/* Robot Arm Line */}
        <line 
          x1="20" 
          y1="40" 
          x2={(20 + gripperX)/2 + 20} 
          y2={(40 + gripperY)/2 - 20} 
          stroke="#06b6d4" 
          strokeWidth="6" 
        />
        <line 
          x1={(20 + gripperX)/2 + 20} 
          y1={(40 + gripperY)/2 - 20} 
          x2={gripperX} 
          y2={gripperY} 
          stroke="#0891b2" 
          strokeWidth="4" 
        />

        {/* Gripper */}
        <circle cx={gripperX} cy={gripperY} r="6" fill="#f8fafc" />
        <line x1={gripperX - 6} y1={gripperY} x2={gripperX - 6} y2={gripperY + 10} stroke="#f8fafc" strokeWidth="2" />
        <line x1={gripperX + 6} y1={gripperY} x2={gripperX + 6} y2={gripperY + 10} stroke="#f8fafc" strokeWidth="2" />

        {/* Frame index text */}
        <text x="12" y="300" fill="#64748b" fontSize="10" fontFamily="monospace">
          FRAME: {String(playingFrame).padStart(2, '0')} / {String(maxFrames).padStart(2, '0')} ({(progress * 100).toFixed(0)}%)
        </text>

        {/* Playback trace paths */}
        <path
          d={`M 60 160 Q 150 120 240 ${isSuccess ? 160 : 180}`}
          fill="none"
          stroke="rgba(6, 182, 212, 0.15)"
          strokeWidth="2"
          strokeDasharray="4,4"
        />
      </svg>
    );
  };

  return (
    <div className="full-page-layout">
      <div className="sidebar-layout">
        {/* Sidebar Left: Search and List */}
        <div className="list-sidebar">
          <div className="form-group" style={{ marginBottom: '10px' }}>
            <label className="form-label">Search Trajectories</label>
            <div className="flex-row-center gap-8 bg-neutral-900 border border-neutral-800 rounded px-3 py-1.5">
              <Search size={14} className="text-neutral-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input-text"
                style={{ flexGrow: 1, border: 'none', padding: 0, background: 'none', fontSize: '12px' }}
                placeholder="Filter by task or skill..."
              />
            </div>
          </div>

          <div className="flex flex-col gap-8 flex-grow" style={{ overflowY: 'auto' }}>
            <span className="form-label" style={{ fontSize: '9px' }}>Matched Results ({filteredTrajectories.length})</span>
            {filteredTrajectories.map((t) => (
              <div
                key={t.id}
                onClick={() => selectTrajectory(t)}
                className={`list-item-card ${selectedTrajectory?.id === t.id ? 'selected' : ''}`}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span className="font-semibold text-neutral-200" style={{ fontSize: '12px' }}>{t.id.toUpperCase()}</span>
                  <span 
                    style={{ 
                      fontSize: '9px', 
                      color: t.status === 'success' ? 'var(--accent-green)' : 'var(--accent-red)',
                      fontWeight: '700',
                      textTransform: 'uppercase'
                    }}
                  >
                    {t.status}
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '8px' }}>
                  {t.task}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#64748b' }}>
                  <span>Skill: <code className="text-cyan-400">{t.skill}</code></span>
                  <span>SR: {(t.successRate * 100).toFixed(0)}% ({t.difficulty})</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Main Panel Right: Trajectory Playback */}
        <div className="panel h-full" style={{ padding: '20px' }}>
          {selectedTrajectory ? (
            <div className="flex flex-col h-full gap-4">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h2 className="panel-title" style={{ fontSize: '18px' }}>
                    Trajectory Playback: {selectedTrajectory.id.toUpperCase()}
                  </h2>
                  <p className="panel-subtitle">
                    Task: {selectedTrajectory.task} • Associated Skill: <code className="text-cyan-400">{selectedTrajectory.skill}</code>
                  </p>
                </div>

                <button
                  onClick={togglePlayback}
                  className="btn-phase btn-phase-action"
                  style={{ padding: '8px 16px', background: isPlaying ? 'var(--accent-red-dim)' : 'var(--accent-green-dim)', borderColor: isPlaying ? 'var(--accent-red)' : 'var(--accent-green)', color: '#fff' }}
                >
                  <Play size={12} className={isPlaying ? 'hidden' : ''} />
                  {isPlaying ? 'Pause Playback' : 'Play Trajectory'}
                </button>
              </div>

              {/* Video Player Box */}
              <div className="viewport-frame flex-grow" style={{ maxHeight: '440px', width: '100%', maxWidth: 'none', aspectRatio: '16/9' }}>
                {renderPlaybackFrame()}
                
                {/* Overlay status tag */}
                <div 
                  style={{ 
                    position: 'absolute', 
                    top: '12px', 
                    right: '12px', 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '4px',
                    background: selectedTrajectory.status === 'success' ? 'var(--accent-green-dim)' : 'var(--accent-red-dim)',
                    border: '1px solid',
                    borderColor: selectedTrajectory.status === 'success' ? 'var(--accent-green)' : 'var(--accent-red)',
                    color: '#fff',
                    padding: '3px 8px',
                    borderRadius: '4px',
                    fontSize: '10px',
                    fontWeight: '700',
                    textTransform: 'uppercase'
                  }}
                >
                  {selectedTrajectory.status === 'success' ? <CheckCircle size={10} /> : <XCircle size={10} />}
                  {selectedTrajectory.status}
                </div>
              </div>

              {/* Success metrics */}
              <div className="grid-phases" style={{ marginTop: 'auto' }}>
                <div className="telemetry-box" style={{ padding: '10px 14px' }}>
                  <span className="form-label" style={{ fontSize: '9px' }}>Checkpoints Level Success Rate</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
                    <BarChart2 size={16} className="text-cyan-400" />
                    <span style={{ fontSize: '18px', fontWeight: '700', color: '#fff' }}>
                      {(selectedTrajectory.successRate * 100).toFixed(0)}%
                    </span>
                    <span 
                      style={{ 
                        fontSize: '10px', 
                        color: selectedTrajectory.difficulty === 'Easy' ? 'var(--accent-green)' : selectedTrajectory.difficulty === 'Medium' ? 'var(--accent-amber)' : 'var(--accent-red)',
                        background: selectedTrajectory.difficulty === 'Easy' ? 'var(--accent-green-dim)' : selectedTrajectory.difficulty === 'Medium' ? 'var(--accent-amber-dim)' : 'var(--accent-red-dim)',
                        padding: '1px 6px',
                        borderRadius: '3px',
                        fontWeight: '600'
                      }}
                    >
                      {selectedTrajectory.difficulty} Task
                    </span>
                  </div>
                </div>

                <div className="telemetry-box" style={{ padding: '10px 14px' }}>
                  <span className="form-label" style={{ fontSize: '9px' }}>Metadata</span>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '6px' }}>
                    Recorded Time: {selectedTrajectory.date}<br />
                    Sequence Length: {selectedTrajectory.framesCount} timesteps
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', justify: 'center', height: '100%', color: '#64748b' }}>
              Select a trajectory to begin playback
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
