import React, { useState } from 'react';
import { GitCommit, Plus, CheckCircle, Info, HelpCircle, ArrowRight, BookOpen } from 'lucide-react';

export default function SkillComposer() {
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [newSkillName, setNewSkillName] = useState('sequence_grasp_lift');
  
  // Rich dummy graph representing Tasks, Skills, and Low-Level Primitives
  const [skills, setSkills] = useState([
    // Higher-Level Tasks
    { id: 't1', name: 'Task: Pick up red cube', type: 'task', x: 80, y: 40, status: 'active' },
    { id: 't2', name: 'Task: Open desk drawer', type: 'task', x: 260, y: 40, status: 'standby' },
    
    // Middle-Level Distilled Skills
    { id: 's1', name: 'reach_cube_near', type: 'skill', x: 60, y: 130, status: 'distilled' },
    { id: 's2', name: 'pinch_cube_fingers', type: 'skill', x: 120, y: 200, status: 'distilled' },
    { id: 's3', name: 'lift_cube_vertical', type: 'skill', x: 180, y: 130, status: 'distilled' },
    { id: 's4', name: 'reach_drawer_handle', type: 'skill', x: 240, y: 200, status: 'fine_tuned' },
    { id: 's5', name: 'pull_drawer_slow', type: 'skill', x: 300, y: 130, status: 'fine_tuned' },
    
    // Low-Level Actuator Primitives
    { id: 'p1', name: 'joint_ctrl_torso_vla', type: 'primitive', x: 100, y: 300, status: 'active' },
    { id: 'p2', name: 'joint_ctrl_hand_ik', type: 'primitive', x: 200, y: 300, status: 'active' },
    { id: 'p3', name: 'joint_ctrl_wrist_align', type: 'primitive', x: 300, y: 300, status: 'standby' }
  ]);

  const [connections, setConnections] = useState([
    // Task to Skills
    { from: 't1', to: 's1' },
    { from: 't1', to: 's2' },
    { from: 't1', to: 's3' },
    { from: 't2', to: 's4' },
    { from: 't2', to: 's5' },
    
    // Skills to Primitives
    { from: 's1', to: 'p1' },
    { from: 's2', to: 'p2' },
    { from: 's3', to: 'p1' },
    { from: 's3', to: 'p3' },
    { from: 's4', to: 'p1' },
    { from: 's4', to: 'p3' },
    { from: 's5', to: 'p2' }
  ]);

  // Handle node selection for composition
  const handleNodeClick = (nodeId) => {
    // Primitives cannot be composed directly into higher-level macros, keep to tasks/skills
    const node = skills.find(n => n.id === nodeId);
    if (!node || node.type === 'primitive') return;

    if (selectedNodes.includes(nodeId)) {
      setSelectedNodes(prev => prev.filter(id => id !== nodeId));
    } else {
      setSelectedNodes(prev => [...prev, nodeId]);
    }
  };

  const handleCompose = (e) => {
    e.preventDefault();
    if (selectedNodes.length < 2 || !newSkillName.trim()) return;

    const newId = `c_${Date.now()}`;
    
    // Compute dynamic coordinates for the new composed node (placing it near center)
    const newSkillNode = {
      id: newId,
      name: newSkillName,
      type: 'composed',
      x: 180,
      y: 80,
      status: 'distilled'
    };

    // Chain connections from dependencies to new composed skill
    const newConns = selectedNodes.map(nodeId => ({
      from: nodeId,
      to: newId
    }));

    setSkills(prev => [...prev, newSkillNode]);
    setConnections(prev => [...prev, ...newConns]);
    setSelectedNodes([]);
    setNewSkillName('');
  };

  return (
    <div className="full-page-layout">
      <div className="sidebar-layout">
        {/* Left Control Column */}
        <div className="list-sidebar">
          <div className="telemetry-box" style={{ padding: '12px', background: 'rgba(6,182,212,0.02)' }}>
            <span className="form-label font-semibold text-cyan-400 flex items-center gap-4" style={{ fontSize: '10px' }}>
              <BookOpen size={12} />
              Skill Composition Manual
            </span>
            <p style={{ fontSize: '11px', color: '#94a3b8', margin: '6px 0 0 0', lineHeight: '1.4' }}>
              1. **Click** two or more middle-level skills (cyan nodes) on the graph.<br />
              2. Enter a unique macro name in the input below.<br />
              3. Click **Compose & Store** to compile the chained execution DAG.
            </p>
          </div>

          <form onSubmit={handleCompose} className="form-group flex-grow" style={{ gap: '14px', marginTop: '12px' }}>
            <div className="form-group">
              <label className="form-label">New Macro-Skill Name</label>
              <input
                type="text"
                value={newSkillName}
                onChange={(e) => setNewSkillName(e.target.value)}
                className="input-text"
                placeholder="e.g. chain_reach_grasp..."
                required
              />
            </div>

            <div className="form-group flex-grow">
              <label className="form-label">Composition Queue ({selectedNodes.length})</label>
              <div 
                style={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  gap: '6px', 
                  background: '#0f1015', 
                  padding: '10px', 
                  borderRadius: '6px', 
                  border: '1px solid #1e293b',
                  minHeight: '120px'
                }}
              >
                {selectedNodes.length === 0 ? (
                  <div style={{ color: '#475569', fontSize: '11px', padding: '10px', textAlign: 'center' }}>
                    Click Middle-Level Skills on the graph to build a chain
                  </div>
                ) : (
                  selectedNodes.map(nodeId => {
                    const node = skills.find(n => n.id === nodeId);
                    return (
                      <div 
                        key={nodeId}
                        style={{ 
                          fontSize: '11px', 
                          background: 'rgba(6, 182, 212, 0.05)', 
                          border: '1px solid rgba(6, 182, 212, 0.15)', 
                          padding: '6px 10px',
                          borderRadius: '4px',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center'
                        }}
                      >
                        <span style={{ color: '#fff', fontWeight: '500' }}>{node?.name}</span>
                        <span style={{ fontSize: '8px', color: 'var(--accent-cyan)', background: 'var(--accent-cyan-dim)', padding: '1px 4px', borderRadius: '2px', textTransform: 'uppercase' }}>
                          {node?.status}
                        </span>
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            <button
              type="submit"
              disabled={selectedNodes.length < 2}
              className="btn-adversary inactive"
              style={{ 
                marginTop: 'auto', 
                background: selectedNodes.length >= 2 ? 'var(--accent-cyan)' : '', 
                color: selectedNodes.length >= 2 ? '#000' : '',
                borderColor: selectedNodes.length >= 2 ? 'var(--accent-cyan)' : '',
                cursor: selectedNodes.length >= 2 ? 'pointer' : 'not-allowed',
                opacity: selectedNodes.length >= 2 ? 1 : 0.5
              }}
            >
              <Plus size={16} />
              Compose & Store Skill
            </button>
          </form>
        </div>

        {/* Right Interactive Graph Column */}
        <div className="panel h-full" style={{ padding: '20px' }}>
          <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h2 className="panel-title">
                <GitCommit className="text-cyan-400" size={18} />
                Skill Graph Library
              </h2>
              <p className="panel-subtitle">Interactive Tasks, Skills, and Primitives mapping</p>
            </div>
            
            {/* Color Legend */}
            <div style={{ display: 'flex', gap: '14px', fontSize: '10px' }}>
              <span className="flex-row-center gap-4"><span style={{ width: '6px', height: '6px', background: '#3b82f6', borderRadius: '50%' }} /> Task</span>
              <span className="flex-row-center gap-4"><span style={{ width: '6px', height: '6px', background: 'var(--accent-cyan)', borderRadius: '50%' }} /> Skill</span>
              <span className="flex-row-center gap-4"><span style={{ width: '6px', height: '6px', background: '#94a3b8', borderRadius: '50%' }} /> Primitive</span>
              <span className="flex-row-center gap-4"><span style={{ width: '6px', height: '6px', background: 'var(--accent-amber)', borderRadius: '50%' }} /> Composed</span>
            </div>
          </div>

          <div className="composer-canvas flex-grow">
            <svg viewBox="0 0 800 450" className="w-full h-full">
              {/* Draw Directed Connection Lines with markers */}
              <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#1e293b" />
                </marker>
                <marker id="arrow-active" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent-cyan)" />
                </marker>
              </defs>

              {connections.map((c, idx) => {
                const fromNode = skills.find(n => n.id === c.from);
                const toNode = skills.find(n => n.id === c.to);
                if (!fromNode || !toNode) return null;
                
                const isComposed = fromNode.type === 'composed' || toNode.type === 'composed';
                const isSelected = selectedNodes.includes(fromNode.id) && selectedNodes.includes(toNode.id);

                return (
                  <line
                    key={idx}
                    x1={fromNode.x * 2.2}
                    y1={fromNode.y * 1.1}
                    x2={toNode.x * 2.2}
                    y2={toNode.y * 1.1}
                    stroke={isSelected ? 'var(--accent-cyan)' : '#1e293b'}
                    strokeWidth={isSelected ? 2 : 1.5}
                    strokeDasharray={isComposed ? '4,4' : 'none'}
                    markerEnd={isSelected ? "url(#arrow-active)" : "url(#arrow)"}
                  />
                );
              })}

              {/* Draw Nodes */}
              {skills.map((node) => {
                const isSelected = selectedNodes.includes(node.id);
                
                // Color mapping
                let nodeColor = 'var(--accent-cyan)';
                if (node.type === 'task') nodeColor = '#3b82f6';
                if (node.type === 'primitive') nodeColor = '#94a3b8';
                if (node.type === 'composed') nodeColor = 'var(--accent-amber)';

                return (
                  <g 
                    key={node.id} 
                    transform={`translate(${node.x * 2.2}, ${node.y * 1.1})`}
                    onClick={() => handleNodeClick(node.id)}
                    className="cursor-pointer"
                  >
                    {/* Ring highlight when selected */}
                    {isSelected && (
                      <circle cx="0" cy="0" r="14" fill="none" stroke="var(--accent-cyan)" strokeWidth="2" className="animate-pulse" />
                    )}

                    {/* Node Circle */}
                    <circle 
                      cx="0" 
                      cy="0" 
                      r="9" 
                      fill={nodeColor}
                      stroke={isSelected ? '#fff' : 'none'}
                      strokeWidth="1.5"
                    />

                    {/* Status Badge overlay */}
                    {node.status === 'active' && (
                      <circle cx="0" cy="0" r="12" fill="none" stroke="var(--accent-green)" strokeWidth="1" strokeDasharray="2,2" className="animate-spin" style={{ transformOrigin: '0 0' }} />
                    )}

                    {/* Node Text Label */}
                    <text 
                      x="14" 
                      y="4" 
                      fill={isSelected ? '#fff' : '#cbd5e1'} 
                      fontSize="9.5" 
                      fontFamily="monospace"
                      fontWeight={isSelected ? 'bold' : '500'}
                    >
                      {node.name}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
