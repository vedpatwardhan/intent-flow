import React, { useState } from 'react';
import { GitCommit, Plus, CheckCircle, Info } from 'lucide-react';

export default function SkillComposer() {
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [newSkillName, setNewSkillName] = useState('sequence_pickup_lift');
  const [skills, setSkills] = useState([
    { id: 't1', name: 'Task: Pick up red cube', type: 'task', x: 80, y: 50 },
    { id: 't2', name: 'Task: Fasten drawer zip-tie', type: 'task', x: 280, y: 50 },
    
    { id: 's1', name: 'reach_cube', type: 'high_level', x: 80, y: 150 },
    { id: 's2', name: 'pinch_cube', type: 'high_level', x: 80, y: 250 },
    { id: 's3', name: 'lift_cube', type: 'high_level', x: 180, y: 200 },
    { id: 's4', name: 'reach_drawer', type: 'high_level', x: 280, y: 150 },
    { id: 's5', name: 'fasten_zip', type: 'high_level', x: 280, y: 250 },
    
    { id: 'p1', name: 'joint_command_torso', type: 'primitive', x: 180, y: 350 },
    { id: 'p2', name: 'joint_command_hand', type: 'primitive', x: 280, y: 350 }
  ]);

  const [connections, setConnections] = useState([
    { from: 't1', to: 's1' },
    { from: 't1', to: 's2' },
    { from: 't1', to: 's3' },
    { from: 't2', to: 's4' },
    { from: 't2', to: 's5' },
    { from: 's1', to: 'p1' },
    { from: 's2', to: 'p2' },
    { from: 's3', to: 'p1' },
    { from: 's4', to: 'p1' },
    { from: 's5', to: 'p2' }
  ]);

  // Click handler to select nodes for chaining
  const handleNodeClick = (nodeId) => {
    if (selectedNodes.includes(nodeId)) {
      setSelectedNodes(prev => prev.filter(id => id !== nodeId));
    } else {
      setSelectedNodes(prev => [...prev, nodeId]);
    }
  };

  const handleCompose = (e) => {
    e.preventDefault();
    if (selectedNodes.length < 2 || !newSkillName.trim()) return;

    // Create a new skill node
    const newId = `c_${Date.now()}`;
    const newSkillNode = {
      id: newId,
      name: newSkillName,
      type: 'composed',
      x: 180,
      y: 100 // Center top composed row
    };

    // Create connections from selected nodes to the new node
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
        {/* Sidebar Left: Composing Panel */}
        <div className="list-sidebar">
          <div>
            <h3 className="form-label" style={{ marginBottom: '4px' }}>Skill Composability</h3>
            <p style={{ fontSize: '11px', color: '#64748b', margin: 0 }}>
              Select two or more skill nodes from the GNN graph to chain them together.
            </p>
          </div>

          <form onSubmit={handleCompose} className="form-group flex-grow" style={{ gap: '12px', marginTop: '10px' }}>
            <div className="form-group">
              <label className="form-label">New Skill Name</label>
              <input
                type="text"
                value={newSkillName}
                onChange={(e) => setNewSkillName(e.target.value)}
                className="input-text"
                placeholder="e.g. sequence_pickup..."
                required
              />
            </div>

            <div className="form-group flex-grow">
              <label className="form-label">Selected Chain Queue ({selectedNodes.length})</label>
              <div 
                style={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  gap: '6px', 
                  background: '#0f1015', 
                  padding: '10px', 
                  borderRadius: '6px', 
                  border: '1px solid #1e293b',
                  minHeight: '100px'
                }}
              >
                {selectedNodes.length === 0 ? (
                  <span style={{ color: '#475569', fontSize: '11px' }}>Click nodes on the graph to add to queue</span>
                ) : (
                  selectedNodes.map(nodeId => {
                    const node = skills.find(n => n.id === nodeId);
                    return (
                      <div 
                        key={nodeId}
                        style={{ 
                          fontSize: '11px', 
                          background: 'rgba(255,255,255,0.02)', 
                          border: '1px solid rgba(255,255,255,0.05)', 
                          padding: '4px 8px',
                          borderRadius: '4px',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center'
                        }}
                      >
                        <span style={{ color: '#e2e8f0' }}>{node?.name}</span>
                        <span style={{ fontSize: '9px', color: '#64748b', textTransform: 'capitalize' }}>{node?.type}</span>
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

        {/* Main Panel Right: Interactive GNN Graph Canvas */}
        <div className="panel h-full" style={{ padding: '20px' }}>
          <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h2 className="panel-title">
                <GitCommit className="text-cyan-400" size={18} />
                Skill Graph Library
              </h2>
              <p className="panel-subtitle">Interactive Tasks, Skills, and Primitives mapping</p>
            </div>
            
            {/* Color keys */}
            <div style={{ display: 'flex', gap: '12px', fontSize: '10px' }}>
              <span className="flex-row-center gap-4"><span style={{ width: '6px', height: '6px', background: '#3b82f6', borderRadius: '50%' }} /> Task</span>
              <span className="flex-row-center gap-4"><span style={{ width: '6px', height: '6px', background: 'var(--accent-cyan)', borderRadius: '50%' }} /> Skill</span>
              <span className="flex-row-center gap-4"><span style={{ width: '6px', height: '6px', background: '#e2e8f0', borderRadius: '50%' }} /> Primitive</span>
              <span className="flex-row-center gap-4"><span style={{ width: '6px', height: '6px', background: 'var(--accent-amber)', borderRadius: '50%' }} /> Composed</span>
            </div>
          </div>

          <div className="composer-canvas flex-grow">
            <svg className="w-full h-full">
              {/* Draw connections */}
              {connections.map((c, idx) => {
                const fromNode = skills.find(n => n.id === c.from);
                const toNode = skills.find(n => n.id === c.to);
                if (!fromNode || !toNode) return null;
                return (
                  <line
                    key={idx}
                    x1={fromNode.x * 2.2}
                    y1={fromNode.y * 1.1}
                    x2={toNode.x * 2.2}
                    y2={toNode.y * 1.1}
                    stroke="#1e293b"
                    strokeWidth="1.5"
                  />
                );
              })}

              {/* Draw Interactive Nodes */}
              {skills.map((node) => {
                const isSelected = selectedNodes.includes(node.id);
                
                // Color mapping
                let nodeColor = 'var(--accent-cyan)'; // high_level
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

                    {/* Label */}
                    <text 
                      x="14" 
                      y="4" 
                      fill={isSelected ? '#fff' : '#94a3b8'} 
                      fontSize="9" 
                      fontFamily="monospace"
                      fontWeight={isSelected ? 'bold' : 'normal'}
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
