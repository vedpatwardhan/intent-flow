import React, { useState } from 'react';
import { GitCommit, CheckCircle, RefreshCw } from 'lucide-react';

export default function GnnLibrary({ skills }) {
  const [hoveredNode, setHoveredNode] = useState(null);

  const width = 320;
  const height = 130;

  const skillList = skills || [
    { id: '1', name: 'reach_cube', type: 'internalized', x: 60, y: 65, active: true },
    { id: '2', name: 'pinch_cube', type: 'internalized', x: 160, y: 35, active: false },
    { id: '3', name: 'lift_cube', type: 'externalized', x: 260, y: 65, active: false }
  ];

  const connections = [
    { from: '1', to: '2' },
    { from: '2', to: '3' },
    { from: '1', to: '3' }
  ];

  return (
    <div className="panel" style={{ padding: '16px', opacity: 0.45, filter: 'grayscale(30%)', position: 'relative' }}>
      <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <div>
          <h2 className="panel-title" style={{ fontSize: '15px' }}>
            <GitCommit className="text-cyan-400" size={16} />
            GNN Skill Library
          </h2>
          <p className="panel-subtitle" style={{ fontSize: '10px' }}>Autonomous dynamic skill evolution</p>
        </div>
        <div 
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '4px', 
            background: 'rgba(255, 255, 255, 0.05)', 
            border: '1px solid var(--border-glass)', 
            color: '#64748b',
            fontSize: '9px',
            padding: '2px 6px',
            borderRadius: '4px',
            textTransform: 'uppercase',
            letterSpacing: '0.05em'
          }}
        >
          VISION / CONCEPT
        </div>
      </div>

      {/* SVG Skills Graph */}
      <div className="gnn-canvas" style={{ marginBottom: '12px' }}>
        <svg className="w-full h-full">
          {connections.map((c, idx) => {
            const fromNode = skillList.find(n => n.id === c.from);
            const toNode = skillList.find(n => n.id === c.to);
            if (!fromNode || !toNode) return null;
            return (
              <line
                key={idx}
                x1={fromNode.x}
                y1={fromNode.y}
                x2={toNode.x}
                y2={toNode.y}
                stroke={fromNode.active && toNode.active ? 'var(--accent-cyan)' : '#1e293b'}
                strokeWidth={fromNode.active && toNode.active ? 2 : 1}
                strokeDasharray={fromNode.type === 'externalized' ? '4,4' : 'none'}
              />
            );
          })}

          {skillList.map((skill) => (
            <g 
              key={skill.id}
              className="cursor-pointer"
              onMouseEnter={() => setHoveredNode(skill)}
              onMouseLeave={() => setHoveredNode(null)}
            >
              <circle
                cx={skill.x}
                cy={skill.y}
                r={skill.active ? 8 : 6}
                fill="none"
                stroke={skill.active ? 'var(--accent-cyan)' : '#334155'}
                strokeWidth={2}
                className={skill.active ? 'animate-pulse' : ''}
              />
              <circle
                cx={skill.x}
                cy={skill.y}
                r={3}
                fill={skill.type === 'internalized' ? 'var(--accent-cyan)' : 'var(--accent-red)'}
              />
            </g>
          ))}
        </svg>

        {hoveredNode && (
          <div 
            style={{
              position: 'absolute',
              bottom: '6px',
              left: '6px',
              right: '6px',
              background: 'rgba(10, 10, 15, 0.95)',
              border: '1px solid #1e293b',
              padding: '6px 10px',
              borderRadius: '6px',
              fontSize: '10px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              backdropFilter: 'blur(4px)'
            }}
          >
            <div>
              <span style={{ fontWeight: '600', color: '#f8fafc', display: 'block' }}>{hoveredNode.name}</span>
              <span style={{ color: '#64748b', fontSize: '9px', textTransform: 'capitalize' }}>{hoveredNode.type} Skill</span>
            </div>
            <div>
              {hoveredNode.active ? (
                <span style={{ color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '2px' }}>
                  <CheckCircle size={8} /> Active
                </span>
              ) : (
                <span style={{ color: '#475569' }}>Standby</span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Skills list table */}
      <div className="gnn-table">
        {skillList.map(skill => (
          <div 
            key={skill.id} 
            className={`gnn-row ${skill.active ? 'active' : ''}`}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span 
                style={{ 
                  width: '6px', 
                  height: '6px', 
                  borderRadius: '50%',
                  backgroundColor: skill.type === 'internalized' ? 'var(--accent-cyan)' : 'var(--accent-red)'
                }} 
              />
              <span>{skill.name}</span>
            </div>
            <span style={{ opacity: 0.7, textTransform: 'capitalize', fontSize: '9px' }}>{skill.type}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
