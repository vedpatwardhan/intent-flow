import React, { useState } from 'react';
import { GitCommit, Plus, CheckCircle, RefreshCw } from 'lucide-react';

export default function GnnLibrary({ skills }) {
  const [hoveredNode, setHoveredNode] = useState(null);

  // SVG dimensions for the skill network map
  const width = 300;
  const height = 150;

  // Safe default skills if empty
  const skillList = skills || [
    { id: '1', name: 'reach_drawer', type: 'internalized', x: 50, y: 75, active: true },
    { id: '2', name: 'pinch_cube', type: 'internalized', x: 150, y: 40, active: false },
    { id: '3', name: 'lift_cube', type: 'externalized', x: 250, y: 75, active: false }
  ];

  // Draw transition connections
  const connections = [
    { from: '1', to: '2' },
    { from: '2', to: '3' },
    { from: '1', to: '3' }
  ];

  return (
    <div className="glass-panel flex flex-col gap-4">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="font-semibold text-lg text-neutral-100 flex items-center gap-2">
            <GitCommit className="text-cyan-400" size={20} />
            GNN Skill Library
          </h2>
          <p className="text-xs text-neutral-500">Autonomous GNN dynamic skill evolution</p>
        </div>
        <div className="flex items-center gap-1 text-[10px] bg-cyan-950/60 border border-cyan-500/20 text-cyan-400 px-2 py-0.5 rounded">
          <RefreshCw size={8} className="animate-spin" /> Evolving
        </div>
      </div>

      {/* SVG Skills Graph */}
      <div className="bg-black/80 rounded-lg border border-neutral-900 h-40 relative overflow-hidden">
        <svg className="w-full h-full">
          {/* Draw connecting lines */}
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
                stroke={fromNode.active && toNode.active ? '#06b6d4' : '#222'}
                strokeWidth={fromNode.active && toNode.active ? 2 : 1}
                strokeDasharray={fromNode.type === 'externalized' ? '4,4' : 'none'}
              />
            );
          })}

          {/* Draw nodes */}
          {skillList.map((skill) => (
            <g 
              key={skill.id}
              className="cursor-pointer"
              onMouseEnter={() => setHoveredNode(skill)}
              onMouseLeave={() => setHoveredNode(null)}
            >
              {/* Outer halo */}
              <circle
                cx={skill.x}
                cy={skill.y}
                r={skill.active ? 10 : 7}
                fill="none"
                stroke={skill.active ? '#06b6d4' : '#333'}
                strokeWidth={2}
                className={skill.active ? 'animate-pulse' : ''}
              />
              {/* Inner core */}
              <circle
                cx={skill.x}
                cy={skill.y}
                r={4}
                fill={skill.type === 'internalized' ? '#06b6d4' : '#ef4444'}
              />
            </g>
          ))}
        </svg>

        {/* Hover info tooltip */}
        {hoveredNode && (
          <div className="absolute bottom-2 left-2 right-2 bg-neutral-950/90 border border-neutral-850 p-2 rounded text-[11px] backdrop-blur flex justify-between items-center">
            <div>
              <span className="font-semibold text-neutral-200">{hoveredNode.name}</span>
              <span className="text-[10px] text-neutral-500 block capitalize">{hoveredNode.type} Skill</span>
            </div>
            <div className="flex items-center gap-1 text-[10px]">
              {hoveredNode.active ? (
                <span className="text-cyan-400 flex items-center gap-0.5">
                  <CheckCircle size={10} /> Active
                </span>
              ) : (
                <span className="text-neutral-600">Standby</span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Skills list table */}
      <div className="flex flex-col gap-1.5 max-h-36 overflow-y-auto">
        {skillList.map(skill => (
          <div 
            key={skill.id} 
            className={`flex justify-between items-center p-2 rounded text-xs border ${
              skill.active 
                ? 'bg-cyan-500/5 border-cyan-500/20 text-cyan-200 font-semibold' 
                : 'bg-neutral-900/30 border-neutral-850 text-neutral-400'
            }`}
          >
            <div className="flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${skill.type === 'internalized' ? 'bg-cyan-400' : 'bg-red-400'}`} />
              <span>{skill.name}</span>
            </div>
            <span className="text-[10px] opacity-70 capitalize">{skill.type}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
