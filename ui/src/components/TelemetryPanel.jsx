import React from 'react';
import { Activity, Thermometer, Radio } from 'lucide-react';

export default function TelemetryPanel({ energy, energyHistory, tactileGrid, joints }) {
  // Safe default arrays
  const safeTactile = tactileGrid || [[0, 0], [0, 0]];
  const safeJoints = joints || { positions: [0, 0, 0, 0], torques: [0, 0, 0, 0] };
  const history = energyHistory || [];

  // Generate SVG path for energy history
  const width = 300;
  const height = 80;
  const maxEnergy = Math.max(...history, 1.0);
  
  const points = history.map((val, idx) => {
    const x = (idx / Math.max(history.length - 1, 1)) * width;
    const y = height - (val / maxEnergy) * height;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="glass-panel flex flex-col gap-4 h-full overflow-y-auto">
      <div>
        <h2 className="font-semibold text-lg text-neutral-100 flex items-center gap-2">
          <Activity className="text-cyan-400" size={20} />
          Telemetry Systems
        </h2>
        <p className="text-xs text-neutral-500">Live energy model compatibility & safety metrics</p>
      </div>

      {/* JEPA Energy compatibility graph */}
      <div className="flex flex-col gap-2 p-3 bg-neutral-900/50 border border-neutral-800/60 rounded-lg">
        <div className="flex justify-between items-center">
          <span className="text-xs text-neutral-400 font-semibold uppercase tracking-wider">JEPA EBM compatibility energy</span>
          <span className={`text-sm font-mono font-bold ${energy > 0.6 ? 'text-red-400' : 'text-cyan-400'}`}>
            {energy.toFixed(4)}
          </span>
        </div>

        <div className="bg-black rounded border border-neutral-900 h-20 relative overflow-hidden">
          {history.length > 1 && (
            <svg className="w-full h-full">
              {/* Fill area */}
              <polyline
                fill="url(#energy-grad)"
                stroke="none"
                points={`0,${height} ${points} ${width},${height}`}
              />
              {/* Line */}
              <polyline
                fill="none"
                stroke={energy > 0.6 ? '#ef4444' : '#06b6d4'}
                strokeWidth="2"
                points={points}
              />
              <defs>
                <linearGradient id="energy-grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={energy > 0.6 ? 'rgba(239,68,68,0.3)' : 'rgba(6,182,212,0.3)'} />
                  <stop offset="100%" stopColor="rgba(0,0,0,0)" />
                </linearGradient>
              </defs>
            </svg>
          )}
          {energy > 0.6 && (
            <div className="absolute top-1 left-2 flex items-center gap-1 bg-red-950/60 border border-red-500/20 text-[10px] text-red-400 px-1 rounded uppercase tracking-wider animate-pulse">
              <Thermometer size={10} /> Critical Anomaly
            </div>
          )}
        </div>
      </div>

      {/* Tactile grid matrix */}
      <div className="flex flex-col gap-2 p-3 bg-neutral-900/50 border border-neutral-800/60 rounded-lg">
        <span className="text-xs text-neutral-400 font-semibold uppercase tracking-wider flex items-center gap-1">
          <Radio size={14} className="text-green-400" />
          Fingertip Tactile Grid (MuJoCo)
        </span>
        <div className="grid grid-cols-2 gap-2 mt-1">
          {safeTactile.map((row, rIdx) => 
            row.map((val, cIdx) => (
              <div 
                key={`${rIdx}-${cIdx}`}
                className="h-10 rounded border transition-all duration-150 flex items-center justify-center font-mono text-xs"
                style={{
                  backgroundColor: `rgba(34, 197, 94, ${0.05 + val * 0.8})`,
                  borderColor: `rgba(34, 197, 94, ${0.1 + val * 0.5})`,
                  color: val > 0.4 ? '#000' : '#888',
                  fontWeight: val > 0.4 ? 'bold' : 'normal'
                }}
              >
                {val.toFixed(2)}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Robot joints load */}
      <div className="flex flex-col gap-2 p-3 bg-neutral-900/50 border border-neutral-800/60 rounded-lg flex-grow">
        <span className="text-xs text-neutral-400 font-semibold uppercase tracking-wider">Actuator Torque Levels</span>
        <div className="flex flex-col gap-2 mt-1">
          {safeJoints.torques.map((t, idx) => {
            const absT = Math.abs(t);
            const percentage = Math.min((absT / 20) * 100, 100);
            return (
              <div key={idx} className="flex flex-col gap-0.5">
                <div className="flex justify-between text-[11px] text-neutral-400">
                  <span>Joint {idx + 1} ({joints.positions[idx]?.toFixed(2)} rad)</span>
                  <span className="font-mono">{t.toFixed(1)} Nm</span>
                </div>
                <div className="h-1.5 w-full bg-neutral-950 rounded-full overflow-hidden border border-neutral-900">
                  <div 
                    className={`h-full rounded-full transition-all duration-150 ${t > 15 ? 'bg-red-500' : 'bg-cyan-500'}`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
