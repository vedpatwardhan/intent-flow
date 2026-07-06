import React from 'react';
import { Activity, Thermometer, Radio } from 'lucide-react';

export default function TelemetryPanel({ energy, energyHistory, tactileGrid, joints }) {
  const safeTactile = tactileGrid || [[0, 0], [0, 0]];
  const safeJoints = joints || { positions: [0, 0, 0, 0], torques: [0, 0, 0, 0] };
  const history = energyHistory || [];

  const width = 320;
  const height = 80;
  const maxEnergy = Math.max(...history, 1.0);
  
  const points = history.map((val, idx) => {
    const x = (idx / Math.max(history.length - 1, 1)) * width;
    const y = height - (val / maxEnergy) * height;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="panel h-full">
      <div className="panel-header">
        <h2 className="panel-title">
          <Activity className="text-cyan-400" size={18} />
          Telemetry Systems
        </h2>
        <p className="panel-subtitle">Live energy model compatibility & safety metrics</p>
      </div>

      <div className="panel-content panel-content-flex">
        {/* EBM energy chart */}
        <div className="telemetry-box">
          <div className="telemetry-row">
            <span className="form-label">JEPA EBM compatibility energy</span>
            <span 
              className="font-mono" 
              style={{ 
                color: energy > 0.6 ? 'var(--accent-red)' : 'var(--accent-cyan)',
                fontWeight: 'bold',
                fontSize: '13px'
              }}
            >
              {energy.toFixed(4)}
            </span>
          </div>

          <div className="chart-container">
            {history.length > 1 && (
              <svg className="w-full h-full" style={{ position: 'absolute', top: 0, left: 0 }}>
                <polyline
                  fill="url(#energy-grad)"
                  stroke="none"
                  points={`0,${height} ${points} ${width},${height}`}
                />
                <polyline
                  fill="none"
                  stroke={energy > 0.6 ? 'var(--accent-red)' : 'var(--accent-cyan)'}
                  strokeWidth="2"
                  points={points}
                />
                <defs>
                  <linearGradient id="energy-grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={energy > 0.6 ? 'rgba(239,68,68,0.2)' : 'rgba(6,182,212,0.2)'} />
                    <stop offset="100%" stopColor="rgba(0,0,0,0)" />
                  </linearGradient>
                </defs>
              </svg>
            )}
            {energy > 0.6 && (
              <div 
                style={{ 
                  position: 'absolute', 
                  top: '6px', 
                  left: '8px', 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '4px', 
                  background: 'rgba(239,68,68,0.2)', 
                  border: '1px solid rgba(239,68,68,0.3)', 
                  color: 'var(--accent-red)',
                  fontSize: '9px',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  fontWeight: '600'
                }}
              >
                <Thermometer size={10} /> Critical Anomaly
              </div>
            )}
          </div>
        </div>

        {/* Tactile grid */}
        <div className="telemetry-box">
          <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Radio size={12} className="text-green-400" />
            Fingertip Tactile Grid (MuJoCo)
          </span>
          <div className="grid-fingertips">
            {safeTactile.map((row, rIdx) => 
              row.map((val, cIdx) => (
                <div 
                  key={`${rIdx}-${cIdx}`}
                  className="tactile-pad"
                  style={{
                    backgroundColor: `rgba(16, 185, 129, ${0.03 + val * 0.7})`,
                    borderColor: `rgba(16, 185, 129, ${0.08 + val * 0.4})`,
                    color: val > 0.4 ? '#000' : '#94a3b8',
                    fontWeight: val > 0.4 ? '600' : '400'
                  }}
                >
                  {val.toFixed(2)}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Joint torques */}
        <div className="telemetry-box" style={{ flexGrow: 1 }}>
          <span className="form-label">Actuator Torque Levels</span>
          <div className="joint-list" style={{ marginTop: '4px' }}>
            {safeJoints.torques.map((t, idx) => {
              const absT = Math.abs(t);
              const percentage = Math.min((absT / 20) * 100, 100);
              return (
                <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8' }}>
                    <span>Joint {idx + 1} ({joints.positions[idx]?.toFixed(2)} rad)</span>
                    <span className="font-mono">{t.toFixed(1)} Nm</span>
                  </div>
                  <div className="joint-bar-container">
                    <div 
                      className="joint-bar-fill"
                      style={{ 
                        width: `${percentage}%`,
                        backgroundColor: t > 15 ? 'var(--accent-red)' : 'var(--accent-cyan)'
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
