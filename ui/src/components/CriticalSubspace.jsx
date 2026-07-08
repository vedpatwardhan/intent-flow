import React from 'react';
import { Eye, Grid, Move, Activity } from 'lucide-react';

const FeatureCard = ({ title, data, icon: Icon, color }) => {
  if (!data || data.length === 0) {
    return (
      <div style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid var(--border-glass)',
        borderRadius: '6px',
        padding: '8px',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color: '#64748b' }}>
          <Icon size={12} />
          <span>{title}</span>
        </div>
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'rgba(0, 0, 0, 0.3)',
          borderRadius: '4px',
          fontSize: '9px',
          color: '#475569',
          minHeight: '80px'
        }}>
          Waiting for annotations...
        </div>
      </div>
    );
  }

  return (
    <div style={{
      background: 'rgba(255, 255, 255, 0.02)',
      border: `1px solid ${color}`,
      borderRadius: '6px',
      padding: '8px',
      display: 'flex',
      flexDirection: 'column',
      gap: '6px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color, fontWeight: 600 }}>
        <Icon size={12} />
        <span>{title}</span>
      </div>
      <div style={{
        flex: 1,
        background: '#000',
        borderRadius: '4px',
        minHeight: '80px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '9px',
        color: '#94a3b8',
        fontFamily: 'monospace'
      }}>
        {/* Placeholder for actual feature visualization */}
        <div style={{ textAlign: 'center' }}>
          <div>Isolated Features</div>
          <div style={{ fontSize: '8px', opacity: 0.7 }}>
            {Array.isArray(data) ? `${data.length} tokens` : 'Active'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default function CriticalSubspace({ frame, isolatedFeatures }) {
  return (
    <div className="panel" style={{ borderColor: 'var(--accent-green)' }}>
      <div className="panel-header" style={{ marginBottom: '10px' }}>
        <h2 className="panel-title">
          <Eye className="text-green-400" size={16} />
          Critical Subspace
        </h2>
        <p className="panel-subtitle">Task-isolated latent features</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px', height: '120px' }}>
        <FeatureCard
          title="DINOv3"
          data={isolatedFeatures?.dino_subspace}
          icon={Eye}
          color="var(--accent-amber)"
        />
        <FeatureCard
          title="PointNeXt"
          data={isolatedFeatures?.pointnext_isolated}
          icon={Grid}
          color="var(--accent-purple)"
        />
        <FeatureCard
          title="VGGT Tracks"
          data={isolatedFeatures?.vggt_local}
          icon={Move}
          color="var(--accent-cyan)"
        />
        <FeatureCard
          title="Tactile"
          data={isolatedFeatures?.tactile_active}
          icon={Activity}
          color="var(--accent-green)"
        />
      </div>

      <div style={{
        marginTop: '6px',
        padding: '6px',
        background: 'rgba(34, 197, 94, 0.05)',
        border: '1px solid rgba(34, 197, 94, 0.2)',
        borderRadius: '4px',
        fontSize: '8px',
        color: '#86efac',
        fontFamily: 'monospace',
        textAlign: 'center'
      }}>
        <span style={{ fontWeight: 600 }}>
          {isolatedFeatures ? 'ACTIVE' : 'INACTIVE'}
        </span>
        <span style={{ opacity: 0.8, marginLeft: '6px' }}>
          {isolatedFeatures ? 'Mask Applied' : 'No Annotations'}
        </span>
      </div>
    </div>
  );
}
