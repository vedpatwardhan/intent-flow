import React from 'react';
import { Eye, Grid, Move, Target } from 'lucide-react';

const HeatmapCanvas = ({ dataMatrix, color }) => {
  const canvasRef = React.useRef(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 120, 120);

    if (!dataMatrix || dataMatrix.length === 0) return;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 14;
    tempCanvas.height = 14;
    const tempCtx = tempCanvas.getContext('2d');
    const imgData = tempCtx.createImageData(14, 14);

    const clamp = (num, min, max) => Math.min(Math.max(num, min), max);
    const getJetRGB = (v) => {
      const r = clamp(Math.min(4 * v - 1.5, -4 * v + 4.5), 0, 1) * 255;
      const g = clamp(Math.min(4 * v - 0.5, -4 * v + 3.5), 0, 1) * 255;
      const b = clamp(Math.min(4 * v + 0.5, -4 * v + 2.5), 0, 1) * 255;
      return [Math.round(r), Math.round(g), Math.round(b)];
    };

    for (let r = 0; r < 14; r++) {
      for (let c = 0; c < 14; c++) {
        const val = dataMatrix[r]?.[c] || 0.0;
        const idx = (r * 14 + c) * 4;
        if (val > 0.05) {
          const [red, green, blue] = getJetRGB(val);
          imgData.data[idx] = red;
          imgData.data[idx + 1] = green;
          imgData.data[idx + 2] = blue;
          imgData.data[idx + 3] = Math.round(val * 0.45 * 255);
        } else {
          imgData.data[idx] = 0;
          imgData.data[idx + 1] = 0;
          imgData.data[idx + 2] = 0;
          imgData.data[idx + 3] = 0;
        }
      }
    }
    tempCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tempCanvas, 0, 0, 14, 14, 0, 0, 120, 120);
  }, [dataMatrix, color]);

  return (
    <canvas
      ref={canvasRef}
      width={120}
      height={120}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        borderRadius: '4px'
      }}
    />
  );
};

const FeatureCard = ({ title, data, icon: Icon, color, renderType }) => {
  if (!data || (Array.isArray(data) && data.length === 0)) {
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

  const isImage = typeof data === 'string' && data.startsWith('data:image');
  const isHeatmap = renderType === 'heatmap';
  const isTracks = renderType === 'tracks';
  const isPointCloud = renderType === 'pointcloud';

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
        fontFamily: 'monospace',
        overflow: 'hidden',
        position: 'relative'
      }}>
        {isImage ? (
          <img
            src={data}
            alt={title}
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }}
          />
        ) : isHeatmap ? (
          <HeatmapCanvas dataMatrix={data} color={color} />
        ) : isTracks ? (
          <svg viewBox="0 0 120 120" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
            <defs>
              <marker id="arrow-critical" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 2 1 L 8 5 L 2 9" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </marker>
            </defs>
            {data.slice(0, 20).map((pt, idx) => {
              if (pt.length < 4) return null;
              const x1 = pt[0] * 120;
              const y1 = pt[1] * 120;
              const x2 = pt[2] * 120;
              const y2 = pt[3] * 120;
              const scale = 2.0;
              const dx = (x2 - x1) * scale;
              const dy = (y2 - y1) * scale;
              const targetX = x1 + dx;
              const targetY = y1 + dy;

              return (
                <g key={idx}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={targetX}
                    y2={targetY}
                    stroke={color}
                    strokeWidth="2"
                    opacity="0.8"
                    markerEnd="url(#arrow-critical)"
                  />
                  <circle cx={x1} cy={y1} r="2" fill={color} opacity="0.6" />
                </g>
              );
            })}
          </svg>
        ) : isPointCloud ? (
          <svg viewBox="0 0 120 120" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', background: '#020204' }}>
            {data.slice(0, 200).map((pt, idx) => {
              const x = ((pt[0] + 1.0) / 2.0) * 120;
              const y = ((1.0 - pt[1]) / 2.0) * 120;
              const r = Math.round(pt[3] * 255);
              const g = Math.round(pt[4] * 255);
              const b = Math.round(pt[5] * 255);
              const pointColor = `rgb(${r},${g},${b})`;

              if (x < 0 || x > 120 || y < 0 || y > 120) return null;

              return (
                <circle
                  key={idx}
                  cx={x}
                  cy={y}
                  r="1.2"
                  fill={pointColor}
                  opacity="0.9"
                />
              );
            })}
          </svg>
        ) : (
          <div style={{ textAlign: 'center' }}>
            <div>Isolated Features</div>
            <div style={{ fontSize: '8px', opacity: 0.7 }}>
              {Array.isArray(data) ? `${data.length} items` : 'Active'}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default function CriticalSubspace({ frame, isolatedFeatures }) {
  return (
    <div className="panel" style={{ borderColor: 'var(--accent-green)', display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header" style={{ marginBottom: '8px' }}>
        <h2 className="panel-title">
          <Eye className="text-green-400" size={16} />
          Critical Subspace
        </h2>
        <p className="panel-subtitle">Task-isolated latent features</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px', flex: 1 }}>
        <FeatureCard
          title="DINOv3 Spatial Attention"
          data={isolatedFeatures?.dino_subspace}
          icon={Eye}
          color="var(--accent-amber)"
          renderType="heatmap"
        />
        <FeatureCard
          title="VGGT Trajectory Tracks"
          data={isolatedFeatures?.vggt_local}
          icon={Move}
          color="var(--accent-cyan)"
          renderType="tracks"
        />
        <FeatureCard
          title="PointNeXt 3D Cloud"
          data={isolatedFeatures?.pointnext_isolated}
          icon={Grid}
          color="var(--accent-purple)"
          renderType="pointcloud"
        />
        <FeatureCard
          title="Tactile Features"
          data={null}
          icon={Target}
          color="var(--accent-red)"
        />
      </div>

      <div style={{
        marginTop: '8px',
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
