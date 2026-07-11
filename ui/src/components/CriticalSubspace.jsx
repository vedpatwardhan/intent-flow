import React from 'react';
import { Eye, Grid, Move, Target } from 'lucide-react';

const clamp = (num, min, max) => Math.min(Math.max(num, min), max);

const getJetRGB = (v) => {
  const r = clamp(Math.min(4 * v - 1.5, -4 * v + 4.5), 0, 1) * 255;
  const g = clamp(Math.min(4 * v - 0.5, -4 * v + 3.5), 0, 1) * 255;
  const b = clamp(Math.min(4 * v + 0.5, -4 * v + 2.5), 0, 1) * 255;
  return [Math.round(r), Math.round(g), Math.round(b)];
};

const getCSSVariableValue = (variableName) => {
  if (typeof window === 'undefined') return '#000';
  const val = window.getComputedStyle(document.documentElement).getPropertyValue(variableName).trim();
  return val || '#000';
};

const IsolatedFeatureCard = ({ title, frame, featureData, maskData, icon: Icon, color, renderType }) => {
  const canvasRef = React.useRef(null);
  const prevDataRef = React.useRef({ featureData: null, maskData: null });

  const hasDataChanged = React.useMemo(() => {
    const prev = prevDataRef.current;
    const featureChanged = JSON.stringify(prev.featureData) !== JSON.stringify(featureData);
    const maskChanged = JSON.stringify(prev.maskData) !== JSON.stringify(maskData);
    return featureChanged || maskChanged;
  }, [featureData, maskData]);

  React.useEffect(() => {
    prevDataRef.current = { featureData, maskData };
  }, [featureData, maskData]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = 240;
    const height = 240;

    const resolvedColor = color.startsWith('var(')
      ? getCSSVariableValue(color.slice(4, -1))
      : color;

    if (!frame) {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, width, height);
      return;
    }

    const img = new Image();
    img.onload = () => {
      // Clear and paint background black atomically before drawing
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, width, height);

      // Draw original frame
      ctx.drawImage(img, 0, 0, width, height);

      // Apply DINO mask to isolate regions
      if (maskData && maskData.length > 0) {
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = width;
        maskCanvas.height = height;
        const maskCtx = maskCanvas.getContext('2d');

        maskCtx.fillStyle = '#000';
        maskCtx.fillRect(0, 0, width, height);

        const isHighRes = maskData.length === 224;
        const resSize = isHighRes ? 224 : 14;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = resSize;
        tempCanvas.height = resSize;
        const tempCtx = tempCanvas.getContext('2d');
        const imgData = tempCtx.createImageData(resSize, resSize);

        for (let r = 0; r < resSize; r++) {
          for (let c = 0; c < resSize; c++) {
            const val = maskData[r]?.[c] || 0.0;
            const idx = (r * resSize + c) * 4;
            if (val > 0.05) {
              imgData.data[idx] = 255;
              imgData.data[idx + 1] = 255;
              imgData.data[idx + 2] = 255;
              imgData.data[idx + 3] = 255;
            } else {
              imgData.data[idx] = 0;
              imgData.data[idx + 1] = 0;
              imgData.data[idx + 2] = 0;
              imgData.data[idx + 3] = 0;
            }
          }
        }
        tempCtx.putImageData(imgData, 0, 0);
        maskCtx.imageSmoothingEnabled = true;
        maskCtx.drawImage(tempCanvas, 0, 0, resSize, resSize, 0, 0, width, height);

        const maskImageData = maskCtx.getImageData(0, 0, width, height);
        const mainImageData = ctx.getImageData(0, 0, width, height);

        for (let i = 0; i < maskImageData.data.length; i += 4) {
          if (maskImageData.data[i] === 0) {
            mainImageData.data[i] = 0;
            mainImageData.data[i + 1] = 0;
            mainImageData.data[i + 2] = 0;
          }
        }
        ctx.putImageData(mainImageData, 0, 0);
      }

      // Overlay specific feature
      if (renderType === 'heatmap' && featureData && featureData.length > 0) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 14;
        tempCanvas.height = 14;
        const tempCtx = tempCanvas.getContext('2d');
        const imgData = tempCtx.createImageData(14, 14);

        for (let r = 0; r < 14; r++) {
          for (let c = 0; c < 14; c++) {
            const val = featureData[r]?.[c] || 0.0;
            const idx = (r * 14 + c) * 4;
            if (val > 0.05) {
              const [red, green, blue] = getJetRGB(val);
              imgData.data[idx] = red;
              imgData.data[idx + 1] = green;
              imgData.data[idx + 2] = blue;
              imgData.data[idx + 3] = Math.round(val * 0.5 * 255);
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
        ctx.drawImage(tempCanvas, 0, 0, 14, 14, 0, 0, width, height);
      }

      if (renderType === 'tracks' && featureData && featureData.length > 0) {
        ctx.strokeStyle = resolvedColor;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;

        featureData.slice(0, 30).forEach(pt => {
          if (pt.length < 4) return;
          const x1 = pt[0] * width;
          const y1 = pt[1] * height;
          const x2 = pt[2] * width;
          const y2 = pt[3] * height;
          const scale = 2.0;
          const dx = (x2 - x1) * scale;
          const dy = (y2 - y1) * scale;
          const targetX = x1 + dx;
          const targetY = y1 + dy;

          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(targetX, targetY);
          ctx.stroke();

          ctx.fillStyle = resolvedColor;
          ctx.beginPath();
          ctx.arc(x1, y1, 3, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.globalAlpha = 1.0;
      }

      if (renderType === 'pointcloud' && featureData && featureData.length > 0) {
        // Use same 3D camera transformation as encoder diagnostics
        const eye = [0.1, 0.1, 2.0];
        const eyeNorm = Math.sqrt(eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]);
        const forward = [-eye[0] / eyeNorm, -eye[1] / eyeNorm, -eye[2] / eyeNorm];
        const up_initial = [0.0, 1.0, 0.0];

        const right_raw = [-forward[2], 0.0, forward[0]];
        const rightNorm = Math.sqrt(right_raw[0] * right_raw[0] + right_raw[2] * right_raw[2]);
        const right = [right_raw[0] / rightNorm, 0.0, right_raw[2] / rightNorm];

        const up = [
          right[1] * forward[2] - right[2] * forward[1],
          right[2] * forward[0] - right[0] * forward[2],
          right[0] * forward[1] - right[1] * forward[0]
        ];

        const pointsWithDepth = featureData.map((pt) => {
          const px = pt[0];
          const py = pt[1];
          const pz = pt[2];

          const x_cam = px * right[0] + py * right[1] + pz * right[2];
          const y_cam = px * up[0] + py * up[1] + pz * up[2];
          const z_cam = px * forward[0] + py * forward[1] + pz * forward[2] + 2.0;

          const focal = 1.8;
          const x_proj = (x_cam * focal) / z_cam;
          const y_proj = (y_cam * focal) / z_cam;

          const screenX = ((x_proj + 1.0) / 2.0) * width;
          const screenY = ((1.0 - y_proj) / 2.0) * height;

          const r = Math.round(pt[3] * 255);
          const g = Math.round(pt[4] * 255);
          const b = Math.round(pt[5] * 255);

          return { screenX, screenY, z_cam, r, g, b };
        });

        pointsWithDepth.sort((a, b) => b.z_cam - a.z_cam);

        ctx.fillStyle = '#020204';
        ctx.fillRect(0, 0, width, height);

        pointsWithDepth.forEach((p) => {
          if (p.screenX < 0 || p.screenX > width || p.screenY < 0 || p.screenY > height) return;
          ctx.fillStyle = `rgb(${p.r},${p.g},${p.b})`;
          ctx.beginPath();
          ctx.arc(p.screenX, p.screenY, 2, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    };
    img.src = frame;
  }, [frame, hasDataChanged, renderType, color]);

  if (!featureData || (Array.isArray(featureData) && featureData.length === 0)) {
    return (
      <div style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid var(--border-glass)',
        borderRadius: '6px',
        padding: '8px',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        height: '270px'
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
          minHeight: '200px'
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
      gap: '6px',
      height: '270px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color, fontWeight: 600 }}>
        <Icon size={12} />
        <span>{title}</span>
      </div>
      <div style={{
        position: 'relative',
        width: '240px',
        height: '240px',
        background: '#000',
        borderRadius: '4px',
        overflow: 'hidden'
      }}>
        <canvas
          ref={canvasRef}
          width={240}
          height={240}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
};

export default function CriticalSubspace({ frame, isolatedFeatures }) {
  return (
    <div className="panel" style={{ borderColor: 'var(--accent-green)', display: 'flex', flexDirection: 'column', height: 'fit-content', gap: '12px' }}>
      <div className="panel-header" style={{ marginBottom: '8px' }}>
        <h2 className="panel-title">
          <Eye className="text-green-400" size={16} />
          Critical Subspace
        </h2>
        <p className="panel-subtitle">Task-isolated latent features</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px' }}>
        <IsolatedFeatureCard
          title="DINOv3 Spatial Attention"
          frame={frame}
          featureData={isolatedFeatures?.dino_subspace}
          maskData={isolatedFeatures?.combined_mask_224}
          icon={Eye}
          color="var(--accent-amber)"
          renderType="heatmap"
        />
        <IsolatedFeatureCard
          title="VGGT Trajectory Tracks"
          frame={frame}
          featureData={isolatedFeatures?.vggt_local}
          maskData={isolatedFeatures?.combined_mask_224}
          icon={Move}
          color="var(--accent-cyan)"
          renderType="tracks"
        />
        <IsolatedFeatureCard
          title="PointNeXt 3D Cloud"
          frame={frame}
          featureData={isolatedFeatures?.pointnext_isolated}
          maskData={isolatedFeatures?.combined_mask_224}
          icon={Grid}
          color="var(--accent-purple)"
          renderType="pointcloud"
        />
        <IsolatedFeatureCard
          title="Tactile Features"
          frame={frame}
          featureData={isolatedFeatures?.tactile_active}
          maskData={isolatedFeatures?.combined_mask_224}
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
