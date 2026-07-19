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

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = 480;
    const height = 480;

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
        const rows = featureData.length;
        const cols = featureData[0]?.length || 0;
        if (cols > 0) {
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = cols;
          tempCanvas.height = rows;
          const tempCtx = tempCanvas.getContext('2d');
          const imgData = tempCtx.createImageData(cols, rows);

          // Find max value in matrix to normalize raw metric magnitudes (like VGGT)
          let maxVal = 1e-8;
          for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
              if (featureData[r]?.[c] > maxVal) {
                maxVal = featureData[r][c];
              }
            }
          }

          for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
              let val = featureData[r]?.[c] || 0.0;
              // If this is the high-res 224x224 motion field, normalize to [0, 1] range for visual contrast
              if (rows === 224) {
                val = val / maxVal;
              }
              const idx = (r * cols + c) * 4;
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
          ctx.drawImage(tempCanvas, 0, 0, cols, rows, 0, 0, width, height);
        }
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
  }, [frame, featureData, maskData, renderType, color]);

  const getPlaceholderText = () => {
    if (maskData !== null) {
      return "Waiting for annotations...";
    }
    if (renderType === 'tracks') {
      return "Move objects in simulation to trace tracks...";
    }
    return "Waiting for GPU features...";
  };

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
        height: '300px'
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
        }}>
          {getPlaceholderText()}
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
      height: '300px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color, fontWeight: 600 }}>
        <Icon size={12} />
        <span>{title}</span>
      </div>
      <div style={{
        flex: 1,
        position: 'relative',
        background: '#000',
        borderRadius: '4px',
        overflow: 'hidden'
      }}>
        <canvas
          ref={canvasRef}
          width={480}
          height={480}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            objectFit: 'contain'
          }}
        />
      </div>
    </div>
  );
};

export default function CriticalSubspace({ frame, isolatedFeatures, dinoAttn, clipSim, vggTracks }) {
  return (
    <div className="panel" style={{
      borderColor: 'var(--accent-green)',
      display: 'flex',
      flexDirection: 'column',
      height: 'fit-content',
      gap: '12px',
      paddingRight: '16px',
    }}>
      <div className="panel-header" style={{ marginBottom: '4px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 className="panel-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Eye className="text-indigo-400" size={16} />
            Representations Dashboard
          </h2>
          <p className="panel-subtitle">Vertical comparison of global and task-isolated features</p>
        </div>
        <div style={{
          padding: '4px 8px',
          background: isolatedFeatures ? 'rgba(34, 197, 94, 0.05)' : 'rgba(148, 163, 184, 0.05)',
          border: isolatedFeatures ? '1px solid rgba(34, 197, 94, 0.2)' : '1px solid rgba(148, 163, 184, 0.2)',
          borderRadius: '4px',
          fontSize: '9px',
          color: isolatedFeatures ? '#86efac' : '#94a3b8',
          fontFamily: 'monospace'
        }}>
          {isolatedFeatures ? 'ACTIVE • Mask Applied' : 'INACTIVE • No Annotations'}
        </div>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '12px',
        flexGrow: 1
      }}>
        {/* Column 1: DINOv3 Spatial Attention */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <IsolatedFeatureCard
            title="Global DINOv3 Attention"
            frame={frame}
            featureData={dinoAttn}
            maskData={null}
            icon={Eye}
            color="var(--accent-amber)"
            renderType="heatmap"
          />
          <IsolatedFeatureCard
            title="Local DINOv3 Subspace"
            frame={frame}
            featureData={isolatedFeatures?.dino_subspace}
            maskData={isolatedFeatures?.combined_mask_224}
            icon={Eye}
            color="var(--accent-amber)"
            renderType="heatmap"
          />
        </div>

        {/* Column 2: VGGT Motion Field */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <IsolatedFeatureCard
            title="Global VGGT Motion"
            frame={frame}
            featureData={vggTracks}
            maskData={null}
            icon={Move}
            color="var(--accent-cyan)"
            renderType="heatmap"
          />
          <IsolatedFeatureCard
            title="Local VGGT Subspace"
            frame={frame}
            featureData={isolatedFeatures?.motion_field_subspace}
            maskData={isolatedFeatures?.combined_mask_224}
            icon={Move}
            color="var(--accent-cyan)"
            renderType="heatmap"
          />
        </div>

        {/* Column 3: CLIP & Tactile */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <IsolatedFeatureCard
            title="Global CLIP Similarity"
            frame={frame}
            featureData={clipSim}
            maskData={null}
            icon={Target}
            color="var(--accent-red)"
            renderType="heatmap"
          />
          <IsolatedFeatureCard
            title="Local Tactile Features"
            frame={frame}
            featureData={isolatedFeatures?.tactile_active}
            maskData={isolatedFeatures?.combined_mask_224}
            icon={Target}
            color="var(--accent-red)"
          />
        </div>
      </div>
    </div>
  );
}
