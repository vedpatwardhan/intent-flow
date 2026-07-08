import React, { useRef, useState, useEffect } from 'react';
import { Crosshair, Navigation, Target, RotateCcw } from 'lucide-react';

export default function UnifiedWorkspace({ frames, activeCam, onInteraction }) {
  const canvasRef = useRef(null);
  
  const [activeTool, setActiveTool] = useState('segment');
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentPos, setCurrentPos] = useState({ x: 0, y: 0 });
  
  const [annotations, setAnnotations] = useState({
    segments: [],
    vectors: [],
    crops: []
  });

  const tools = [
    { id: 'segment', label: 'SAM Segment', icon: Crosshair, color: 'var(--accent-cyan)' },
    { id: 'vector', label: 'Motion Vector', icon: Navigation, color: 'var(--accent-amber)' },
    { id: 'crop', label: 'Target Crop', icon: Target, color: 'var(--accent-green)' }
  ];

  const activeFrame = frames?.[activeCam];

  const getMousePos = (e) => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);

    return {
      x: ((clientX - rect.left) / rect.width) * 224,
      y: ((clientY - rect.top) / rect.height) * 224
    };
  };

  const handleMouseDown = (e) => {
    const pos = getMousePos(e);
    setIsDrawing(true);
    setStartPos(pos);
    setCurrentPos(pos);

    if (activeTool === 'segment') {
      onInteraction({
        type: 'original_click',
        x: pos.x,
        y: pos.y
      });
      setAnnotations(prev => ({
        ...prev,
        segments: [...prev.segments, { x: pos.x, y: pos.y }]
      }));
    }
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    setCurrentPos(getMousePos(e));
  };

  const handleMouseUp = () => {
    if (!isDrawing) return;
    setIsDrawing(false);

    if (activeTool === 'vector') {
      const vector = {
        start: [startPos.x, startPos.y],
        end: [currentPos.x, currentPos.y]
      };
      const distance = Math.hypot(vector.end[0] - vector.start[0], vector.end[1] - vector.start[1]);
      if (distance > 5) {
        setAnnotations(prev => ({
          ...prev,
          vectors: [...prev.vectors, vector]
        }));
        onInteraction({
          type: 'add_vector',
          coordinates: vector
        });
      }
    } else if (activeTool === 'crop') {
      const crop = {
        x: Math.min(startPos.x, currentPos.x),
        y: Math.min(startPos.y, currentPos.y),
        width: Math.abs(currentPos.x - startPos.x),
        height: Math.abs(currentPos.y - startPos.y)
      };
      if (crop.width > 5 && crop.height > 5) {
        setAnnotations(prev => ({
          ...prev,
          crops: [...prev.crops, crop]
        }));
        onInteraction({
          type: 'add_crop',
          coordinates: crop
        });
      }
    }
  };

  const clearAnnotations = () => {
    setAnnotations({ segments: [], vectors: [], crops: [] });
    onInteraction({ type: 'clear_annotations' });
  };

  const drawArrow = (ctx, fromx, fromy, tox, toy, color) => {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(fromx, fromy);
    ctx.lineTo(tox, toy);
    ctx.stroke();

    const angle = Math.atan2(toy - fromy, tox - fromx);
    ctx.beginPath();
    ctx.moveTo(tox, toy);
    ctx.lineTo(tox - 12 * Math.cos(angle - Math.PI / 6), toy - 12 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(tox - 12 * Math.cos(angle + Math.PI / 6), toy - 12 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw existing annotations
    annotations.segments.forEach(seg => {
      ctx.strokeStyle = 'var(--accent-cyan)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(seg.x, seg.y, 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = 'rgba(6, 182, 212, 0.3)';
      ctx.fill();
    });

    annotations.vectors.forEach(vec => {
      drawArrow(ctx, vec.start[0], vec.start[1], vec.end[0], vec.end[1], 'var(--accent-amber)');
    });

    annotations.crops.forEach(crop => {
      ctx.strokeStyle = 'var(--accent-green)';
      ctx.lineWidth = 2;
      ctx.strokeRect(crop.x, crop.y, crop.width, crop.height);
      ctx.fillStyle = 'rgba(34, 197, 94, 0.15)';
      ctx.fillRect(crop.x, crop.y, crop.width, crop.height);
    });

    // Draw current drawing state
    if (isDrawing) {
      if (activeTool === 'vector') {
        drawArrow(ctx, startPos.x, startPos.y, currentPos.x, currentPos.y, 'rgba(6, 182, 212, 0.8)');
      } else if (activeTool === 'crop') {
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)';
        ctx.lineWidth = 2;
        ctx.strokeRect(
          startPos.x,
          startPos.y,
          currentPos.x - startPos.x,
          currentPos.y - startPos.y
        );
      }
    }
  }, [annotations, isDrawing, startPos, currentPos, activeTool]);

  return (
    <div className="panel" style={{ borderColor: 'var(--accent-cyan)' }}>
      <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <div>
          <h2 className="panel-title">
            <Crosshair className="text-cyan-400" size={18} />
            Unified Workspace
          </h2>
          <p className="panel-subtitle">Single viewport with multi-tool annotation</p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <select
            value={activeTool}
            onChange={(e) => setActiveTool(e.target.value)}
            style={{
              padding: '6px 10px',
              background: 'var(--accent-cyan-dim)',
              border: '1px solid var(--accent-cyan)',
              borderRadius: '4px',
              color: 'var(--accent-cyan)',
              fontSize: '11px',
              cursor: 'pointer'
            }}
          >
            {tools.map(tool => (
              <option key={tool.id} value={tool.id}>{tool.label}</option>
            ))}
          </select>
          <button
            onClick={clearAnnotations}
            className="btn-phase btn-phase-action"
            style={{ padding: '6px 10px' }}
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      <div style={{ position: 'relative', width: '100%', maxWidth: '320px', margin: '0 auto' }}>
        <div style={{ position: 'relative', width: '100%', aspectRatio: '1', background: '#000', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-glass)' }}>
          {activeFrame ? (
            <img
              src={activeFrame}
              alt="camera"
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }}
            />
          ) : (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', fontSize: '11px' }}>
              Waiting...
            </div>
          )}
          <canvas
            ref={canvasRef}
            width={224}
            height={224}
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              cursor: 'crosshair'
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onTouchStart={handleMouseDown}
            onTouchMove={handleMouseMove}
            onTouchEnd={handleMouseUp}
          />
        </div>
        
        <div style={{ marginTop: '8px', display: 'flex', gap: '12px', fontSize: '9px', color: '#64748b', fontFamily: 'monospace' }}>
          {tools.map(tool => {
            const Icon = tool.icon;
            return (
              <div key={tool.id} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: tool.color }} />
                <Icon size={10} />
                <span>{tool.label}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
