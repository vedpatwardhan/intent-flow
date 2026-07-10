import React, { useRef, useState, useEffect } from 'react';
import { Crosshair, Navigation, Target, RotateCcw, Maximize2, Minimize2, Trash2, MousePointer } from 'lucide-react';

export default function UnifiedWorkspace({ frames, activeCam, onInteraction, samMask }) {
  const canvasRef = useRef(null);

  const [activeTool, setActiveTool] = useState('segment');
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentPos, setCurrentPos] = useState({ x: 0, y: 0 });
  const [isMaximized, setIsMaximized] = useState(false);
  const [selectedAnnotation, setSelectedAnnotation] = useState(null); // { type: 'segments'|'vectors'|'crops', index: number }

  const [annotations, setAnnotations] = useState({
    segments: [],
    vectors: [],
    crops: []
  });

  const tools = [
    { id: 'select', label: 'Select / Edit', icon: MousePointer, color: 'var(--accent-primary, #6366f1)' },
    { id: 'segment', label: 'SAM Segment', icon: Crosshair, color: 'var(--accent-cyan)' },
    { id: 'vector', label: 'Motion Vector', icon: Navigation, color: 'var(--accent-amber)' },
    { id: 'crop', label: 'Target Crop', icon: Target, color: 'var(--accent-green)' }
  ];

  const activeFrame = frames?.[activeCam];

  // Helper to map and synchronize annotations to the 224x224 space for model processing
  const syncWithBackend = (nextAnnotations) => {
    const scaled = {
      segments: nextAnnotations.segments.map(seg => ({
        x: Math.round((seg.x / 480) * 224),
        y: Math.round((seg.y / 480) * 224)
      })),
      crops: nextAnnotations.crops.map(crop => ({
        x: Math.round((crop.x / 480) * 224),
        y: Math.round((crop.y / 480) * 224),
        width: Math.round((crop.width / 480) * 224),
        height: Math.round((crop.height / 480) * 224)
      })),
      vectors: nextAnnotations.vectors.map(vec => ({
        start: [Math.round((vec.start[0] / 480) * 224), Math.round((vec.start[1] / 480) * 224)],
        end: [Math.round((vec.end[0] / 480) * 224), Math.round((vec.end[1] / 480) * 224)]
      }))
    };
    onInteraction({ type: 'sync_annotations', annotations: scaled });
  };

  // Helper to snap coordinates to segment points, crop centers, and crop corners in 480x480 space
  const getSnappedPos = (pos, currentAnnotations, threshold = 30) => {
    let bestSnap = { ...pos };
    let minDistance = threshold;

    // Check segment points
    currentAnnotations.segments.forEach(seg => {
      const dist = Math.hypot(pos.x - seg.x, pos.y - seg.y);
      if (dist < minDistance) {
        minDistance = dist;
        bestSnap = { x: seg.x, y: seg.y };
      }
    });

    // Check crop boxes
    currentAnnotations.crops.forEach(crop => {
      // Center
      const centerX = crop.x + crop.width / 2;
      const centerY = crop.y + crop.height / 2;
      const distCenter = Math.hypot(pos.x - centerX, pos.y - centerY);
      if (distCenter < minDistance) {
        minDistance = distCenter;
        bestSnap = { x: centerX, y: centerY };
      }

      // Corners
      const corners = [
        { x: crop.x, y: crop.y },
        { x: crop.x + crop.width, y: crop.y },
        { x: crop.x, y: crop.y + crop.height },
        { x: crop.x + crop.width, y: crop.y + crop.height }
      ];
      corners.forEach(corner => {
        const distCorner = Math.hypot(pos.x - corner.x, pos.y - corner.y);
        if (distCorner < minDistance) {
          minDistance = distCorner;
          bestSnap = { x: corner.x, y: corner.y };
        }
      });
    });

    return bestSnap;
  };

  // Find annotation at mouse position for selection in 480x480 space
  const findAnnotationAtPosition = (pos, threshold = 30) => {
    let found = null;
    let minDistance = threshold;

    // Check segments
    annotations.segments.forEach((seg, idx) => {
      const dist = Math.hypot(pos.x - seg.x, pos.y - seg.y);
      if (dist < minDistance) {
        minDistance = dist;
        found = { type: 'segments', index: idx };
      }
    });

    // Check crops (inside or edge)
    annotations.crops.forEach((crop, idx) => {
      const inside = pos.x >= crop.x && pos.x <= crop.x + crop.width &&
        pos.y >= crop.y && pos.y <= crop.y + crop.height;
      if (inside) {
        const centerX = crop.x + crop.width / 2;
        const centerY = crop.y + crop.height / 2;
        const distCenter = Math.hypot(pos.x - centerX, pos.y - centerY);
        const effectiveDist = distCenter * 0.5; // Prioritize inside click
        if (effectiveDist < minDistance) {
          minDistance = effectiveDist;
          found = { type: 'crops', index: idx };
        }
      }
    });

    // Check vectors (distance to line segment)
    const distToSegment = (p, v, w) => {
      const l2 = Math.hypot(v[0] - w[0], v[1] - w[1]) ** 2;
      if (l2 === 0) return Math.hypot(p.x - v[0], p.y - v[1]);
      let t = ((p.x - v[0]) * (w[0] - v[0]) + (p.y - v[1]) * (w[1] - v[1])) / l2;
      t = Math.max(0, Math.min(1, t));
      return Math.hypot(p.x - (v[0] + t * (w[0] - v[0])), p.y - (v[1] + t * (w[1] - v[1])));
    };

    annotations.vectors.forEach((vec, idx) => {
      const dist = distToSegment(pos, vec.start, vec.end);
      if (dist < minDistance) {
        minDistance = dist;
        found = { type: 'vectors', index: idx };
      }
    });

    return found;
  };

  const getMousePos = (e) => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);

    return {
      x: ((clientX - rect.left) / rect.width) * 480,
      y: ((clientY - rect.top) / rect.height) * 480
    };
  };

  const handleMouseDown = (e) => {
    const rawPos = getMousePos(e);

    if (activeTool === 'select') {
      const found = findAnnotationAtPosition(rawPos);
      setSelectedAnnotation(found);
      return;
    }

    let pos = rawPos;
    if (activeTool === 'vector') {
      pos = getSnappedPos(rawPos, annotations);
    }

    setIsDrawing(true);
    setStartPos(pos);
    setCurrentPos(pos);

    if (activeTool === 'segment') {
      onInteraction({
        type: 'original_click',
        x: Math.round((pos.x / 480) * 224),
        y: Math.round((pos.y / 480) * 224)
      });
      setAnnotations(prev => {
        const next = {
          ...prev,
          segments: [...prev.segments, { x: pos.x, y: pos.y }]
        };
        syncWithBackend(next);
        return next;
      });
    }
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    const rawPos = getMousePos(e);
    let pos = rawPos;
    if (activeTool === 'vector') {
      pos = getSnappedPos(rawPos, annotations);
    }
    setCurrentPos(pos);
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
      if (distance > 10) {
        setAnnotations(prev => {
          const next = {
            ...prev,
            vectors: [...prev.vectors, vector]
          };
          syncWithBackend(next);
          return next;
        });
      }
    } else if (activeTool === 'crop') {
      const crop = {
        x: Math.min(startPos.x, currentPos.x),
        y: Math.min(startPos.y, currentPos.y),
        width: Math.abs(currentPos.x - startPos.x),
        height: Math.abs(currentPos.y - startPos.y)
      };
      if (crop.width > 10 && crop.height > 10) {
        setAnnotations(prev => {
          const next = {
            ...prev,
            crops: [...prev.crops, crop]
          };
          syncWithBackend(next);
          return next;
        });
      }
    }
  };

  const clearAnnotations = () => {
    setAnnotations({ segments: [], vectors: [], crops: [] });
    setSelectedAnnotation(null);
    onInteraction({ type: 'clear_annotations' });
  };

  const deleteSelected = () => {
    if (!selectedAnnotation) return;
    const { type, index } = selectedAnnotation;

    setAnnotations(prev => {
      const updatedList = [...prev[type]];
      updatedList.splice(index, 1);
      const next = {
        ...prev,
        [type]: updatedList
      };
      syncWithBackend(next);
      return next;
    });
    setSelectedAnnotation(null);
  };

  // Keyboard listener for deleting annotations
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedAnnotation) {
        deleteSelected();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedAnnotation]);

  const drawArrow = (ctx, fromx, fromy, tox, toy, color, isDashed = false) => {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 4;
    if (isDashed) {
      ctx.setLineDash([8, 8]);
    } else {
      ctx.setLineDash([]);
    }
    ctx.beginPath();
    ctx.moveTo(fromx, fromy);
    ctx.lineTo(tox, toy);
    ctx.stroke();
    ctx.setLineDash([]);

    const angle = Math.atan2(toy - fromy, tox - fromx);
    ctx.beginPath();
    ctx.moveTo(tox, toy);
    ctx.lineTo(tox - 18 * Math.cos(angle - Math.PI / 6), toy - 18 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(tox - 18 * Math.cos(angle + Math.PI / 6), toy - 18 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw existing annotations
    annotations.segments.forEach((seg, idx) => {
      const isSelected = selectedAnnotation?.type === 'segments' && selectedAnnotation?.index === idx;
      ctx.strokeStyle = isSelected ? '#ef4444' : 'var(--accent-cyan)';
      ctx.lineWidth = isSelected ? 4 : 3;
      ctx.beginPath();
      ctx.arc(seg.x, seg.y, 10, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = isSelected ? 'rgba(239, 68, 68, 0.4)' : 'rgba(6, 182, 212, 0.3)';
      ctx.fill();
    });

    annotations.vectors.forEach((vec, idx) => {
      const isSelected = selectedAnnotation?.type === 'vectors' && selectedAnnotation?.index === idx;
      drawArrow(ctx, vec.start[0], vec.start[1], vec.end[0], vec.end[1], isSelected ? '#ef4444' : 'var(--accent-amber)', isSelected);
    });

    annotations.crops.forEach((crop, idx) => {
      const isSelected = selectedAnnotation?.type === 'crops' && selectedAnnotation?.index === idx;
      ctx.strokeStyle = isSelected ? '#ef4444' : 'var(--accent-green)';
      ctx.lineWidth = isSelected ? 4 : 3;
      ctx.strokeRect(crop.x, crop.y, crop.width, crop.height);
      ctx.fillStyle = isSelected ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.15)';
      ctx.fillRect(crop.x, crop.y, crop.width, crop.height);
    });

    // Draw current drawing state
    if (isDrawing) {
      if (activeTool === 'vector') {
        drawArrow(ctx, startPos.x, startPos.y, currentPos.x, currentPos.y, 'rgba(6, 182, 212, 0.8)');
      } else if (activeTool === 'crop') {
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)';
        ctx.lineWidth = 3;
        ctx.strokeRect(
          startPos.x,
          startPos.y,
          currentPos.x - startPos.x,
          currentPos.y - startPos.y
        );
      }
    }
  }, [annotations, isDrawing, startPos, currentPos, activeTool, selectedAnnotation]);

  const panelStyles = isMaximized ? {
    position: 'fixed',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    zIndex: 9999,
    width: '90vw',
    maxWidth: '760px',
    background: '#0b0f19',
    border: '2px solid var(--accent-cyan)',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.9)',
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  } : {};

  const overlay = isMaximized ? (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9998,
        background: 'rgba(2, 6, 23, 0.9)',
        backdropFilter: 'blur(6px)',
      }}
      onClick={() => setIsMaximized(false)}
    />
  ) : null;

  return (
    <>
      {overlay}
      <div className="panel" style={{ borderColor: 'var(--accent-cyan)', ...panelStyles }}>
        <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
          <div>
            <h2 className="panel-title">
              <Crosshair className="text-cyan-400" size={18} />
              Unified Workspace
            </h2>
            <p className="panel-subtitle">Single viewport with multi-tool annotation (480x480 scale)</p>
          </div>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <select
              value={activeTool}
              onChange={(e) => {
                setActiveTool(e.target.value);
                setSelectedAnnotation(null);
              }}
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

            {selectedAnnotation && (
              <button
                onClick={deleteSelected}
                className="btn-phase btn-phase-action"
                style={{ padding: '6px 10px', color: '#ef4444', borderColor: '#ef4444', display: 'flex', alignItems: 'center', gap: '4px' }}
                title="Delete Selected"
              >
                <Trash2 size={14} />
              </button>
            )}

            <button
              onClick={clearAnnotations}
              className="btn-phase btn-phase-action"
              style={{ padding: '6px 10px' }}
              title="Clear All"
            >
              <RotateCcw size={14} />
            </button>

            <button
              onClick={() => setIsMaximized(!isMaximized)}
              className="btn-phase btn-phase-action"
              style={{ padding: '6px 10px', color: 'var(--accent-cyan)' }}
              title={isMaximized ? "Minimize" : "Maximize"}
            >
              {isMaximized ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
            </button>
          </div>
        </div>

        <div style={{ position: 'relative', width: '100%', maxWidth: isMaximized ? '600px' : '280px', margin: '0 auto' }}>
          <div style={{ position: 'relative', width: '100%', aspectRatio: '1', background: '#000', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-glass)' }}>
            {activeFrame ? (
              <>
                <img
                  src={activeFrame}
                  alt="camera"
                  style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }}
                />
                {samMask && (
                  <img
                    src={samMask}
                    alt="sam mask"
                    style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', mixBlendMode: 'screen', opacity: 0.65, pointerEvents: 'none' }}
                  />
                )}
              </>
            ) : (
              <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', fontSize: '11px' }}>
                Waiting...
              </div>
            )}
            <canvas
              ref={canvasRef}
              width={480}
              height={480}
              style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                cursor: activeTool === 'select' ? 'pointer' : 'crosshair'
              }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onTouchStart={handleMouseDown}
              onTouchMove={handleMouseMove}
              onTouchEnd={handleMouseUp}
            />
          </div>

          <div style={{ marginTop: '8px', display: 'flex', gap: '12px', fontSize: '9px', color: '#64748b', fontFamily: 'monospace', flexWrap: 'wrap' }}>
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
    </>
  );
}
