import React, { useRef, useState, useEffect } from 'react';
import { Focus, Navigation, Camera } from 'lucide-react';

export default function SimulatorView({ frames, onInteraction, connectionStatus }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  const [tool, setTool] = useState('box');
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentPos, setCurrentPos] = useState({ x: 0, y: 0 });
  const [drawnBox, setDrawnBox] = useState(null);
  const [drawnVector, setDrawnVector] = useState(null);
  const [activeCam, setActiveCam] = useState('world_center');

  const cameras = [
    { id: 'world_center', name: 'Center' },
    { id: 'world_top', name: 'Top' },
    { id: 'world_left', name: 'Left' },
    { id: 'world_right', name: 'Right' },
    { id: 'world_wrist', name: 'Wrist' }
  ];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (drawnBox) {
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2;
      ctx.strokeRect(drawnBox.x, drawnBox.y, drawnBox.width, drawnBox.height);
      ctx.fillStyle = 'rgba(34, 197, 94, 0.15)';
      ctx.fillRect(drawnBox.x, drawnBox.y, drawnBox.width, drawnBox.height);
      ctx.fillStyle = '#22c55e';
      ctx.font = '10px Outfit';
      ctx.fillText("Goal Mask", drawnBox.x + 4, drawnBox.y + 14);
    }

    if (drawnVector) {
      drawArrow(ctx, drawnVector.start[0], drawnVector.start[1], drawnVector.end[0], drawnVector.end[1]);
    }

    if (isDrawing) {
      if (tool === 'box') {
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)';
        ctx.lineWidth = 2;
        ctx.strokeRect(
          startPos.x,
          startPos.y,
          currentPos.x - startPos.x,
          currentPos.y - startPos.y
        );
      } else if (tool === 'vector') {
        drawArrow(ctx, startPos.x, startPos.y, currentPos.x, currentPos.y, 'rgba(6, 182, 212, 0.8)');
      }
    }
  }, [isDrawing, startPos, currentPos, drawnBox, drawnVector, tool, activeCam]);

  const drawArrow = (ctx, fromx, fromy, tox, toy, color = '#06b6d4') => {
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

  const getMousePos = (e) => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);

    // Map bounding client coordinates to 224x224 camera pixels
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
    onInteraction({
      type: 'original_click',
      x: pos.x,
      y: pos.y
    });
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    setCurrentPos(getMousePos(e));
  };

  const handleMouseUp = () => {
    if (!isDrawing) return;
    setIsDrawing(false);

    if (tool === 'box') {
      const box = {
        x: Math.min(startPos.x, currentPos.x),
        y: Math.min(startPos.y, currentPos.y),
        width: Math.abs(currentPos.x - startPos.x),
        height: Math.abs(currentPos.y - startPos.y)
      };

      if (box.width > 5 && box.height > 5) {
        setDrawnBox(box);
        onInteraction({
          type: 'bounding_box',
          coordinates: box
        });
      }
    } else {
      const vector = {
        start: [startPos.x, startPos.y],
        end: [currentPos.x, currentPos.y]
      };
      const distance = Math.hypot(vector.end[0] - vector.start[0], vector.end[1] - vector.start[1]);
      if (distance > 5) {
        setDrawnVector(vector);
        onInteraction({
          type: 'motion_vector',
          coordinates: vector
        });
      }
    }
  };

  const clearCanvas = () => {
    setDrawnBox(null);
    setDrawnVector(null);
    onInteraction({ type: 'clear' });
  };

  const handleCameraChange = (camId) => {
    setActiveCam(camId);
    // Clear canvas when switching camera focus to prevent coordinate leaks
    setDrawnBox(null);
    setDrawnVector(null);
    onInteraction({
      type: 'select_camera',
      camera: camId
    });
  };

  return (
    <div className="panel h-full" style={{ borderColor: 'var(--accent-cyan)' }}>
      <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <div>
          <h2 className="panel-title">
            <Camera className="text-cyan-400" size={18} />
            Simulator Grid
          </h2>
          <p className="panel-subtitle">Multi-view observation cockpit (Click a view to Focus & Annotate)</p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => setTool('box')}
            className={`btn-phase btn-phase-action ${tool === 'box' ? 'active' : ''}`}
            style={{
              padding: '6px 10px',
              background: tool === 'box' ? 'var(--accent-cyan-dim)' : '',
              borderColor: tool === 'box' ? 'var(--accent-cyan)' : ''
            }}
            title="Goal Bounding Box"
          >
            <Focus size={14} className={tool === 'box' ? 'text-cyan-400' : ''} />
          </button>
          <button
            onClick={() => setTool('vector')}
            className={`btn-phase btn-phase-action ${tool === 'vector' ? 'active' : ''}`}
            style={{
              padding: '6px 10px',
              background: tool === 'vector' ? 'var(--accent-cyan-dim)' : '',
              borderColor: tool === 'vector' ? 'var(--accent-cyan)' : ''
            }}
            title="Directional Motion Vector"
          >
            <Navigation size={14} className={`rotate-45 ${tool === 'vector' ? 'text-cyan-400' : ''}`} />
          </button>
          <button
            onClick={clearCanvas}
            className="btn-phase btn-phase-action"
            style={{ padding: '6px 10px' }}
          >
            Clear
          </button>
        </div>
      </div>

      {/* Simulator Multi-View Grid */}
      <div className="cam-grid" ref={containerRef}>
        {cameras.map((cam) => (
          <div
            key={cam.id}
            onClick={() => handleCameraChange(cam.id)}
            className={`cam-grid-card ${activeCam === cam.id ? 'focused' : ''}`}
          >
            {frames && frames[cam.id] ? (
              <img
                src={frames[cam.id]}
                alt={cam.name}
              />
            ) : (
              <div style={{ color: '#475569', fontSize: '11px' }}>Waiting...</div>
            )}
            <span className="cam-grid-label">{cam.name} View</span>

            {/* Canvas overlay ONLY loaded on the focused camera */}
            {activeCam === cam.id && (
              <>
                <span className="cam-focus-overlay">Focus</span>
                <canvas
                  ref={canvasRef}
                  width={224}
                  height={224}
                  className="focused-canvas"
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onTouchStart={handleMouseDown}
                  onTouchMove={handleMouseMove}
                  onTouchEnd={handleMouseUp}
                />
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
