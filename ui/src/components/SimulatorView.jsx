import React, { useRef, useState, useEffect } from 'react';
import { Focus, Navigation } from 'lucide-react';

export default function SimulatorView({ frame, onInteraction, connectionStatus }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  
  const [tool, setTool] = useState('box'); // 'box' or 'vector'
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentPos, setCurrentPos] = useState({ x: 0, y: 0 });
  const [drawnBox, setDrawnBox] = useState(null);
  const [drawnVector, setDrawnVector] = useState(null);

  // Redraw canvas content (bounding box or vector arrow)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw active box
    if (drawnBox) {
      ctx.strokeStyle = '#22c55e'; // Green
      ctx.lineWidth = 2;
      ctx.strokeRect(drawnBox.x, drawnBox.y, drawnBox.width, drawnBox.height);
      ctx.fillStyle = 'rgba(34, 197, 94, 0.15)';
      ctx.fillRect(drawnBox.x, drawnBox.y, drawnBox.width, drawnBox.height);
      
      // Target text label
      ctx.fillStyle = '#22c55e';
      ctx.font = '12px Outfit';
      ctx.fillText("Goal Mask", drawnBox.x + 4, drawnBox.y + 16);
    }

    // Draw active vector arrow
    if (drawnVector) {
      drawArrow(ctx, drawnVector.start[0], drawnVector.start[1], drawnVector.end[0], drawnVector.end[1]);
    }

    // Draw active drawing feedback
    if (isDrawing) {
      if (tool === 'box') {
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)'; // Cyan
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
  }, [isDrawing, startPos, currentPos, drawnBox, drawnVector, tool]);

  const drawArrow = (ctx, fromx, fromy, tox, toy, color = '#06b6d4') => {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    
    // Draw line
    ctx.beginPath();
    ctx.moveTo(fromx, fromy);
    ctx.lineTo(tox, toy);
    ctx.stroke();

    // Draw arrowhead
    const angle = Math.atan2(toy - fromy, tox - fromx);
    ctx.beginPath();
    ctx.moveTo(tox, toy);
    ctx.lineTo(tox - 12 * Math.cos(angle - Math.PI / 6), toy - 12 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(tox - 12 * Math.cos(angle + Math.PI / 6), toy - 12 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  };

  const getMousePos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    // Support mouse and touch events
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);
    return {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
  };

  const handleMouseDown = (e) => {
    const pos = getMousePos(e);
    setIsDrawing(true);
    setStartPos(pos);
    setCurrentPos(pos);
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
      
      // Ensure box has non-trivial dimensions
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

  return (
    <div className="glass-panel flex flex-col h-full glow-cyan">
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}`} />
          <h2 className="font-semibold text-lg">MuJoCo Simulator View</h2>
        </div>
        <div className="flex gap-2">
          <button 
            onClick={() => setTool('box')} 
            className={`p-2 rounded border transition ${tool === 'box' ? 'bg-cyan-500/20 border-cyan-500 text-cyan-400' : 'border-neutral-800 hover:bg-neutral-800'}`}
            title="Goal Bounding Box"
          >
            <Focus size={16} />
          </button>
          <button 
            onClick={() => setTool('vector')} 
            className={`p-2 rounded border transition ${tool === 'vector' ? 'bg-cyan-500/20 border-cyan-500 text-cyan-400' : 'border-neutral-800 hover:bg-neutral-800'}`}
            title="Directional Motion Vector"
          >
            <Navigation size={16} className="rotate-45" />
          </button>
          <button 
            onClick={clearCanvas} 
            className="px-3 py-1 text-xs border border-neutral-800 hover:bg-red-500/10 hover:border-red-500/40 hover:text-red-400 rounded transition"
          >
            Clear Overlay
          </button>
        </div>
      </div>

      <div 
        ref={containerRef} 
        className="relative flex-grow bg-black rounded-lg overflow-hidden border border-neutral-900 aspect-video flex items-center justify-center"
      >
        {frame ? (
          <img 
            src={frame} 
            alt="Sim feed" 
            className="w-full h-full object-contain pointer-events-none"
          />
        ) : (
          <div className="text-neutral-500 text-sm">Waiting for simulator feed...</div>
        )}
        
        <canvas
          ref={canvasRef}
          width={640}
          height={360}
          className="absolute inset-0 w-full h-full cursor-crosshair z-10"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onTouchStart={handleMouseDown}
          onTouchMove={handleMouseMove}
          onTouchEnd={handleMouseUp}
        />
      </div>
    </div>
  );
}
