import React, { useState } from 'react';
import { Target, Search, Image as ImageIcon, Eye } from 'lucide-react';

export default function EncoderDiagnostics({ frame }) {
  const [selectedWord, setSelectedWord] = useState('block');
  const [focalPoint, setFocalPoint] = useState({ x: 120, y: 140 });

  const textPromptWords = ['pinch', 'the', 'red', 'block', 'and', 'lift', 'it'];

  // Handle canvas click to set focal point
  const handleCanvasClick = (e) => {
    const rect = e.target.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 224;
    const y = ((e.clientY - rect.top) / rect.height) * 224;
    setFocalPoint({ x, y });
  };

  // Generate a mock DINO self-attention map centered on the focal point
  const drawDinoAttnMap = () => {
    // Generate simple SVG circles radiating outward from the focal point representing attention intensities
    return (
      <svg viewBox="0 0 224 224" className="w-full h-full" style={{ background: '#050508' }}>
        {/* Draw mock background image frames */}
        {frame && <image href={frame} width="100%" height="100%" opacity="0.45" />}
        
        {/* Radiating heat rings */}
        <circle cx={focalPoint.x} cy={focalPoint.y} r="50" fill="none" stroke="rgba(239, 68, 68, 0.4)" strokeWidth="8" strokeDasharray="3,3" />
        <circle cx={focalPoint.x} cy={focalPoint.y} r="30" fill="rgba(249, 115, 22, 0.25)" stroke="rgba(249, 115, 22, 0.5)" strokeWidth="4" />
        <circle cx={focalPoint.x} cy={focalPoint.y} r="10" fill="#facc15" />
        
        {/* Crosshair indicator */}
        <line x1={focalPoint.x - 12} y1={focalPoint.y} x2={focalPoint.x + 12} y2={focalPoint.y} stroke="#fff" strokeWidth="1.5" />
        <line x1={focalPoint.x} y1={focalPoint.y - 12} x2={focalPoint.x} y2={focalPoint.y + 12} stroke="#fff" strokeWidth="1.5" />
      </svg>
    );
  };

  // Generate a mock CLIP attention map matching the selected word
  const drawClipAttnMap = () => {
    const isRed = selectedWord === 'red';
    const isBlock = selectedWord === 'block';
    
    // Position target depending on word
    let targetX = 112;
    let targetY = 112;
    if (isRed || isBlock) {
      targetX = 140;
      targetY = 160;
    }

    return (
      <svg viewBox="0 0 224 224" className="w-full h-full" style={{ background: '#050508' }}>
        {frame && <image href={frame} width="100%" height="100%" opacity="0.4" />}
        
        {/* Render Jet colormap heatmap */}
        <circle 
          cx={targetX} 
          cy={targetY} 
          r={isBlock || isRed ? "40" : "70"} 
          fill="rgba(59, 130, 246, 0.2)" 
          stroke="rgba(59, 130, 246, 0.5)" 
          strokeWidth="6" 
        />
        <circle 
          cx={targetX} 
          cy={targetY} 
          r={isBlock || isRed ? "25" : "40"} 
          fill="rgba(16, 185, 129, 0.3)" 
          stroke="rgba(16, 185, 129, 0.6)" 
          strokeWidth="4" 
        />
        <circle 
          cx={targetX} 
          cy={targetY} 
          r={isBlock || isRed ? "12" : "15"} 
          fill="rgba(239, 68, 68, 0.6)" 
          stroke="rgba(239, 68, 68, 0.8)" 
          strokeWidth="2" 
        />
      </svg>
    );
  };

  // Generate an isometric projected 3D PointNeXt cloud representation
  const drawPointNextMesh = () => {
    // Generate a set of points in 3D representing a table and a block, projected isometrically
    const pts = [];
    // Floor points
    for (let i = 0; i < 15; i++) {
      for (let j = 0; j < 15; j++) {
        const x = (i - 7) * 12;
        const y = (j - 7) * 12;
        // Project isometrically
        const px = 112 + (x - y) * 0.8;
        const py = 150 + (x + y) * 0.4;
        pts.push({ x: px, y: py, color: '#1e293b' });
      }
    }
    // Block points
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 4; k++) {
          const bx = 140 + (i - 2) * 6;
          const by = 130 + (j - 2) * 6;
          const bz = k * 6;
          const px = bx + (bx - by) * 0.2;
          const py = by - bz;
          pts.push({ x: px, y: py, color: '#ef4444' });
        }
      }
    }

    return (
      <svg viewBox="0 0 224 224" className="w-full h-full" style={{ background: '#020204' }}>
        {pts.map((p, idx) => (
          <circle key={idx} cx={p.x} cy={p.y} r="2.5" fill={p.color} opacity="0.8" />
        ))}
      </svg>
    );
  };

  return (
    <div className="full-page-layout">
      {/* Search Header for Selection */}
      <div className="panel" style={{ padding: '14px 20px', marginBottom: '16px', flexGrow: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 className="panel-title" style={{ fontSize: '15px' }}>
              <Eye size={18} className="text-cyan-400" />
              Pre-trained Encoders Diagnostics
            </h2>
            <p className="panel-subtitle">Visualize foundation feature maps (DINOv3, CLIP, PointNeXt)</p>
          </div>
          
          {/* Word Token selector for CLIP */}
          <div className="flex-row-center gap-8">
            <span className="form-label" style={{ fontSize: '9px' }}>CLIP Text Embedder Alignment:</span>
            <div style={{ display: 'flex', gap: '4px', background: '#0f1015', padding: '4px', borderRadius: '6px', border: '1px solid #1e293b' }}>
              {textPromptWords.map((word, idx) => (
                <button
                  key={idx}
                  onClick={() => setSelectedWord(word)}
                  className={`px-2 py-0.5 text-xs font-mono rounded transition ${
                    selectedWord === word 
                      ? 'bg-cyan-500 text-black font-semibold' 
                      : 'text-neutral-400 hover:text-neutral-200'
                  }`}
                >
                  {word}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Diagnostics Visualizers Grid */}
      <div className="diagnostics-grid">
        {/* DINOv3 Attn Map */}
        <div className="panel diagnostics-card">
          <div className="panel-header">
            <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Target size={12} className="text-red-400" />
              DINOv3 Spatial Attention (Click to Focus)
            </span>
          </div>
          <div className="diagnostics-viewport" onClick={handleCanvasClick}>
            {drawDinoAttnMap()}
          </div>
          <div style={{ fontSize: '11px', color: '#64748b', textAlign: 'center' }}>
            Visualizes dense self-attention overlays. Coordinates: ({focalPoint.x.toFixed(0)}, {focalPoint.y.toFixed(0)})
          </div>
        </div>

        {/* CLIP Attention Heatmap */}
        <div className="panel diagnostics-card">
          <div className="panel-header">
            <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <ImageIcon size={12} className="text-cyan-400" />
              CLIP Semantic Matching (Jet Map)
            </span>
          </div>
          <div className="diagnostics-viewport">
            {drawClipAttnMap()}
          </div>
          <div style={{ fontSize: '11px', color: '#64748b', textAlign: 'center' }}>
            Visualizes target overlap for token: <code className="text-cyan-400">"{selectedWord}"</code>
          </div>
        </div>

        {/* PointNeXt 3D cloud */}
        <div className="panel diagnostics-card">
          <div className="panel-header">
            <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Target size={12} className="text-green-400" />
              PointNeXt 3D Geometry Activations
            </span>
          </div>
          <div className="diagnostics-viewport">
            {drawPointNextMesh()}
          </div>
          <div style={{ fontSize: '11px', color: '#64748b', textAlign: 'center' }}>
            1024-point cloud generated via Depth Anything V2
          </div>
        </div>
      </div>
    </div>
  );
}
