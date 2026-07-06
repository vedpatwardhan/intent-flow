import React, { useState, useEffect, useRef } from 'react';
import SimulatorView from './components/SimulatorView';
import ControlPanel from './components/ControlPanel';
import TelemetryPanel from './components/TelemetryPanel';
import GnnLibrary from './components/GnnLibrary';
import { Cpu, RefreshCw } from 'lucide-react';

export default function App() {
  const [frame, setFrame] = useState(null);
  const [energy, setEnergy] = useState(0.0);
  const [energyHistory, setEnergyHistory] = useState(Array(30).fill(0.0));
  const [tactileGrid, setTactileGrid] = useState([[0, 0], [0, 0]]);
  const [joints, setJoints] = useState({ positions: [0, 0, 0, 0], torques: [0, 0, 0, 0] });
  const [skills, setSkills] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  
  const wsRef = useRef(null);

  // WebSocket connect logic
  useEffect(() => {
    connectWS();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const connectWS = () => {
    setConnectionStatus('connecting');
    const ws = new WebSocket('ws://localhost:8000/ws');
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus('connected');
      console.log('WebSocket Connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.frame) setFrame(data.frame);
        if (data.energy !== undefined) {
          setEnergy(data.energy);
          setEnergyHistory(prev => [...prev.slice(1), data.energy]);
        }
        if (data.tactile_grid) setTactileGrid(data.tactile_grid);
        if (data.joints) setJoints(data.joints);
        if (data.skills) setSkills(data.skills);
      } catch (err) {
        console.error('Error parsing WS frame:', err);
      }
    };

    ws.onclose = () => {
      setConnectionStatus('disconnected');
      console.log('WebSocket Closed. Retrying in 3 seconds...');
      setTimeout(connectWS, 3000);
    };

    ws.onerror = (err) => {
      console.error('WebSocket Error:', err);
      ws.close();
    };
  };

  const handleInteraction = (interaction) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(interaction));
    }
  };

  const handleComboStocChange = (group, val) => {
    handleInteraction({
      type: 'combostoc_noise',
      group,
      value: val
    });
  };

  const handleTriggerAttack = (active) => {
    handleInteraction({
      type: 'trigger_attack',
      active
    });
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Top Navigation / Dashboard Header */}
      <header className="glass-panel mx-4 mt-4 flex justify-between items-center py-3 px-6 border border-white/5 rounded-xl bg-black/40">
        <div className="flex items-center gap-3">
          <div className="bg-cyan-500/10 p-2 rounded-lg border border-cyan-500/20">
            <Cpu className="text-cyan-400 animate-pulse" size={24} />
          </div>
          <div>
            <h1 className="text-lg font-bold text-neutral-100 tracking-wide font-sans">CORTEX OS</h1>
            <span className="text-[10px] text-neutral-500 font-mono">LATENT-FLOW CONTROLLER • STAGE 3 VLA</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex flex-col items-end">
            <span className="text-[10px] text-neutral-400 font-mono uppercase">Simulation Engine</span>
            <span className="text-xs text-neutral-500 font-mono">MuJoCo (Local)</span>
          </div>
          <button 
            onClick={() => { if (wsRef.current) wsRef.current.close(); }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded border text-xs transition ${
              connectionStatus === 'connected' 
                ? 'bg-green-500/10 border-green-500/20 text-green-400' 
                : connectionStatus === 'connecting'
                ? 'bg-amber-500/10 border-amber-500/20 text-amber-400'
                : 'bg-red-500/10 border-red-500/20 text-red-400'
            }`}
          >
            {connectionStatus === 'connecting' && <RefreshCw size={12} className="animate-spin" />}
            {connectionStatus === 'connected' ? 'ONLINE' : connectionStatus === 'connecting' ? 'CONNECTING' : 'OFFLINE'}
          </button>
        </div>
      </header>

      {/* Main Grid Workspace */}
      <main className="dashboard-grid flex-grow">
        {/* Left Column: Control Panel */}
        <ControlPanel 
          onUserCommand={handleInteraction}
          onComboStocChange={handleComboStocChange}
          onTriggerAttack={handleTriggerAttack}
        />

        {/* Center Column: Simulator Feed & Canvas Overlay */}
        <SimulatorView 
          frame={frame}
          onInteraction={handleInteraction}
          connectionStatus={connectionStatus}
        />

        {/* Right Column: Telemetry & GNN Library */}
        <div className="flex flex-col gap-4">
          <div className="flex-grow">
            <TelemetryPanel 
              energy={energy} 
              energyHistory={energyHistory}
              tactileGrid={tactileGrid}
              joints={joints}
            />
          </div>
          <GnnLibrary skills={skills} />
        </div>
      </main>
    </div>
  );
}
