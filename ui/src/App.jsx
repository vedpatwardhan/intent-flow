import React, { useState, useEffect, useRef } from 'react';
import SimulatorView from './components/SimulatorView';
import ControlPanel from './components/ControlPanel';
import TelemetryPanel from './components/TelemetryPanel';
import GnnLibrary from './components/GnnLibrary';
import TrajectoryExplorer from './components/TrajectoryExplorer';
import EncoderDiagnostics from './components/EncoderDiagnostics';
import SkillComposer from './components/SkillComposer';
import { Cpu, RefreshCw, Grid, PlayCircle, Eye, GitCommit } from 'lucide-react';

export default function App() {
  const [activePage, setActivePage] = useState('command'); // 'command', 'trajectories', 'encoders', 'skills'
  const [frames, setFrames] = useState(null);
  const [energy, setEnergy] = useState(0.0);
  const [energyHistory, setEnergyHistory] = useState(Array(30).fill(0.0));
  const [tactileGrid, setTactileGrid] = useState([[0, 0], [0, 0]]);
  const [joints, setJoints] = useState({ positions: [0, 0, 0, 0], torques: [0, 0, 0, 0] });
  const [skills, setSkills] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  const wsRef = useRef(null);

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
        if (data.frames) setFrames(data.frames);
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
      {/* Header Navbar with Navigation Tabs */}
      <header className="app-header">
        <div className="header-left">
          <div className="header-logo">
            <Cpu size={20} />
          </div>
          <div className="header-title-group">
            <h1>LATENT-FLOW</h1>
            <span>RL CONTROLLER</span>
          </div>
        </div>

        {/* Dashboard Navigation Tabs */}
        <nav className="header-nav">
          <button
            onClick={() => setActivePage('command')}
            className={`nav-tab flex-row-center gap-8 ${activePage === 'command' ? 'active' : ''}`}
          >
            <Grid size={14} />
            Command Center
          </button>
          <button
            onClick={() => setActivePage('trajectories')}
            className={`nav-tab flex-row-center gap-8 ${activePage === 'trajectories' ? 'active' : ''}`}
          >
            <PlayCircle size={14} />
            Trajectory Explorer
          </button>
          <button
            onClick={() => setActivePage('encoders')}
            className={`nav-tab flex-row-center gap-8 ${activePage === 'encoders' ? 'active' : ''}`}
          >
            <Eye size={14} />
            Encoder Diagnostics
          </button>
          <button
            onClick={() => setActivePage('skills')}
            className={`nav-tab flex-row-center gap-8 ${activePage === 'skills' ? 'active' : ''}`}
          >
            <GitCommit size={14} />
            Skill Composer
          </button>
        </nav>

        <div className="header-right">
          <div className="status-indicator-group">
            <div className="status-label">
              Simulation Engine
            </div>
            <div className="status-val">
              MuJoCo (Local)
            </div>
          </div>
          <button
            onClick={() => { if (wsRef.current) wsRef.current.close(); }}
            className={`btn-conn ${connectionStatus}`}
          >
            {connectionStatus === 'connecting' && <RefreshCw size={11} className="animate-spin" />}
            {connectionStatus === 'connected' ? 'ONLINE' : connectionStatus === 'connecting' ? 'CONNECTING' : 'OFFLINE'}
          </button>
        </div>
      </header>

      {/* Main Pages Content */}
      {activePage === 'command' && (
        <main className="dashboard-layout">
          {/* Left Column: Control Panel */}
          <ControlPanel
            onUserCommand={handleInteraction}
            onComboStocChange={handleComboStocChange}
            onTriggerAttack={handleTriggerAttack}
          />

          {/* Center Column: Grid View of 5 Cameras */}
          <SimulatorView
            frames={frames}
            onInteraction={handleInteraction}
            connectionStatus={connectionStatus}
          />

          {/* Right Column: Telemetry & GNN Library summary */}
          <div className="dashboard-column">
            <TelemetryPanel
              energy={energy}
              energyHistory={energyHistory}
              tactileGrid={tactileGrid}
              joints={joints}
            />
            <GnnLibrary skills={skills} />
          </div>
        </main>
      )}

      {activePage === 'trajectories' && (
        <TrajectoryExplorer />
      )}

      {activePage === 'encoders' && (
        <EncoderDiagnostics frame={frames?.world_center} />
      )}

      {activePage === 'skills' && (
        <SkillComposer />
      )}
    </div>
  );
}
