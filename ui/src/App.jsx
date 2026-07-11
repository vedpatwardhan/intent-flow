import React, { useState, useEffect, useRef } from 'react';
import SimulatorView from './components/SimulatorView';
import ControlPanel from './components/ControlPanel';
import TelemetryPanel from './components/TelemetryPanel';
import GnnLibrary from './components/GnnLibrary';
import TrajectoryExplorer from './components/TrajectoryExplorer';
import EncoderDiagnostics from './components/EncoderDiagnostics';
import SkillComposer from './components/SkillComposer';
import { Cpu, RefreshCw, Grid, PlayCircle, Eye, GitCommit } from 'lucide-react';

const base64ToBlob = (base64Str, contentType = 'image/jpeg') => {
  const parts = base64Str.split(';base64,');
  const base64Data = parts[1] || parts[0];
  const sliceSize = 1024;
  const byteCharacters = atob(base64Data);
  const byteArrays = [];

  for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
    const slice = byteCharacters.slice(offset, offset + sliceSize);
    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }
  return new Blob(byteArrays, { type: contentType });
};

export default function App() {
  const [activePage, setActivePage] = useState('command'); // 'command', 'trajectories', 'encoders', 'skills'
  const [frames, setFrames] = useState(null);
  const [energy, setEnergy] = useState(0.0);
  const [energyHistory, setEnergyHistory] = useState(Array(30).fill(0.0));
  const [tactileGrid, setTactileGrid] = useState([[0, 0], [0, 0]]);
  const [joints, setJoints] = useState({ positions: [0, 0, 0, 0], torques: [0, 0, 0, 0] });
  const [skills, setSkills] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [activeCam, setActiveCam] = useState('world_center');
  const [dinoAttnCache, setDinoAttnCache] = useState({});
  const [clipSimCache, setClipSimCache] = useState({});
  const [samMaskCache, setSamMaskCache] = useState({});
  const [pointCloudCache, setPointCloudCache] = useState({});
  const [vggtTracksCache, setVggtTracksCache] = useState({});
  const [taskIsolatedFeaturesCache, setTaskIsolatedFeaturesCache] = useState({});
  const [recentCameras, setRecentCameras] = useState([]);

  const wsRef = useRef(null);
  const activeCamRef = useRef('world_center');
  const framesRef = useRef(null);

  useEffect(() => {
    framesRef.current = frames;
  }, [frames]);

  useEffect(() => {
    activeCamRef.current = activeCam;
  }, [activeCam]);

  // Cleanup caches based on recent cameras sliding window
  useEffect(() => {
    setDinoAttnCache(prev => {
      const updated = { ...prev };
      Object.keys(updated).forEach(cam => {
        if (!recentCameras.includes(cam)) delete updated[cam];
      });
      return updated;
    });
    setClipSimCache(prev => {
      const updated = { ...prev };
      Object.keys(updated).forEach(cam => {
        if (!recentCameras.includes(cam)) delete updated[cam];
      });
      return updated;
    });
    setSamMaskCache(prev => {
      const updated = { ...prev };
      Object.keys(updated).forEach(cam => {
        if (!recentCameras.includes(cam)) delete updated[cam];
      });
      return updated;
    });
    setPointCloudCache(prev => {
      const updated = { ...prev };
      Object.keys(updated).forEach(cam => {
        if (!recentCameras.includes(cam)) delete updated[cam];
      });
      return updated;
    });
    setVggtTracksCache(prev => {
      const updated = { ...prev };
      Object.keys(updated).forEach(cam => {
        if (!recentCameras.includes(cam)) delete updated[cam];
      });
      return updated;
    });
    setTaskIsolatedFeaturesCache(prev => {
      const updated = { ...prev };
      Object.keys(updated).forEach(cam => {
        if (!recentCameras.includes(cam)) delete updated[cam];
      });
      return updated;
    });
  }, [recentCameras]);

  useEffect(() => {
    connectWS();
    return () => {
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
      if (framesRef.current) {
        Object.values(framesRef.current).forEach(url => {
          if (url.startsWith('blob:')) {
            URL.revokeObjectURL(url);
          }
        });
      }
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
        if (data.frames) {
          setFrames(prevFrames => {
            if (prevFrames) {
              Object.values(prevFrames).forEach(url => {
                if (url.startsWith('blob:')) {
                  URL.revokeObjectURL(url);
                }
              });
            }
            const newFrames = {};
            Object.entries(data.frames).forEach(([camId, base64Str]) => {
              try {
                const blob = base64ToBlob(base64Str);
                newFrames[camId] = URL.createObjectURL(blob);
              } catch (e) {
                console.error("Error converting frame to blob:", e);
                newFrames[camId] = base64Str;
              }
            });
            return newFrames;
          });
        }
        if (data.energy !== undefined) {
          setEnergy(data.energy);
          setEnergyHistory(prev => [...prev.slice(1), data.energy]);
        }
        if (data.tactile_grid) setTactileGrid(data.tactile_grid);
        if (data.joints) setJoints(data.joints);
        if (data.skills) setSkills(data.skills);
        const currentCam = activeCamRef.current;

        // Update recent cameras list with sliding window of max 2
        setRecentCameras(prev => {
          const updated = [currentCam, ...prev.filter(cam => cam !== currentCam)];
          return updated.slice(0, 2);
        });

        if (data.dino_attn !== undefined) {
          setDinoAttnCache(prev => ({ ...prev, [currentCam]: data.dino_attn }));
        }
        if (data.clip_sim !== undefined) {
          setClipSimCache(prev => ({ ...prev, [currentCam]: data.clip_sim }));
        }
        if (data.sam_mask !== undefined) {
          setSamMaskCache(prev => ({ ...prev, [currentCam]: data.sam_mask }));
        }
        if (data.point_cloud !== undefined) {
          setPointCloudCache(prev => ({ ...prev, [currentCam]: data.point_cloud }));
        }
        if (data.vggt_tracks !== undefined) {
          setVggtTracksCache(prev => ({ ...prev, [currentCam]: data.vggt_tracks }));
        }
        if (data.task_isolated_features !== undefined) {
          setTaskIsolatedFeaturesCache(prev => ({ ...prev, [currentCam]: data.task_isolated_features }));
        }
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
            className={`btn-conn ${connectionStatus === 'connected' ? 'connected' : 'disconnected'}`}
          >
            {connectionStatus === 'connected' ? 'ONLINE' : 'OFFLINE'}
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
        <EncoderDiagnostics
          frame={frames?.[activeCam] || frames?.world_center}
          frames={frames}
          dinoAttn={dinoAttnCache[activeCam]}
          clipSim={clipSimCache[activeCam]}
          samMask={samMaskCache[activeCam]}
          pointCloud={pointCloudCache[activeCam] || []}
          vggtTracks={vggtTracksCache[activeCam] || []}
          activeCam={activeCam}
          onCameraChange={setActiveCam}
          onInteraction={handleInteraction}
          taskIsolatedFeatures={taskIsolatedFeaturesCache[activeCam]}
        />
      )}
      {activePage === 'skills' && (
        <SkillComposer />
      )}
    </div>
  );
}
