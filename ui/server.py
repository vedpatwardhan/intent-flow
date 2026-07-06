import asyncio
import base64
import io
import json
import math
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from PIL import Image, ImageDraw

app = FastAPI()

# GNN Skill Library mock nodes
skills = [
    {
        "id": "1",
        "name": "reach_drawer",
        "type": "internalized",
        "x": 60,
        "y": 75,
        "active": True,
    },
    {
        "id": "2",
        "name": "pinch_cube",
        "type": "internalized",
        "x": 150,
        "y": 45,
        "active": False,
    },
    {
        "id": "3",
        "name": "lift_cube",
        "type": "externalized",
        "x": 240,
        "y": 75,
        "active": False,
    },
]

# State variables
robot_pos = [320, 100]  # Base of arm
gripper_pos = [200, 260]  # Hand position
target_pos = [400, 260]  # Cube target
block_pos = [400, 260]  # Dynamic target block

energy = 0.5
combostoc_noise = {"torso": 0.0, "arm": 0.0, "hand": 0.0, "vision": 0.0}
attack_active = False


def draw_mock_simulator():
    """Generates a beautiful 640x360 simulation image using PIL."""
    # Create image with deep dark gray workspace background
    img = Image.new("RGB", (640, 360), "#0A0A0F")
    draw = ImageDraw.Draw(img)

    # Draw grid floorlines
    for y in range(200, 360, 20):
        draw.line([(0, y), (640, y)], fill="#151520", width=1)
    for x in range(0, 640, 40):
        # Isometric floor grid lines
        draw.line([(x, 200), (x - 80, 360)], fill="#151520", width=1)
        draw.line([(x, 200), (x + 80, 360)], fill="#151520", width=1)

    # Draw table/surface boundary
    draw.line([(0, 200), (640, 200)], fill="#222230", width=2)

    # Draw block/cube target
    bx, by = block_pos
    draw.rectangle(
        [bx - 15, by - 15, bx + 15, by + 15], fill="#22c55e", outline="#16a34a", width=2
    )
    # Give the block a 3D shading look
    draw.rectangle([bx - 15, by - 15, bx + 15, by - 10], fill="#4ade80")

    # Draw simple two-joint robot arm
    # Joint 1 (elbow)
    elbow_pos = [
        (robot_pos[0] + gripper_pos[0]) // 2 + 50,
        (robot_pos[1] + gripper_pos[1]) // 2 - 40,
    ]
    # Draw upper arm
    draw.line(
        [robot_pos[0], robot_pos[1], elbow_pos[0], elbow_pos[1]],
        fill="#06b6d4",
        width=8,
    )
    # Draw forearm
    draw.line(
        [elbow_pos[0], elbow_pos[1], gripper_pos[0], gripper_pos[1]],
        fill="#0891b2",
        width=6,
    )

    # Draw joint highlights
    draw.ellipse(
        [robot_pos[0] - 8, robot_pos[1] - 8, robot_pos[0] + 8, robot_pos[1] + 8],
        fill="#3b82f6",
    )
    draw.ellipse(
        [elbow_pos[0] - 6, elbow_pos[1] - 6, elbow_pos[0] + 6, elbow_pos[1] + 6],
        fill="#3b82f6",
    )

    # Draw simple gripper/fingers
    gx, gy = gripper_pos
    draw.line([gx - 10, gy, gx - 10, gy + 15], fill="#e2e8f0", width=4)
    draw.line([gx + 10, gy, gx + 10, gy + 15], fill="#e2e8f0", width=4)
    draw.line([gx - 10, gy, gx + 10, gy], fill="#cbd5e1", width=3)

    # If attack is active, draw red static lines / visual glitching
    if attack_active:
        for _ in range(3):
            gy_rand = random.randint(20, 340)
            draw.line([(0, gy_rand), (640, gy_rand)], fill="#ef4444", width=1)

    # Buffer image to base64 string
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(byte_im).decode("utf-8")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global target_pos, block_pos, gripper_pos, energy, attack_active, combostoc_noise
    await websocket.accept()
    print("UI Connected via WebSocket")

    # Track steps to simulate motion
    step_count = 0

    try:
        while True:
            # Check for messages from client (non-blocking style)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.03)
                payload = json.loads(data)
                print(f"Received interactive command: {payload}")

                # Update target when user draws a bounding box or arrow vector
                if (
                    payload.get("type") == "bounding_box"
                    or payload.get("type") == "motion_vector"
                ):
                    coords = payload["coordinates"]
                    if "x" in coords:  # Bounding Box
                        # Shift block to box center (scale down 640x360 canvas targets if needed)
                        target_pos = [
                            int(coords["x"] + coords["width"] / 2),
                            int(coords["y"] + coords["height"] / 2),
                        ]
                    elif "start" in coords:  # Motion Vector
                        target_pos = [int(coords["end"][0]), int(coords["end"][1])]

                elif payload.get("type") == "text_command":
                    prompt = payload.get("prompt", "").lower()
                    print(f"Executing CLIP text prompt conditioning: {prompt}")
                    # Change block destination based on text instruction
                    if "lift" in prompt:
                        target_pos = [target_pos[0], 120]
                    elif "left" in prompt:
                        target_pos = [150, 260]
                    elif "right" in prompt:
                        target_pos = [450, 260]

                elif payload.get("type") == "combostoc_noise":
                    group = payload["group"]
                    val = payload["value"]
                    combostoc_noise[group] = val

                elif payload.get("type") == "trigger_attack":
                    attack_active = payload["active"]

                elif payload.get("type") == "clear":
                    target_pos = [400, 260]

            except asyncio.TimeoutError:
                pass  # No messages received, continue loop

            # Simulate robot kinematics moving toward target
            # Smooth interpolation
            dx = target_pos[0] - gripper_pos[0]
            dy = target_pos[1] - gripper_pos[1]
            dist = math.hypot(dx, dy)

            if dist > 2:
                # Interpolate towards target
                gripper_pos[0] += dx * 0.15
                gripper_pos[1] += dy * 0.15
                # Block follows gripper if picked up
                if dist < 25 and target_pos[1] < 200:  # Lift block
                    block_pos = list(gripper_pos)
            else:
                # Add tiny random jitter
                gripper_pos[0] += random.uniform(-0.5, 0.5)
                gripper_pos[1] += random.uniform(-0.5, 0.5)

            # Update EBM energy index (closer to target block = lower energy compatibility score)
            err = math.hypot(block_pos[0] - target_pos[0], block_pos[1] - target_pos[1])
            energy = 0.05 + (err / 300.0) * 0.8
            # Add ComboStoc timeline noise or adversarial attack penalties
            if attack_active:
                energy += random.uniform(0.4, 0.6)
            energy += sum(combostoc_noise.values()) * 0.15
            energy = min(max(energy, 0.01), 1.2)

            # Generate tactile feedback (spikes when gripping block)
            dist_to_block = math.hypot(
                gripper_pos[0] - block_pos[0], gripper_pos[1] - block_pos[1]
            )
            touch_val = (
                max(0.0, 1.0 - (dist_to_block / 35.0)) if dist_to_block < 35 else 0.0
            )

            # Construct 2x2 tactile matrix
            tactile_grid = [
                [touch_val * 0.9 + random.uniform(0, 0.05), touch_val * 0.2],
                [touch_val * 0.1, touch_val * 0.85 + random.uniform(0, 0.05)],
            ]

            # Generate joint torques based on velocity forces
            joint_torques = [
                5.0 + math.sin(step_count * 0.1) * 3.0 + combostoc_noise["torso"] * 10,
                -2.0 + math.cos(step_count * 0.15) * 4.0 + combostoc_noise["arm"] * 8,
                touch_val * 14.0 + combostoc_noise["hand"] * 5,
                (dx * 0.05) + random.uniform(-1, 1),
            ]
            joint_positions = [
                (gripper_pos[0] / 300.0) - 1.0,
                (gripper_pos[1] / 180.0) - 1.0,
                touch_val * 0.5,
                (target_pos[0] / 300.0) - 1.0,
            ]

            # Evolve GNN node active status over time
            if step_count % 150 == 0:  # Every 4.5 seconds
                # Rotate active node
                for s in skills:
                    s["active"] = False
                active_idx = (step_count // 150) % len(skills)
                skills[active_idx]["active"] = True

            # Draw image
            img_b64 = draw_mock_simulator()

            # Compile payload
            ws_payload = {
                "frame": img_b64,
                "energy": energy,
                "tactile_grid": tactile_grid,
                "joints": {"positions": joint_positions, "torques": joint_torques},
                "skills": skills,
            }

            # Send payload to client
            await websocket.send_text(json.dumps(ws_payload))

            step_count += 1
            await asyncio.sleep(0.03)  # ~30fps loop

    except WebSocketDisconnect:
        print("UI Disconnected")
    except Exception as e:
        print(f"WebSocket Loop Error: {e}")
