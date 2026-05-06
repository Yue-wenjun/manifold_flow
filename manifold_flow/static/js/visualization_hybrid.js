import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// Global state
const socket = io();
let streamId = null;
let currentSystemId = null;
let scene, camera, renderer, composer, controls, swarmRenderer;

const toggleTraj = document.getElementById('toggle-traj');
const togglePart = document.getElementById('toggle-part');

function createGlowTexture(coreColorStr) {
    const canvas = document.createElement('canvas');
    canvas.width = 64; canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
    grad.addColorStop(0.2, coreColorStr);
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(canvas);
}

class SwarmRenderer {
    constructor(scene, maxParticles = 2000, trailLength = 40) {
        this.scene = scene;
        this.trailLength = trailLength;
        this.history = [];

        this.pointGeo = new THREE.BufferGeometry();
        this.pointGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxParticles * 3), 3));
        const pointMat = new THREE.PointsMaterial({
            map: createGlowTexture('rgba(255, 0, 255, 0.9)'),
            size: 1.2, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false
        });
        this.points = new THREE.Points(this.pointGeo, pointMat);
        this.scene.add(this.points);

        this.lineGeo = new THREE.BufferGeometry();
        this.maxLineVertices = maxParticles * 2 * (trailLength - 1);
        this.lineGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.maxLineVertices * 3), 3));
        const lineMat = new THREE.LineBasicMaterial({
            color: 0x00ffff,
            blending: THREE.AdditiveBlending, transparent: true, opacity: 0.35
        });
        this.lines = new THREE.LineSegments(this.lineGeo, lineMat);
        this.scene.add(this.lines);
    }

    update(pointsArray, showPart, showTraj) {
        const count = pointsArray.length;
        const pos = this.points.geometry.attributes.position.array;
        const linePos = this.lines.geometry.attributes.position.array;
        let lineIdx = 0;

        for (let i = 0; i < count; i++) {
            const p = pointsArray[i];

            pos[i * 3] = p[0]; pos[i * 3 + 1] = p[1]; pos[i * 3 + 2] = p[2];

            if (!this.history[i]) this.history[i] = [];
            this.history[i].push(new THREE.Vector3(p[0], p[1], p[2]));
            if (this.history[i].length > this.trailLength) this.history[i].shift();

            const hist = this.history[i];
            for (let j = 0; j < hist.length - 1; j++) {
                linePos[lineIdx++] = hist[j].x;   linePos[lineIdx++] = hist[j].y;   linePos[lineIdx++] = hist[j].z;
                linePos[lineIdx++] = hist[j+1].x; linePos[lineIdx++] = hist[j+1].y; linePos[lineIdx++] = hist[j+1].z;
            }
        }

        this.points.geometry.setDrawRange(0, count);
        this.points.geometry.attributes.position.needsUpdate = true;
        this.points.visible = showPart;

        this.lines.geometry.setDrawRange(0, lineIdx / 3);
        this.lines.geometry.attributes.position.needsUpdate = true;
        this.lines.visible = showTraj;
    }

    clear() {
        this.history = [];
        this.points.geometry.setDrawRange(0, 0);
        this.lines.geometry.setDrawRange(0, 0);
    }
}

function initThreeJS() {
    const container = document.getElementById('canvas-container');
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.02);

    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(30, 20, 40);

    renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.toneMapping = THREE.ReinhardToneMapping;
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.05;
    controls.autoRotate = true; controls.autoRotateSpeed = 0.5;

    swarmRenderer = new SwarmRenderer(scene, 2000, 30);

    const renderScene = new RenderPass(scene, camera);
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 2.2, 0.5, 0.1);
    composer = new EffectComposer(renderer);
    composer.addPass(renderScene);
    composer.addPass(bloomPass);

    window.addEventListener('resize', onWindowResize, false);
    animate();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight); composer.setSize(window.innerWidth, window.innerHeight);
}

function animate() { requestAnimationFrame(animate); controls.update(); composer.render(); }

toggleTraj.addEventListener('change', () => { if(swarmRenderer) swarmRenderer.lines.visible = toggleTraj.checked; });
togglePart.addEventListener('change', () => { if(swarmRenderer) swarmRenderer.points.visible = togglePart.checked; });

const sliderParticles = document.getElementById('num-particles');
const sliderSteps = document.getElementById('steps-per-emit');
sliderParticles.addEventListener('input', () => { document.getElementById('val-particles').innerText = sliderParticles.value; });
sliderSteps.addEventListener('input', () => { document.getElementById('val-steps').innerText = sliderSteps.value; });

const statusEl = document.getElementById('status-bar');
const systemSelect = document.getElementById('system-select');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const paramContainer = document.getElementById('parameters-container');

function setStatus(msg) { statusEl.innerText = `Status: ${msg}`; }

async function loadSystems() {
    try {
        const res = await fetch('/api/systems'); const data = await res.json();
        if (data.success) {
            systemSelect.innerHTML = '<option value="">-- Select a System --</option>';
            data.systems.forEach(sys => {
                const opt = document.createElement('option'); opt.value = sys.id; opt.textContent = `[${sys.category}] ${sys.name}`;
                systemSelect.appendChild(opt);
            });
        }
    } catch (e) { setStatus('Failed to load systems API'); }
}

systemSelect.addEventListener('change', async (e) => {
    currentSystemId = e.target.value; paramContainer.innerHTML = ''; if (!currentSystemId) return;
    const res = await fetch(`/api/systems/${currentSystemId}`); const data = await res.json();
    if (data.success && data.system.parameters) {
        Object.entries(data.system.parameters).forEach(([key, val]) => {
            const isFloat = !Number.isInteger(val); const step = isFloat ? 0.01 : 1;
            const min = isFloat ? Math.min(0, val - Math.abs(val)) : 0; const max = isFloat ? Math.max(1, val * 3) : val * 3 + 10;
            const html = `
                <div class="control-group">
                    <label>${key}</label>
                    <div class="slider-container">
                        <input type="range" id="param-${key}" data-key="${key}" min="${min}" max="${max}" step="${step}" value="${val}">
                        <span class="param-value" id="val-${key}">${val}</span>
                    </div>
                </div>
            `;
            paramContainer.insertAdjacentHTML('beforeend', html);
            setTimeout(() => {
                document.getElementById(`param-${key}`).addEventListener('input', (ev) => {
                    const newVal = parseFloat(ev.target.value); document.getElementById(`val-${key}`).innerText = newVal.toFixed(isFloat ? 2 : 0);
                    if (streamId) socket.emit('update_parameters', { stream_id: streamId, parameters: { [key]: newVal } });
                });
            }, 0);
        });
    }
});

btnStart.addEventListener('click', () => {
    if (!currentSystemId) return alert("Please select a system first!");
    streamId = 'stream_' + Math.random().toString(36).substr(2, 9);
    swarmRenderer.clear();

    const initialParams = {};
    document.querySelectorAll('input[type="range"]').forEach(s => initialParams[s.dataset.key] = parseFloat(s.value));

    const numParticles = parseInt(sliderParticles.value);
    const stepsPerEmit = parseInt(sliderSteps.value);
    socket.emit('start_stream', { stream_id: streamId, system_id: currentSystemId, time_step: 0.01, steps_per_emit: stepsPerEmit, update_interval: 0.05, num_particles: numParticles, parameters: initialParams });
    btnStart.style.display = 'none'; btnStop.style.display = 'block'; systemSelect.disabled = true;
});

btnStop.addEventListener('click', () => {
    if (streamId) socket.emit('stop_stream', { stream_id: streamId });
    btnStart.style.display = 'block'; btnStop.style.display = 'none'; systemSelect.disabled = false; streamId = null; setStatus('Simulation stopped');
});

socket.on('connect', () => { setStatus('Connected to Backend'); });
socket.on('disconnect', () => { setStatus('Disconnected'); });
socket.on('stream_started', (data) => { setStatus(`Streaming ${data.system}...`); });
socket.on('trajectory_update', (data) => {
    if (data.stream_id !== streamId) return;
    const pointsArray = data.is_scatter ? data.point : [data.point];
    swarmRenderer.update(pointsArray, togglePart.checked, toggleTraj.checked);
});

initThreeJS(); loadSystems();
