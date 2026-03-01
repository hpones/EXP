// --- REFERENCIAS DOM ---
const video        = document.getElementById('video');
const glcanvas     = document.getElementById('glcanvas');
const canvas       = document.getElementById('canvas');
const filterSelect = document.getElementById('filterSelect');
const captureBtn   = document.getElementById('capture-button');
const recordBtn    = document.getElementById('record-button');
const pauseBtn     = document.getElementById('pause-button');
const stopBtn      = document.getElementById('stop-button');
const fullscreenBtn    = document.getElementById('fullscreen-button');
const filterBtn        = document.getElementById('filter-button');
const filtersDropdown  = document.getElementById('filters-dropdown');
const gallery          = document.getElementById('gallery');
const controls         = document.getElementById('controls');
const recordingControls = document.getElementById('recording-controls');
const cameraContainer  = document.getElementById('camera-container');

// --- ESTADO GENERAL ---
let currentStream;
let mediaRecorder;
let chunks = [];
let isRecording = false;
let isPaused    = false;
let selectedFilter = 'none';
let currentCameraDeviceId = null;
let currentFacingMode     = null;
let mediaCounter = 0;
let availableCameraDevices = [];

// --- WEBGL ---
let gl;
let program;
let positionBuffer;
let texCoordBuffer;
let videoTexture;
let mpOutputTexture;
let filterTypeLocation;
let timeLocation;
let flipXLocation;
let colorShiftUniformLocation;
let bassAmpUniformLocation;
let midAmpUniformLocation;
let highAmpUniformLocation;

// --- AUDIO ---
let audioContext;
let analyser;
let microphone;
let dataArray;
const AUDIO_THRESHOLD = 0.15;
let paletteIndex = 0;
const palettes = [
    [1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],
    [1.0,1.0,0.0],[1.0,0.0,1.0],[0.0,1.0,1.0],
];
// Buffer reutilizable para evitar allocations en cada frame
const colorShiftBuffer = new Float32Array(3);

// --- MEDIAPIPE ---
let selfieSegmentation;
let mpCamera;
let mpResults    = null;
let mpProcessing = false;
const mpCanvasCtx = canvas.getContext('2d');

// Canvases temporales reutilizables (evita crearlos cada frame en los filtros MP)
const tempCanvasA = document.createElement('canvas');
const tempCtxA    = tempCanvasA.getContext('2d');
const tempCanvasB = document.createElement('canvas');
const tempCtxB    = tempCanvasB.getContext('2d');

// --- SHADERS ---
const vsSource = `
    attribute vec4 a_position;
    attribute vec2 a_texCoord;
    uniform float u_flipX;
    varying vec2 v_texCoord;
    void main() {
        gl_Position = a_position;
        float coordX = 0.5 + (a_texCoord.x - 0.5) * u_flipX;
        v_texCoord = vec2(coordX, a_texCoord.y);
    }
`;

const fsSource = `
    precision mediump float;

    uniform sampler2D u_image;
    uniform int u_filterType;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform vec3 u_colorShift;
    uniform float u_bassAmp;
    uniform float u_midAmp;
    uniform float u_highAmp;
    uniform float u_aspectRatio;

    varying vec2 v_texCoord;

    const int FILTER_NONE               = 0;
    const int FILTER_GRAYSCALE          = 1;
    const int FILTER_INVERT             = 2;
    const int FILTER_SEPIA              = 3;
    const int FILTER_ECO_PINK           = 4;
    const int FILTER_WEIRD              = 5;
    const int FILTER_GLOW_OUTLINE       = 6;
    const int FILTER_ANGELICAL_GLITCH   = 7;
    const int FILTER_AUDIO_COLOR_SHIFT  = 8;
    const int FILTER_MODULAR_COLOR_SHIFT = 9;
    const int FILTER_KALEIDOSCOPE       = 10;
    const int FILTER_MIRROR             = 11;
    const int FILTER_FISHEYE            = 12;
    const int FILTER_RECUERDO           = 13;
    const int FILTER_GLITCH2            = 14;
    const int FILTER_VHS                = 15;

    float random(vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
    }

    float brightness(vec3 c) {
        return dot(c, vec3(0.299, 0.587, 0.114));
    }

    vec2 fisheye(vec2 uv, float strength) {
        vec2 c  = uv * 2.0 - 1.0;
        float r = length(c);
        float theta = atan(c.y, c.x);
        float rd = (strength > 0.0) ? r / (1.0 - strength * r) : r * (1.0 + strength * r);
        return (vec2(rd * cos(theta), rd * sin(theta)) + 1.0) * 0.5;
    }

    void main() {
        vec2 texCoord   = v_texCoord;
        vec4 color      = texture2D(u_image, texCoord);
        vec3 finalColor = color.rgb;
        float alpha     = color.a;

        if (u_filterType == FILTER_GRAYSCALE) {
            float lum = color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
            finalColor = vec3(lum);

        } else if (u_filterType == FILTER_INVERT) {
            finalColor = 1.0 - finalColor;

        } else if (u_filterType == FILTER_SEPIA) {
            float sr = color.r, sg = color.g, sb = color.b;
            finalColor = clamp(vec3(
                sr*0.393 + sg*0.769 + sb*0.189,
                sr*0.349 + sg*0.686 + sb*0.168,
                sr*0.272 + sg*0.534 + sb*0.131
            ), 0.0, 1.0);

        } else if (u_filterType == FILTER_ECO_PINK) {
            float bri = (color.r + color.g + color.b) / 3.0;
            if (bri < 0.3137) {
                finalColor = vec3(
                    min(1.0, color.r + 0.3137),
                    max(0.0, color.g - 0.1961),
                    min(1.0, color.b + 0.3922)
                );
            }

        } else if (u_filterType == FILTER_WEIRD) {
            float bri = (color.r + color.g + color.b) / 3.0;
            if      (bri > 0.7058) finalColor = vec3(color.b, color.r, color.g);
            else if (bri < 0.3921) finalColor *= 0.5;

        } else if (u_filterType == FILTER_GLOW_OUTLINE) {
            vec2 px = 1.0 / u_resolution;
            float df = 0.005;
            vec4 up = texture2D(u_image, texCoord + vec2( sin(texCoord.y*100.0)*df,  px.y+cos(texCoord.x*100.0)*df));
            vec4 dn = texture2D(u_image, texCoord + vec2( cos(texCoord.y*100.0)*df, -px.y+sin(texCoord.x*100.0)*df));
            vec4 lt = texture2D(u_image, texCoord + vec2(-px.x+sin(texCoord.y*100.0)*df, cos(texCoord.x*100.0)*df));
            vec4 rt = texture2D(u_image, texCoord + vec2( px.x+cos(texCoord.y*100.0)*df, sin(texCoord.x*100.0)*df));
            float diff = abs(color.r-up.r)+abs(color.r-dn.r)+abs(color.r-lt.r)+abs(color.r-rt.r);
            float edge = smoothstep(0.01, 0.1, diff);
            float glow = smoothstep(0.7, 1.0, brightness(color.rgb)) * 0.5;
            finalColor = mix(finalColor + glow, vec3(1.0), edge);

        } else if (u_filterType == FILTER_ANGELICAL_GLITCH) {
            vec2 uv  = texCoord;
            vec4 col = texture2D(u_image, uv);
            col.rgb *= 1.3;
            float b  = brightness(col.rgb);
            vec2 dist = vec2(
                (random(uv + vec2(sin(u_time*0.1), cos(u_time*0.1))) - 0.5) * 0.1,
                (random(uv + vec2(cos(u_time*0.1), sin(u_time*0.1))) - 0.5) * 0.1
            );
            vec4 distorted = texture2D(u_image, uv + dist);
            if (b > 0.5) {
                vec3 gc = vec3(0.5+0.3*sin(u_time*2.0), 0.2+0.5*cos(u_time*1.5), 0.6+0.4*sin(u_time*3.0));
                finalColor = mix(distorted.rgb, gc, 0.5);
            } else {
                finalColor = distorted.rgb;
            }
            alpha = col.a;

        } else if (u_filterType == FILTER_AUDIO_COLOR_SHIFT) {
            finalColor = mod(color.rgb + u_colorShift, 1.0);

        } else if (u_filterType == FILTER_MODULAR_COLOR_SHIFT) {
            const vec3 p0 = vec3(0.3137, 0.4706, 0.7059);
            const vec3 p1 = vec3(0.3922, 0.7059, 0.7843);
            const vec3 p2 = vec3(0.4706, 0.5882, 1.0);
            float bv = (color.r + color.g + color.b) / 3.0;
            if      (bv > 0.6667) finalColor = clamp(mix(color.rgb, p2, u_highAmp), 0.0, 1.0);
            else if (bv > 0.3922) finalColor = clamp(mix(color.rgb, p1, u_midAmp),  0.0, 1.0);
            else                  finalColor = clamp(mix(color.rgb, p0, u_bassAmp), 0.0, 1.0);

        } else if (u_filterType == FILTER_KALEIDOSCOPE) {
            vec2 c  = texCoord - 0.5;
            float r = length(c);
            float angle = abs(mod(atan(c.y, c.x), radians(60.0)) - radians(30.0));
            finalColor = texture2D(u_image, vec2(r*cos(angle), r*sin(angle)) + 0.5).rgb;

        } else if (u_filterType == FILTER_MIRROR) {
            vec2 uv = texCoord;
            if (uv.x > 0.5) uv.x = 1.0 - uv.x;
            finalColor = texture2D(u_image, uv).rgb;

        } else if (u_filterType == FILTER_FISHEYE) {
            finalColor = texture2D(u_image, fisheye(texCoord, 0.8)).rgb;

        } else if (u_filterType == FILTER_RECUERDO) {
            vec2 uv    = texCoord;
            float wob  = sin(u_time * 0.8) * 0.01;
            uv.x += sin(uv.y * 20.0 + u_time * 3.0) * wob;
            uv.y += cos(uv.x * 22.0 + u_time * 2.5) * wob;
            float grain = random(uv * u_time) * 0.1;
            vec2 dir    = uv - 0.5;
            float focus = smoothstep(0.4, 0.2, length(dir) * 2.5);
            float bstr  = (1.0 - focus) * 0.015;
            vec4 blurred = vec4(0.0);
            const float N = 8.0;
            for (float i = 0.0; i < N; i++) blurred += texture2D(u_image, uv + dir * bstr * (i / N));
            blurred /= N;
            vec3 mixed = mix(blurred, texture2D(u_image, uv), focus).rgb * vec3(1.1, 1.05, 0.9);
            float mr = mixed.r, mg = mixed.g, mb = mixed.b;
            vec3 sep = clamp(vec3(mr*0.393+mg*0.769+mb*0.189, mr*0.349+mg*0.686+mb*0.168, mr*0.272+mg*0.534+mb*0.131), 0.0, 1.0);
            finalColor  = clamp(mix(mixed, sep, 0.5) + grain, 0.0, 1.0);

        } else if (u_filterType == FILTER_GLITCH2) {
            vec2 uv  = texCoord;
            float jt = floor(u_time * 12.0);
            float jr  = random(vec2(jt*0.017, jt*0.031));
            float jr2 = random(vec2(jt*0.053, jt*0.072));
            float jx = 0.0, jy = 0.0;
            if (jr > 0.75) { jx = (jr-0.75)*0.18*sign(jr2-0.5); jy = (jr2-0.5)*0.06; }
            uv = fract(uv + vec2(jx, jy));
            float fz  = smoothstep(0.08,0.14,uv.y)*(1.0-smoothstep(0.50,0.56,uv.y));
            float fzx = smoothstep(0.20,0.30,uv.x)*(1.0-smoothstep(0.70,0.80,uv.x));
            float fm  = fz * fzx;
            float fby = floor(uv.y*80.0)/80.0;
            float fg  = random(vec2(fby*3.1, floor(u_time*20.0)*0.7));
            float fs  = (fg > 0.70 && fm > 0.3) ? (fg-0.70)*0.35 : 0.0;
            uv.x = fract(uv.x + sin(uv.y*60.0+u_time*18.0)*0.025*fm + fs);
            uv.y = fract(uv.y + cos(uv.x*45.0+u_time*14.0)*0.018*fm);
            float by  = floor(uv.y*40.0)/40.0;
            float gs  = random(vec2(by, floor(u_time*12.0)));
            float gs2 = random(vec2(by+0.3, floor(u_time*8.0)));
            if (gs > 0.65) uv.x = fract(uv.x + (gs-0.65)*1.2);
            float ab = 0.018 + gs2*0.025 + abs(jx)*0.3;
            vec3 aber = vec3(
                texture2D(u_image, vec2(uv.x+ab, uv.y)).r,
                texture2D(u_image, uv).g,
                texture2D(u_image, vec2(uv.x-ab, uv.y)).b
            );
            float lum  = brightness(aber);
            float wave = sin(uv.y*18.0+u_time*5.0)*0.5+0.5;
            vec3 pal   = mix(vec3(1.0,0.05,0.75), vec3(0.0,1.0,0.2), wave);
            vec3 col   = mix(aber*0.3, pal, smoothstep(0.2,0.8,lum));
            float sln  = mod(floor(uv.y*u_resolution.y), 3.0);
            float ns   = random(vec2(uv.x*100.0, floor(u_time*20.0)+uv.y*50.0));
            if (sln == 0.0 && ns > 0.6) col = mix(col, pal.brg, 0.8);
            float flash = (jr > 0.88) ? (jr-0.88)*4.0 : 0.0;
            finalColor  = clamp(mix(col, vec3(1.0), flash*0.5), 0.0, 1.0);

        } else if (u_filterType == FILTER_VHS) {
            vec2 vhs_uv  = texCoord;
            vec2 vhs_pix = (floor(vhs_uv * vec2(320.0,240.0)) + 0.5) / vec2(320.0,240.0);
            float vhs_sl = mod(floor(vhs_uv.y * u_resolution.y), 2.0);
            float vhs_sf = 1.0 - vhs_sl * 0.18;
            float vhs_cs = 2.0 / u_resolution.x;
            vec3 vhs_c   = vec3(
                texture2D(u_image, vec2(vhs_pix.x+vhs_cs, vhs_pix.y)).r,
                texture2D(u_image, vhs_pix).g,
                texture2D(u_image, vec2(vhs_pix.x-vhs_cs, vhs_pix.y)).b
            );
            float vhs_lum   = dot(vhs_c, vec3(0.299,0.587,0.114));
            float vhs_grain = random(vhs_uv + vec2(u_time*0.017, u_time*0.031)) * 0.12;
            vhs_c += vhs_grain * smoothstep(0.5, 0.0, vhs_lum);
            float vhs_jt   = floor(u_time * 15.0);
            float vhs_js   = random(vec2(floor(vhs_uv.y*u_resolution.y*0.25), vhs_jt));
            float vhs_ja   = step(0.92, vhs_js);
            float vhs_jamt = (random(vec2(vhs_js, vhs_jt*0.1)) - 0.5) * 0.03;
            vec2  vhs_juv  = clamp(vec2(vhs_pix.x+vhs_jamt*vhs_ja, vhs_pix.y), 0.0, 1.0);
            vec3  vhs_jc   = vec3(
                texture2D(u_image, vec2(vhs_juv.x+vhs_cs, vhs_juv.y)).r,
                texture2D(u_image, vhs_juv).g,
                texture2D(u_image, vec2(vhs_juv.x-vhs_cs, vhs_juv.y)).b
            );
            finalColor = clamp(mix(vhs_c, vhs_jc, vhs_ja) * vhs_sf, 0.0, 1.0);
        }

        gl_FragColor = vec4(finalColor, alpha);
    }
`;

// --- INIT WEBGL ---
function compileShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function createProgram(gl, vs, fs) {
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.error('Program error:', gl.getProgramInfoLog(prog));
        gl.deleteProgram(prog);
        return null;
    }
    return prog;
}

function setupTexture(gl) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    return tex;
}

function initWebGL() {
    gl = glcanvas.getContext('webgl', { preserveDrawingBuffer: true });
    if (!gl) { alert('Tu navegador no soporta WebGL.'); return; }

    program = createProgram(gl,
        compileShader(gl, vsSource, gl.VERTEX_SHADER),
        compileShader(gl, fsSource, gl.FRAGMENT_SHADER)
    );
    gl.useProgram(program);

    program.positionLocation    = gl.getAttribLocation(program, 'a_position');
    program.texCoordLocation    = gl.getAttribLocation(program, 'a_texCoord');
    program.imageLocation       = gl.getUniformLocation(program, 'u_image');
    program.resolutionLocation  = gl.getUniformLocation(program, 'u_resolution');
    program.aspectRatioLocation = gl.getUniformLocation(program, 'u_aspectRatio');
    filterTypeLocation          = gl.getUniformLocation(program, 'u_filterType');
    timeLocation                = gl.getUniformLocation(program, 'u_time');
    colorShiftUniformLocation   = gl.getUniformLocation(program, 'u_colorShift');
    flipXLocation               = gl.getUniformLocation(program, 'u_flipX');
    bassAmpUniformLocation      = gl.getUniformLocation(program, 'u_bassAmp');
    midAmpUniformLocation       = gl.getUniformLocation(program, 'u_midAmp');
    highAmpUniformLocation      = gl.getUniformLocation(program, 'u_highAmp');

    gl.enableVertexAttribArray(program.positionLocation);
    gl.enableVertexAttribArray(program.texCoordLocation);

    positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), gl.STATIC_DRAW);

    texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0,1,1,1,0,0,0,0,1,1,1,0]), gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.vertexAttribPointer(program.positionLocation, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.vertexAttribPointer(program.texCoordLocation, 2, gl.FLOAT, false, 0, 0);

    videoTexture    = setupTexture(gl);
    mpOutputTexture = setupTexture(gl);

    gl.uniform1i(program.imageLocation, 0);
    gl.uniform1i(filterTypeLocation, 0);
    gl.clearColor(0, 0, 0, 1);
}

// --- MEDIAPIPE ---
function initMediaPipe() {
    selfieSegmentation = new SelfieSegmentation({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
    });
    selfieSegmentation.setOptions({ modelSelection: 1 });
    selfieSegmentation.onResults((results) => {
        mpResults    = results;
        mpProcessing = false;
    });
}

function syncTempCanvases(w, h) {
    if (tempCanvasA.width !== w || tempCanvasA.height !== h) {
        tempCanvasA.width = w;  tempCanvasA.height = h;
        tempCanvasB.width = w;  tempCanvasB.height = h;
    }
}

// --- CÁMARA ---
async function listCameras() {
    try {
        const perm = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        perm.getTracks().forEach(t => t.stop());
        const devices = await navigator.mediaDevices.enumerateDevices();
        availableCameraDevices = devices.filter(d => d.kind === 'videoinput');
        if (availableCameraDevices.length > 0) {
            currentCameraDeviceId = availableCameraDevices[0].deviceId;
            startCamera(currentCameraDeviceId);
        } else {
            alert('No se encontraron dispositivos de cámara.');
        }
    } catch (err) {
        console.error('Error listando cámaras:', err);
        alert('Error al acceder a la cámara. Revisa los permisos.');
    }
}

async function startCamera(deviceId) {
    if (currentStream) currentStream.getTracks().forEach(t => t.stop());
    try {
        currentStream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: true
        });
        video.srcObject = currentStream;

        const settings = currentStream.getVideoTracks()[0].getSettings();
        currentCameraDeviceId = settings.deviceId;
        currentFacingMode     = settings.facingMode || 'unknown';

        // Audio
        const audioTracks = currentStream.getAudioTracks();
        if (audioTracks.length > 0) {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser     = audioContext.createAnalyser();
                analyser.fftSize = 256;
                analyser.smoothingTimeConstant = 0.7;
                dataArray    = new Uint8Array(analyser.frequencyBinCount);
            }
            if (microphone) microphone.disconnect();
            microphone = audioContext.createMediaStreamSource(currentStream);
            microphone.connect(analyser);
        } else {
            if (microphone) { microphone.disconnect(); microphone = null; }
            if (analyser)   { analyser.disconnect();   analyser   = null; }
        }

        video.onloadedmetadata = () => {
            video.play();
            const w = video.videoWidth, h = video.videoHeight;
            if (glcanvas.width !== w || glcanvas.height !== h) {
                glcanvas.width = w;  glcanvas.height = h;
                canvas.width   = w;  canvas.height   = h;
                syncTempCanvases(w, h);
                if (gl) {
                    gl.viewport(0, 0, w, h);
                    gl.uniform2f(program.resolutionLocation, w, h);
                    gl.uniform1f(program.aspectRatioLocation, w / h);
                }
            }
            if (!gl)                initWebGL();
            if (!selfieSegmentation) initMediaPipe();
            if (!mpCamera) {
                mpCamera = new Camera(video, {
                    onFrame: async () => {
                        if (!mpProcessing) {
                            mpProcessing = true;
                            await selfieSegmentation.send({ image: video });
                        }
                    },
                    width: w, height: h,
                });
                mpCamera.start();
            }
            drawVideoFrame();
        };
    } catch (err) {
        console.error('startCamera error:', err);
        alert('No se pudo acceder a la cámara/micrófono. Error: ' + err.name);
    }
}

// --- BUCLE DE RENDERIZADO ---
const MP_FILTERS = new Set(['whiteGlow','blackBg','whiteBg','blur','static-silhouette','echo-visual']);

const FILTER_INDEX = {
    'grayscale':1, 'invert':2, 'sepia':3, 'eco-pink':4, 'weird':5,
    'glow-outline':6, 'angelical-glitch':7, 'audio-color-shift':8,
    'modular-color-shift':9, 'kaleidoscope':10, 'mirror':11,
    'fisheye':12, 'recuerdo':13, 'glitch2':14, 'vhs':15,
};

function drawVideoFrame() {
    if (!gl || !program || !video.srcObject || video.readyState !== video.HAVE_ENOUGH_DATA) {
        requestAnimationFrame(drawVideoFrame);
        return;
    }

    // Audio color shift
    if (analyser && dataArray && selectedFilter === 'audio-color-shift') {
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
        if ((sum / dataArray.length) / 255 > AUDIO_THRESHOLD) {
            paletteIndex = (paletteIndex + 1) % palettes.length;
        }
    }

    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform1f(timeLocation, performance.now() / 1000.0);

    const isFront = (currentFacingMode === 'user' || currentFacingMode === 'unknown');
    const isMP    = MP_FILTERS.has(selectedFilter);

    // Filtros WebGL: GPU maneja el espejo. Filtros MediaPipe: canvas 2D maneja el espejo, GPU no.
    gl.uniform1f(flipXLocation, (!isMP && isFront) ? -1.0 : 1.0);

    if (isMP && mpResults && mpResults.segmentationMask && mpResults.image) {
        const w = canvas.width, h = canvas.height;
        syncTempCanvases(w, h);

        mpCanvasCtx.save();
        if (isFront) {
            mpCanvasCtx.translate(w, 0);
            mpCanvasCtx.scale(-1, 1);
        }
        mpCanvasCtx.clearRect(0, 0, w, h);

        switch (selectedFilter) {
            case 'whiteGlow': {
                mpCanvasCtx.drawImage(mpResults.image, 0, 0, w, h);
                tempCtxA.clearRect(0, 0, w, h);
                tempCtxA.fillStyle = 'red';
                tempCtxA.fillRect(0, 0, w, h);
                tempCtxA.globalCompositeOperation = 'destination-in';
                tempCtxA.drawImage(mpResults.segmentationMask, 0, 0, w, h);
                tempCtxA.globalCompositeOperation = 'source-over';
                mpCanvasCtx.drawImage(tempCanvasA, 0, 0);
                break;
            }
            case 'blackBg':
            case 'whiteBg': {
                mpCanvasCtx.drawImage(mpResults.image, 0, 0, w, h);
                tempCtxA.clearRect(0, 0, w, h);
                tempCtxA.fillStyle = selectedFilter === 'blackBg' ? 'black' : 'white';
                tempCtxA.fillRect(0, 0, w, h);
                tempCtxA.globalCompositeOperation = 'destination-in';
                tempCtxA.drawImage(mpResults.segmentationMask, 0, 0, w, h);
                tempCtxA.globalCompositeOperation = 'source-over';
                mpCanvasCtx.drawImage(tempCanvasA, 0, 0);
                break;
            }
            case 'blur': {
                mpCanvasCtx.drawImage(mpResults.image, 0, 0, w, h);
                tempCtxA.clearRect(0, 0, w, h);
                tempCtxA.filter = 'blur(18px)';
                tempCtxA.drawImage(mpResults.image, -30, -30, w+60, h+60);
                tempCtxA.filter = 'none';
                tempCtxA.globalCompositeOperation = 'destination-in';
                tempCtxA.drawImage(mpResults.segmentationMask, 0, 0, w, h);
                tempCtxA.globalCompositeOperation = 'source-over';
                mpCanvasCtx.drawImage(tempCanvasA, 0, 0);
                break;
            }
            case 'static-silhouette': {
                mpCanvasCtx.drawImage(mpResults.image, 0, 0, w, h);
                const imgData = tempCtxA.createImageData(w, h);
                const d = imgData.data;
                for (let i = 0; i < d.length; i += 4) {
                    const v = Math.random() * 255 | 0;
                    d[i] = d[i+1] = d[i+2] = v; d[i+3] = 255;
                }
                tempCtxA.putImageData(imgData, 0, 0);
                tempCtxA.globalCompositeOperation = 'destination-in';
                tempCtxA.drawImage(mpResults.segmentationMask, 0, 0, w, h);
                tempCtxA.globalCompositeOperation = 'source-over';
                mpCanvasCtx.drawImage(tempCanvasA, 0, 0);
                break;
            }
            case 'echo-visual': {
                mpCanvasCtx.drawImage(mpResults.image, 0, 0, w, h);
                tempCtxA.clearRect(0, 0, w, h);
                tempCtxA.fillStyle = 'rgb(10,10,10)';
                tempCtxA.fillRect(0, 0, w, h);
                tempCtxA.globalCompositeOperation = 'destination-in';
                tempCtxA.drawImage(mpResults.segmentationMask, 0, 0, w, h);
                tempCtxA.globalCompositeOperation = 'source-over';
                const echoCount = 7;
                for (let i = echoCount; i >= 1; i--) {
                    mpCanvasCtx.globalAlpha = 0.30 + 0.50 * (echoCount - i) / (echoCount - 1);
                    mpCanvasCtx.drawImage(tempCanvasA, 22*i, 6*i);
                }
                mpCanvasCtx.globalAlpha = 1.0;
                tempCtxB.clearRect(0, 0, w, h);
                tempCtxB.drawImage(mpResults.image, 0, 0, w, h);
                tempCtxB.globalCompositeOperation = 'destination-in';
                tempCtxB.drawImage(mpResults.segmentationMask, 0, 0, w, h);
                tempCtxB.globalCompositeOperation = 'source-over';
                mpCanvasCtx.drawImage(tempCanvasB, 0, 0);
                break;
            }
        }

        mpCanvasCtx.globalAlpha = 1.0;
        mpCanvasCtx.globalCompositeOperation = 'source-over';
        mpCanvasCtx.restore();

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, mpOutputTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
        gl.uniform1i(filterTypeLocation, 0);

    } else {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, videoTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

        if (selectedFilter === 'audio-color-shift') {
            const p = palettes[paletteIndex];
            colorShiftBuffer[0] = p[0]; colorShiftBuffer[1] = p[1]; colorShiftBuffer[2] = p[2];
            gl.uniform3fv(colorShiftUniformLocation, colorShiftBuffer);
        } else if (selectedFilter === 'modular-color-shift') {
            const t = performance.now() / 1000.0;
            gl.uniform1f(bassAmpUniformLocation, mapValue(Math.sin(t*0.8), -1,1, 0.0,2.0));
            gl.uniform1f(midAmpUniformLocation,  mapValue(Math.sin(t*1.2 + Math.PI/3), -1,1, 0.0,1.5));
            gl.uniform1f(highAmpUniformLocation, mapValue(Math.sin(t*1.5 + Math.PI*2/3), -1,1, 0.0,2.5));
        }

        gl.uniform1i(filterTypeLocation, FILTER_INDEX[selectedFilter] ?? 0);
    }

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(drawVideoFrame);
}

function mapValue(v, iMin, iMax, oMin, oMax) {
    return (v - iMin) * (oMax - oMin) / (iMax - iMin) + oMin;
}

// --- EVENTOS ---
captureBtn.addEventListener('click', () => {
    if (!gl || !glcanvas.width || !glcanvas.height) return;
    const img = new Image();
    img.src = glcanvas.toDataURL('image/png');
    img.onload = () => addToGallery(img, 'img');
});

let recordStream = null;

recordBtn.addEventListener('click', () => {
    if (isRecording) return;
    chunks = [];

    // FPS fijo: evita que el encoder acople su ritmo al bucle de render
    recordStream = glcanvas.captureStream(30);
    const audioTracks = currentStream.getAudioTracks();
    if (audioTracks.length > 0) recordStream.addTrack(audioTracks[0]);

    mediaRecorder = new MediaRecorder(recordStream, { mimeType: 'video/webm; codecs=vp8' });
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    mediaRecorder.onstop = () => {
        // Liberar todos los tracks del stream de grabación al terminar
        recordStream.getTracks().forEach(t => t.stop());
        recordStream = null;
        const url = URL.createObjectURL(new Blob(chunks, { type: 'video/webm' }));
        const vid = document.createElement('video');
        vid.src = url; vid.controls = true;
        addToGallery(vid, 'video');
    };
    mediaRecorder.start();
    isRecording = true;
    controls.style.display = 'none';
    recordingControls.style.display = 'flex';
});

pauseBtn.addEventListener('click', () => {
    if (isPaused) { mediaRecorder.resume(); pauseBtn.textContent = 'Pausa'; }
    else          { mediaRecorder.pause();  pauseBtn.textContent = 'Reanudar'; }
    isPaused = !isPaused;
});

stopBtn.addEventListener('click', () => {
    mediaRecorder.stop();
    isRecording = false;
    isPaused = false;
    pauseBtn.textContent = 'Pausa';
    controls.style.display = 'flex';
    recordingControls.style.display = 'none';
});

filterBtn.addEventListener('click', () => {
    filtersDropdown.style.display = filtersDropdown.style.display === 'block' ? 'none' : 'block';
});

filterSelect.addEventListener('change', () => {
    selectedFilter = filterSelect.value;
    filtersDropdown.style.display = 'none';
});

fullscreenBtn.addEventListener('click', () => {
    if (!document.fullscreenElement) cameraContainer.requestFullscreen();
    else document.exitFullscreen();
});

// --- GALERÍA ---
function addToGallery(element, type) {
    mediaCounter++;
    const itemTitle = `${type === 'img' ? 'Foto' : 'Video'} ${mediaCounter}`;
    const ext       = type === 'img' ? '.png' : '.webm';
    const mimeType  = type === 'img' ? 'image/png' : 'video/webm';
    const fileName  = itemTitle.replace(/\s/g, '_') + ext;

    const container = document.createElement('div');
    container.className = 'gallery-item';
    container.dataset.title = itemTitle;

    const label = document.createElement('div');
    label.className = 'gallery-label';
    label.textContent = itemTitle;
    container.appendChild(label);
    container.appendChild(element);

    // Preview al hacer click
    container.addEventListener('click', () => {
        const win = document.createElement('div');
        win.className = 'preview-window';

        const closeBtn = document.createElement('span');
        closeBtn.className = 'close-preview-window-button';
        closeBtn.textContent = '✕';
        win.appendChild(closeBtn);

        let clone;
        if (type === 'img') {
            clone = new Image(); clone.src = element.src;
        } else {
            clone = document.createElement('video');
            clone.src = element.src; clone.controls = true; clone.autoplay = true;
        }
        win.appendChild(clone);

        const actions = document.createElement('div');
        actions.className = 'preview-actions';

        const dlBtn = document.createElement('button');
        dlBtn.textContent = 'Descargar';
        dlBtn.onclick = () => { const a = document.createElement('a'); a.href=element.src; a.download=fileName; a.click(); };
        actions.appendChild(dlBtn);

        if (navigator.share) {
            const shareBtn = document.createElement('button');
            shareBtn.textContent = 'Compartir';
            shareBtn.onclick = async () => {
                try {
                    const blob = await fetch(element.src).then(r => r.blob());
                    await navigator.share({ files:[new File([blob],fileName,{type:mimeType})], title:'Mi creación desde Experimental Camera' });
                } catch(e) {}
            };
            actions.appendChild(shareBtn);
        }

        const delBtn = document.createElement('button');
        delBtn.textContent = 'Eliminar';
        delBtn.onclick = () => { if (clone && type==='video') clone.pause(); win.remove(); container.remove(); };
        actions.appendChild(delBtn);
        win.appendChild(actions);

        closeBtn.addEventListener('click', () => { if (clone && type==='video') clone.pause(); win.remove(); });

        // Drag
        let px=0,py=0,mx=0,my=0;
        win.onmousedown = (e) => {
            e.preventDefault(); mx=e.clientX; my=e.clientY;
            document.onmouseup   = () => { document.onmouseup=null; document.onmousemove=null; };
            document.onmousemove = (e) => {
                px=mx-e.clientX; py=my-e.clientY; mx=e.clientX; my=e.clientY;
                win.style.top  = (win.offsetTop  - py)+'px';
                win.style.left = (win.offsetLeft - px)+'px';
            };
        };
        document.body.appendChild(win);
    });

    // Acciones en miniatura
    const actions = document.createElement('div');
    actions.className = 'gallery-actions';

    const dlBtn = document.createElement('button');
    dlBtn.textContent = 'Descargar';
    dlBtn.onclick = () => { const a=document.createElement('a'); a.href=element.src; a.download=fileName; a.click(); };
    actions.appendChild(dlBtn);

    if (navigator.share) {
        const shareBtn = document.createElement('button');
        shareBtn.textContent = 'Compartir';
        shareBtn.onclick = async () => {
            try {
                const blob = await fetch(element.src).then(r => r.blob());
                await navigator.share({ files:[new File([blob],fileName,{type:mimeType})], title:'Mi creación desde Experimental Camera' });
            } catch(e) {}
        };
        actions.appendChild(shareBtn);
    }

    const delBtn = document.createElement('button');
    delBtn.textContent = 'Eliminar';
    delBtn.onclick = () => { if (type==='video' && element.src.startsWith('blob:')) URL.revokeObjectURL(element.src); container.remove(); };
    actions.appendChild(delBtn);

    container.appendChild(actions);
    gallery.prepend(container);
}

// --- CAMBIO DE CÁMARA ---
let lastTap = 0;
const DBL_TAP_THRESHOLD = 300;

glcanvas.addEventListener('touchend', (e) => {
    const now = Date.now();
    const delta = now - lastTap;
    if (delta < DBL_TAP_THRESHOLD && delta > 0) { e.preventDefault(); toggleCamera(); }
    lastTap = now;
}, { passive: false });

glcanvas.addEventListener('dblclick', toggleCamera);

function toggleCamera() {
    if (availableCameraDevices.length < 2) { alert('Solo hay una cámara disponible.'); return; }
    const idx = availableCameraDevices.findIndex(d => d.deviceId === currentCameraDeviceId);
    startCamera(availableCameraDevices[(idx+1) % availableCameraDevices.length].deviceId);
}

// --- SWIPE PARA FILTROS ---
let touchStartX = 0, touchEndX = 0;
const SWIPE_THRESHOLD = 50;

glcanvas.addEventListener('touchstart', (e) => { touchStartX = e.touches[0].clientX; });
glcanvas.addEventListener('touchmove',  (e) => { touchEndX   = e.touches[0].clientX; });
glcanvas.addEventListener('touchend',   () => {
    const diff = touchEndX - touchStartX;
    if (Math.abs(diff) > SWIPE_THRESHOLD) {
        const opts = Array.from(filterSelect.querySelectorAll('option')).map(o => o.value);
        let idx = opts.indexOf(selectedFilter);
        idx = diff > 0
            ? (idx > 0 ? idx-1 : opts.length-1)
            : (idx < opts.length-1 ? idx+1 : 0);
        selectedFilter = opts[idx];
        filterSelect.value = selectedFilter;
    }
    touchStartX = 0; touchEndX = 0;
});

glcanvas.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); });

// --- TUTORIAL ---
(function() {
    const overlay = document.getElementById('tutorial-overlay');
    const skipBtn = document.getElementById('tutorial-skip');
    if (!overlay || !skipBtn) return;
    const KEY = 'exp_cam_tutorial_seen';
    if (localStorage.getItem(KEY)) { overlay.style.display = 'none'; return; }
    skipBtn.addEventListener('click', () => {
        overlay.style.display = 'none';
        try { localStorage.setItem(KEY, '1'); } catch(e) {}
    });
})();

listCameras();
