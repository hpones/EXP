let video = document.getElementById('video');
let glcanvas = document.getElementById('glcanvas');
let canvas = document.getElementById('canvas'); // Canvas 2D auxiliar para MediaPipe
let filterSelect = document.getElementById('filterSelect');
let captureBtn = document.getElementById('capture-button');
let recordBtn = document.getElementById('record-button');
let pauseBtn = document.getElementById('pause-button');
let stopBtn = document.getElementById('stop-button');
let fullscreenBtn = document.getElementById('fullscreen-button');
let filterBtn = document.getElementById('filter-button');
let filtersDropdown = document.getElementById('filters-dropdown');
let gallery = document.getElementById('gallery');
let controls = document.getElementById('controls');
let recordingControls = document.getElementById('recording-controls');
let cameraContainer = document.getElementById('camera-container');

let currentStream;
let mediaRecorder;
let chunks = [];
let isRecording = false;
let isPaused = false;
let selectedFilter = 'none';
let currentCameraDeviceId = null;
let currentFacingMode = null;
let mediaCounter = 0; // Contador para enumerar las fotos y videos

// --- VARIABLES Y CONFIGURACIÓN DE WEBG L ---
let gl; // Contexto WebGL
let program; // Programa de shaders
let positionBuffer; // Buffer para las posiciones de los vértices
let texCoordBuffer; // Buffer para las coordenadas de textura
let videoTexture; // Textura donde se cargará el fotograma del video (para filtros WebGL)
let mpOutputTexture; // Nueva textura para el output de MediaPipe (para filtros avanzados)
let filterTypeLocation; // Ubicación del uniform para el tipo de filtro
let timeLocation; // Ubicación del uniform para el tiempo (para efectos dinámicos)

// --- VARIABLES Y CONFIGURACIÓN DE AUDIO ---
let audioContext;
let analyser;
let microphone;
let dataArray;
const AUDIO_THRESHOLD = 0.15;

let paletteIndex = 0;
const palettes = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
];
let colorShiftUniformLocation;

// --- NUEVOS UNIFORMS PARA EL FILTRO MODULAR COLOR SHIFT ---
let bassAmpUniformLocation;
let midAmpUniformLocation;
let highAmpUniformLocation;

// --- VARIABLES Y CONFIGURACIÓN DE MEDIAPIPE ---
let selfieSegmentation;
let mpCamera;
let mpResults = null; // Para almacenar el último resultado de MediaPipe
let mpCanvasCtx = canvas.getContext('2d'); // Contexto 2D para el canvas auxiliar
let mpProcessing = false; // Bandera para controlar si MediaPipe está procesando

// Vertex Shader: define la posición de los vértices y las coordenadas de textura
const vsSource = `
    attribute vec4 a_position;
    attribute vec2 a_texCoord;

    varying vec2 v_texCoord;

    void main() {
        gl_Position = a_position;
        v_texCoord = a_texCoord;
    }
`;

// Fragment Shader: define el color de cada píxel, ahora con lógica de filtros
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

    // Enumeración de filtros (coincide con los índices en JavaScript)
    const int FILTER_NONE = 0;
    const int FILTER_GRAYSCALE = 1;
    const int FILTER_INVERT = 2;
    const int FILTER_SEPIA = 3;
    const int FILTER_ECO_PINK = 4;
    const int FILTER_WEIRD = 5;
    const int FILTER_GLOW_OUTLINE = 6;
    const int FILTER_ANGELICAL_GLITCH = 7;
    const int FILTER_AUDIO_COLOR_SHIFT = 8;
    const int FILTER_MODULAR_COLOR_SHIFT = 9;
    const int FILTER_KALEIDOSCOPE = 10;
    const int FILTER_MIRROR = 11;
    const int FILTER_FISHEYE = 12;
    const int FILTER_RECUERDO = 13; // Filtro "Recuerdo" (WebGL puro y post-proceso)
    const int FILTER_GLITCH2 = 14;  // Glitch 2: interferencia intensa rosas y verdes

    // Función para generar ruido básico
    float random(vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
    }
    
    // Función de brillo para detectar luz
    float brightness(vec3 color) {
        return dot(color, vec3(0.299, 0.587, 0.114));
    }

    // Rotar un punto alrededor del centro
    vec2 rotate2D(vec2 st, float angle) {
        st -= 0.5;
        st = mat2(cos(angle), -sin(angle), sin(angle), cos(angle)) * st;
        st += 0.5;
        return st;
    }

    // Efecto de ojo de pez
    vec2 fisheye(vec2 uv, float strength) {
        vec2 centered = uv * 2.0 - 1.0; // Coordenadas de -1 a 1
        float r = length(centered);
        float theta = atan(centered.y, centered.x);
        
        float r_distorted = r;
        if (strength > 0.0) { // Convexo
            r_distorted = r / (1.0 - strength * r);
        } else { // Cóncavo
            r_distorted = r * (1.0 + strength * r);
        }
        
        vec2 distorted_centered = vec2(r_distorted * cos(theta), r_distorted * sin(theta));
        return (distorted_centered + 1.0) * 0.5; // Devolver a 0-1
    }

    void main() {
        vec2 texCoord = v_texCoord;
        vec4 color = texture2D(u_image, texCoord);
        vec3 finalColor = color.rgb;
        float alpha = color.a;

        if (u_filterType == FILTER_GRAYSCALE) {
            float brightness = (color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722);
            finalColor = vec3(brightness);
        } else if (u_filterType == FILTER_INVERT) {
            finalColor = 1.0 - finalColor;
        } else if (u_filterType == FILTER_SEPIA) {
            float r = color.r;
            float g = color.g;
            float b = color.b;
            finalColor.r = (r * 0.393) + (g * 0.769) + (b * 0.189);
            finalColor.g = (r * 0.349) + (g * 0.686) + (b * 0.168);
            finalColor.b = (r * 0.272) + (g * 0.534) + (b * 0.131);
            finalColor = clamp(finalColor, 0.0, 1.0);
        } else if (u_filterType == FILTER_ECO_PINK) {
            float brightness = (color.r + color.g + color.b) / 3.0;
            if (brightness < 0.3137) {
                finalColor.r = min(1.0, color.r + (80.0/255.0));
                finalColor.g = max(0.0, color.g - (50.0/255.0));
                finalColor.b = min(1.0, color.b + (100.0/255.0));
            }
        } else if (u_filterType == FILTER_WEIRD) {
            float brightness = (color.r + color.g + color.b) / 3.0;
            if (brightness > 0.7058) {
                float temp_r = color.r;
                finalColor.r = color.b;
                finalColor.b = color.g;
                finalColor.g = temp_r;
            } else if (brightness < 0.3921) {
                finalColor *= 0.5;
            }
        } else if (u_filterType == FILTER_GLOW_OUTLINE) {
            vec2 onePixel = vec2(1.0, 1.0) / u_resolution;
            float distortionFactor = 0.005;
            vec2 offsetUp = vec2(sin(texCoord.y * 100.0) * distortionFactor, onePixel.y + cos(texCoord.x * 100.0) * distortionFactor);
            vec2 offsetDown = vec2(cos(texCoord.y * 100.0) * distortionFactor, -onePixel.y + sin(texCoord.x * 100.0) * distortionFactor);
            vec2 offsetLeft = vec2(-onePixel.x + sin(texCoord.y * 100.0) * distortionFactor, cos(texCoord.x * 100.0) * distortionFactor);
            vec2 offsetRight = vec2(onePixel.x + cos(texCoord.y * 100.0) * distortionFactor, sin(texCoord.x * 100.0) * distortionFactor);

            vec4 up = texture2D(u_image, texCoord + offsetUp);
            vec4 down = texture2D(u_image, texCoord + offsetDown);
            vec4 left = texture2D(u_image, texCoord + offsetLeft);
            vec4 right = texture2D(u_image, texCoord + offsetRight);

            float diff = abs(color.r - up.r) + abs(color.r - down.r) + abs(color.r - left.r) + abs(color.r - right.r);
            float edge = smoothstep(0.01, 0.1, diff);
            vec3 outlineColor = vec3(1.0);

            float brightness = dot(color.rgb, vec3(0.299, 0.587, 0.114));
            float glowFactor = smoothstep(0.7, 1.0, brightness) * 0.5;

            finalColor = mix(finalColor + glowFactor, outlineColor, edge);
        } else if (u_filterType == FILTER_ANGELICAL_GLITCH) {
            vec2 uv = texCoord;

            vec4 col = texture2D(u_image, uv);
            col.rgb *= 1.3;

            float b = brightness(col.rgb);

            vec2 distortion = vec2(
                (random(uv + vec2(sin(u_time * 0.1), cos(u_time * 0.1))) - 0.5) * 0.1,
                (random(uv + vec2(cos(u_time * 0.1), sin(u_time * 0.1))) - 0.5) * 0.1
            );
            
            vec4 distorted = texture2D(u_image, uv + distortion);

            if (b > 0.5) {
                vec3 glowColor = vec3(
                    0.5 + 0.3 * sin(u_time * 2.0),
                    0.2 + 0.5 * cos(u_time * 1.5),
                    0.6 + 0.4 * sin(u_time * 3.0)
                );

                finalColor = mix(distorted.rgb, glowColor, 0.5);
            } else {
                finalColor = distorted.rgb;
            }
            alpha = col.a;
        } else if (u_filterType == FILTER_AUDIO_COLOR_SHIFT) { 
            finalColor = mod(color.rgb + u_colorShift, 1.0);
        } else if (u_filterType == FILTER_MODULAR_COLOR_SHIFT) {
            const vec3 palette0 = vec3(80.0/255.0, 120.0/255.0, 180.0/255.0);
            const vec3 palette1 = vec3(100.0/255.0, 180.0/255.0, 200.0/255.0);
            const vec3 palette2 = vec3(120.0/255.0, 150.0/255.0, 255.0/255.0);

            float brightness_val = (color.r + color.g + color.b) / 3.0;

            if (brightness_val > (170.0/255.0)) {
                finalColor.rgb = mix(color.rgb, palette2, u_highAmp);
            } else if (brightness_val > (100.0/255.0)) {
                finalColor.rgb = mix(color.rgb, palette1, u_midAmp);
            } else {
                finalColor.rgb = mix(color.rgb, palette0, u_bassAmp);
            }
            finalColor.rgb = clamp(finalColor.rgb, 0.0, 1.0);
        } else if (u_filterType == FILTER_KALEIDOSCOPE) {
            // Caleidoscopio de 6 sectores
            vec2 c = texCoord - 0.5;
            float r = length(c);
            float angle = atan(c.y, c.x);
            angle = mod(angle, radians(60.0)); // 360 / 6 = 60 grados por sector
            angle = abs(angle - radians(30.0)); // Reflejar en el centro del sector
            
            vec2 newTexCoord = vec2(r * cos(angle), r * sin(angle)) + 0.5;
            finalColor = texture2D(u_image, newTexCoord).rgb;
        } else if (u_filterType == FILTER_MIRROR) {
            vec2 uv = texCoord;
            if (uv.x > 0.5) {
                uv.x = 1.0 - uv.x;
            }
            finalColor = texture2D(u_image, uv).rgb;
        } else if (u_filterType == FILTER_FISHEYE) {
            finalColor = texture2D(u_image, fisheye(texCoord, 0.8)).rgb; // 0.8 para un efecto convexo fuerte
        } else if (u_filterType == FILTER_RECUERDO) {
            vec2 uv = texCoord;
            
            // 1. Distorsión amorfa/Oscilación MÁS SUAVE (wobble=0.01)
            float wobble = sin(u_time * 0.8) * 0.01; // Menor deformación
            uv.x += sin(uv.y * 20.0 + u_time * 3.0) * wobble; 
            uv.y += cos(uv.x * 22.0 + u_time * 2.5) * wobble; 

            // 2. Ruido visual (Grano)
            float grain = random(uv * u_time) * 0.1; 
            
            // 3. Desenfoque Radial y Punto de enfoque MÁS COMPRIMIDO
            vec2 center = vec2(0.5, 0.5);
            vec2 direction = uv - center;
            float dist = length(direction);
            
            // Enfoque/Máscara de la zona nítida
            float focus = smoothstep(0.4, 0.2, dist * 2.5); 

            // Implementación de Desenfoque Radial
            vec4 blurredColor = vec4(0.0);
            
            const float blurSamples = 8.0;
            float blurStrength = (1.0 - focus) * 0.015; // La fuerza del blur es inversamente proporcional al foco
            
            // Muestreo radial a lo largo de la dirección
            for (float i = 0.0; i < blurSamples; i++) {
                // Offset a lo largo del vector "direction"
                vec2 offset = direction * blurStrength * (i / blurSamples); 
                blurredColor += texture2D(u_image, uv + offset);
            }
            blurredColor /= blurSamples;

            // Mezclar entre el color borroso y el original (nítido) usando 'focus'
            vec4 originalColor = texture2D(u_image, uv);
            vec4 finalColorMixed = mix(blurredColor, originalColor, focus);

            // 4. Aplicar Color (Tonos lavados / Sobreexposición leve)
            finalColor = finalColorMixed.rgb * vec3(1.1, 1.05, 0.9); 

            // 5. Aplicar Sepia (tonos lavados)
            float r = finalColor.r;
            float g = finalColor.g;
            float b = finalColor.b;
            
            float sepiaR = (r * 0.393) + (g * 0.769) + (b * 0.189);
            float sepiaG = (r * 0.349) + (g * 0.686) + (b * 0.168);
            float sepiaB = (r * 0.272) + (g * 0.534) + (b * 0.131);
            
            vec3 sepiaColor = clamp(vec3(sepiaR, sepiaG, sepiaB), 0.0, 1.0);
            
            finalColor = mix(finalColor, sepiaColor, 0.5); 
            
            // 6. Aplicar Ruido
            finalColor += grain;
            
            finalColor = clamp(finalColor, 0.0, 1.0); 
        } else if (u_filterType == FILTER_GLITCH2) {
            vec2 uv = texCoord;

            // Líneas de glitch horizontales: bloques de filas desplazados
            float blockY = floor(uv.y * 40.0) / 40.0;
            float glitchSeed = random(vec2(blockY, floor(u_time * 12.0)));
            float glitchSeed2 = random(vec2(blockY + 0.3, floor(u_time * 8.0)));

            // Solo algunos bloques se distorsionan intensamente
            float shift = 0.0;
            if (glitchSeed > 0.65) {
                shift = (glitchSeed - 0.65) * 1.2; // desplazamiento horizontal fuerte
            }
            uv.x = fract(uv.x + shift);

            // Separación cromática intensa (aberración)
            float aberration = 0.018 + glitchSeed2 * 0.025;
            float r = texture2D(u_image, vec2(uv.x + aberration, uv.y)).r;
            float g = texture2D(u_image, uv).g;
            float b = texture2D(u_image, vec2(uv.x - aberration, uv.y)).b;
            vec3 aberrated = vec3(r, g, b);

            // Paleta: empujar hacia rosas y verdes vivos
            // Rosa: boost R, boost B, bajar G. Verde: boost G, bajar R y B
            float lum = (aberrated.r * 0.299 + aberrated.g * 0.587 + aberrated.b * 0.114);
            vec3 pink  = vec3(1.0, 0.05, 0.75);
            vec3 green = vec3(0.0, 1.0, 0.2);
            // Alterna entre rosa y verde según posición y tiempo
            float wave = sin(uv.y * 18.0 + u_time * 5.0) * 0.5 + 0.5;
            vec3 palette = mix(pink, green, wave);
            // Mezclar: zonas brillantes toman el color de paleta, zonas oscuras se quedan oscuras
            vec3 colorized = mix(aberrated * 0.3, palette, smoothstep(0.2, 0.8, lum));

            // Ruido digital en bloques (scanlines finas)
            float scanline = mod(floor(uv.y * u_resolution.y), 3.0);
            float noise = random(vec2(uv.x * 100.0, floor(u_time * 20.0) + uv.y * 50.0));
            if (scanline == 0.0 && noise > 0.6) {
                colorized = mix(colorized, palette.brg, 0.8); // píxeles corruptos
            }

            finalColor = clamp(colorized, 0.0, 1.0);
        }

        gl_FragColor = vec4(finalColor, alpha);
    }
`;

// Helper para compilar un shader
function compileShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Error al compilar el shader:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

// Helper para enlazar shaders a un programa
function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Error al enlazar el programa:', gl.getProgramInfoLog(program));
        gl.deleteProgram(program);
        return null;
    }
    return program;
}

// CORRECCIÓN DE CÁMARA INVERTIDA: Invertir el orden de los vértices X para corregir el efecto espejo
function setupQuadBuffers(gl) {
    const positions = new Float32Array([
        1, -1, 
        -1, -1,
        1,  1,
        1,  1,
        -1, -1,
        -1,  1, 
    ]);
    positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    // Las coordenadas de textura deben seguir el nuevo orden de los vértices
    const texCoords = new Float32Array([
        1, 1, 
        0, 1, 
        1, 0, 
        1, 0,
        0, 1,
        0, 0, 
    ]);
    texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
}

// Inicializa las texturas
function setupTextures(gl) {
    // Textura para el video original (para filtros WebGL)
    videoTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Textura para el output de MediaPipe (dibujado en el canvas 2D auxiliar)
    mpOutputTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, mpOutputTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
}

// --- FUNCIÓN DE INICIALIZACIÓN WEBG L ---
function initWebGL() {
    gl = glcanvas.getContext('webgl', { preserveDrawingBuffer: true }); 
    if (!gl) {
        alert('Tu navegador no soporta WebGL. No se podrán aplicar filtros avanzados.');
        console.error('WebGL no soportado.');
        return;
    }
    console.log('Contexto WebGL obtenido.');

    const vertexShader = compileShader(gl, vsSource, gl.VERTEX_SHADER);
    const fragmentShader = compileShader(gl, fsSource, gl.FRAGMENT_SHADER);
    program = createProgram(gl, vertexShader, fragmentShader);
    gl.useProgram(program);

    program.positionLocation = gl.getAttribLocation(program, 'a_position');
    program.texCoordLocation = gl.getAttribLocation(program, 'a_texCoord');
    program.imageLocation = gl.getUniformLocation(program, 'u_image');
    filterTypeLocation = gl.getUniformLocation(program, 'u_filterType');
    program.resolutionLocation = gl.getUniformLocation(program, 'u_resolution');
    timeLocation = gl.getUniformLocation(program, 'u_time');
    colorShiftUniformLocation = gl.getUniformLocation(program, 'u_colorShift');
    program.aspectRatioLocation = gl.getUniformLocation(program, 'u_aspectRatio');

    bassAmpUniformLocation = gl.getUniformLocation(program, 'u_bassAmp');
    midAmpUniformLocation = gl.getUniformLocation(program, 'u_midAmp');
    highAmpUniformLocation = gl.getUniformLocation(program, 'u_highAmp');

    gl.enableVertexAttribArray(program.positionLocation);
    gl.enableVertexAttribArray(program.texCoordLocation);

    setupQuadBuffers(gl); // Llama a la función con la corrección de espejo
    setupTextures(gl); 

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.vertexAttribPointer(program.positionLocation, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.vertexAttribPointer(program.texCoordLocation, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(program.imageLocation, 0); 

    gl.uniform1i(filterTypeLocation, 0);
    console.log('WebGL inicialización completa.');
}

// --- FUNCIÓN DE INICIALIZACIÓN MEDIAPIPE ---
function initMediaPipe() {
    selfieSegmentation = new SelfieSegmentation({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
    });

    // 0 para modelos generales (rápido), 1 para modelos más grandes (más precisos para segmentación humana)
    selfieSegmentation.setOptions({
        modelSelection: 1, 
    });

    selfieSegmentation.onResults(onMediaPipeResults);
    console.log('MediaPipe SelfieSegmentation inicializado.');
}

// Callback de MediaPipe: se llama cuando MediaPipe ha procesado un fotograma
function onMediaPipeResults(results) {
    mpResults = results; // Almacenar los resultados
    mpProcessing = false; // Permitir el siguiente envío de fotogramas
}

// --- LÓGICA DE CÁMARA Y STREAMING ---
let availableCameraDevices = [];

async function listCameras() {
  try {
    await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(stream => {
            stream.getTracks().forEach(track => track.stop());
        })
        .catch(err => {
            console.warn('listCameras: Error al obtener permisos de cámara (puede ser ignorado si el usuario deniega):', err);
        });

    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');

    availableCameraDevices = videoDevices;
    
    if (availableCameraDevices.length > 0) {
      if (!currentCameraDeviceId || !availableCameraDevices.some(d => d.deviceId === currentCameraDeviceId)) {
        currentCameraDeviceId = availableCameraDevices[0].deviceId;
      }
      startCamera(currentCameraDeviceId);
    } else {
      alert('No se encontraron dispositivos de cámara.');
    }
  } catch (err) {
    console.error('listCameras: Error al listar dispositivos de cámara:', err);
    alert('Error al listar dispositivos de cámara. Revisa los permisos.');
  }
}

async function startCamera(deviceId) {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }

  const constraints = {
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width: { ideal: 1280 },
      height: { ideal: 720 }
    },
    audio: true
  };

  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = currentStream;
    
    const videoTrack = currentStream.getVideoTracks()[0];
    const settings = videoTrack.getSettings();
    currentCameraDeviceId = settings.deviceId;
    currentFacingMode = settings.facingMode || 'unknown';

    // --- Web Audio API setup ---
    if (currentStream.getAudioTracks().length > 0) {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.7;
            dataArray = new Uint8Array(analyser.frequencyBinCount);
        }

        if (microphone) {
            microphone.disconnect();
        }

        const audioSource = audioContext.createMediaStreamSource(currentStream);
        audioSource.connect(analyser);
        microphone = audioSource;
    } else {
        if (microphone) {
            microphone.disconnect();
            microphone = null;
        }
        if (analyser) {
            analyser.disconnect();
            analyser = null;
        }
    }
    // --- Fin Web Audio API setup ---

    video.onloadedmetadata = () => {
      video.play();
      if (glcanvas.width !== video.videoWidth || glcanvas.height !== video.videoHeight) {
        glcanvas.width = video.videoWidth;
        glcanvas.height = video.videoHeight;
        canvas.width = video.videoWidth; // Ajustar también el canvas 2D
        canvas.height = video.videoHeight; // Ajustar también el canvas 2D
        if (gl) {
          gl.viewport(0, 0, glcanvas.width, glcanvas.height);
          gl.uniform1f(program.aspectRatioLocation, glcanvas.width / glcanvas.height);
        }
      }
      if (!gl) {
        initWebGL();
      }
      // Inicializar MediaPipe y Camera si no están ya inicializados
      if (!selfieSegmentation) {
        initMediaPipe();
      }
      if (!mpCamera) {
        mpCamera = new Camera(video, { // Pasar el elemento de video directamente
          onFrame: async () => {
            if (!mpProcessing) { // Solo enviar un nuevo frame si el anterior ha sido procesado
                mpProcessing = true;
                await selfieSegmentation.send({ image: video });
            }
          },
          width: video.videoWidth,
          height: video.videoHeight,
        });
        mpCamera.start();
        console.log('MediaPipe Camera inicializada y en marcha.');
      }
      drawVideoFrame(); // Iniciar el bucle de renderizado
    };
  } catch (err) {
    console.error('startCamera: No se pudo acceder a la cámara/micrófono:', err);
    alert('No se pudo acceder a la cámara/micrófono. Revisa los permisos. Error: ' + err.name);
  }
}

// --- BUCLE PRINCIPAL DE RENDERIZADO WEBG L ---
function drawVideoFrame() {
    if (!gl || !program || !video.srcObject || video.readyState !== video.HAVE_ENOUGH_DATA) {
        requestAnimationFrame(drawVideoFrame);
        return;
    }

    // --- Detección de nivel de audio (solo si el filtro de audio está seleccionado) ---
    if (analyser && dataArray && selectedFilter === 'audio-color-shift') {
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
        }
        let average = sum / dataArray.length;
        let normalizedLevel = average / 255.0;

        if (normalizedLevel > AUDIO_THRESHOLD) {
            changePaletteIndex();
        }
    }
    // --- Fin detección de nivel de nivel de audio ---

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);

    gl.uniform2f(program.resolutionLocation, glcanvas.width, glcanvas.height);
    gl.uniform1f(program.aspectRatioLocation, glcanvas.width / glcanvas.height); // Pasar el aspect ratio
    const currentTime = performance.now() / 1000.0;
    gl.uniform1f(timeLocation, currentTime);

    // Los filtros eliminados se quitan de la lista de MediaPipe
    const isMediaPipeFilter = ['whiteGlow', 'blackBg', 'whiteBg', 'invisible', 'static-silhouette', 'echo-visual'].includes(selectedFilter);

    if (isMediaPipeFilter && mpResults && mpResults.segmentationMask && mpResults.image) {
        // Renderizar con MediaPipe en el canvas 2D auxiliar
        
        let postProcessFilterIndex = 0; // Por defecto: FILTER_NONE

        switch (selectedFilter) {
            case "whiteGlow": { // Silueta Roja: fondo real + silueta rellena de rojo
                mpCanvasCtx.clearRect(0, 0, canvas.width, canvas.height);

                // 1. Dibujar el fondo del video (imagen completa)
                mpCanvasCtx.drawImage(mpResults.image, 0, 0, canvas.width, canvas.height);

                // 2. Pintar la silueta de la persona en rojo sólido usando la máscara como clip
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = canvas.width;
                tempCanvas.height = canvas.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.fillStyle = "red";
                tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
                tempCtx.globalCompositeOperation = "destination-in";
                tempCtx.drawImage(mpResults.segmentationMask, 0, 0, tempCanvas.width, tempCanvas.height);
                mpCanvasCtx.drawImage(tempCanvas, 0, 0);
                break;
            }

            case "blackBg": // Silueta Negra: fondo real + silueta rellena de negro
            case "whiteBg": { // Silueta Blanca: fondo real + silueta rellena de blanco
                mpCanvasCtx.clearRect(0, 0, canvas.width, canvas.height);

                // 1. Dibujar el fondo del video (imagen completa)
                mpCanvasCtx.drawImage(mpResults.image, 0, 0, canvas.width, canvas.height);

                // 2. Pintar la silueta con el color sólido usando la máscara como clip
                const tempCanvasBW = document.createElement('canvas');
                tempCanvasBW.width = canvas.width;
                tempCanvasBW.height = canvas.height;
                const tempCtxBW = tempCanvasBW.getContext('2d');
                tempCtxBW.fillStyle = selectedFilter === "blackBg" ? "black" : "white";
                tempCtxBW.fillRect(0, 0, tempCanvasBW.width, tempCanvasBW.height);
                tempCtxBW.globalCompositeOperation = "destination-in";
                tempCtxBW.drawImage(mpResults.segmentationMask, 0, 0, tempCanvasBW.width, tempCanvasBW.height);
                mpCanvasCtx.drawImage(tempCanvasBW, 0, 0);
                break;
            }

            case "invisible": { // Invisible: camufla la silueta con el entorno difuminado
                mpCanvasCtx.clearRect(0, 0, canvas.width, canvas.height);

                // 1. Dibujar la imagen completa del video
                mpCanvasCtx.drawImage(mpResults.image, 0, 0, canvas.width, canvas.height);

                // 2. Crear una versión difuminada del video para usarla como "fondo reconstruido"
                const bgCanvas = document.createElement('canvas');
                bgCanvas.width = canvas.width;
                bgCanvas.height = canvas.height;
                const bgCtx = bgCanvas.getContext('2d');
                bgCtx.filter = "blur(18px)";
                bgCtx.drawImage(mpResults.image, -30, -30, canvas.width + 60, canvas.height + 60);
                bgCtx.filter = "none";

                // 3. Recortar ese fondo difuminado a la silueta de la persona
                bgCtx.globalCompositeOperation = "destination-in";
                bgCtx.drawImage(mpResults.segmentationMask, 0, 0, bgCanvas.width, bgCanvas.height);

                // 4. Superponer el fondo difuminado recortado sobre la imagen (camufla a la persona)
                mpCanvasCtx.drawImage(bgCanvas, 0, 0);
                break;
            }

            case "static-silhouette": { // Silueta estática: fondo real + silueta rellena de ruido TV
                mpCanvasCtx.clearRect(0, 0, canvas.width, canvas.height);

                // 1. Dibujar el fondo del video
                mpCanvasCtx.drawImage(mpResults.image, 0, 0, canvas.width, canvas.height);

                // 2. Generar ruido tipo televisor en un canvas temporal
                const noiseCanvas = document.createElement('canvas');
                noiseCanvas.width = canvas.width;
                noiseCanvas.height = canvas.height;
                const noiseCtx = noiseCanvas.getContext('2d');
                const imageData = noiseCtx.createImageData(noiseCanvas.width, noiseCanvas.height);
                const data = imageData.data;
                for (let i = 0; i < data.length; i += 4) {
                    const v = Math.random() * 255;
                    data[i]     = v;
                    data[i + 1] = v;
                    data[i + 2] = v;
                    data[i + 3] = 255;
                }
                noiseCtx.putImageData(imageData, 0, 0);

                // 3. Recortar el ruido a la silueta de la persona
                noiseCtx.globalCompositeOperation = "destination-in";
                noiseCtx.drawImage(mpResults.segmentationMask, 0, 0, noiseCanvas.width, noiseCanvas.height);

                // 4. Dibujar el ruido recortado sobre el fondo
                mpCanvasCtx.drawImage(noiseCanvas, 0, 0);
                break;
            }

            case "echo-visual": { // Eco visual: siluetas repetidas desplazadas detrás de la persona
                mpCanvasCtx.clearRect(0, 0, canvas.width, canvas.height);

                // 1. Dibujar el fondo del video
                mpCanvasCtx.drawImage(mpResults.image, 0, 0, canvas.width, canvas.height);

                // 2. Crear silueta recortada de la persona (en blanco/negro) en canvas temporal
                const ecoBase = document.createElement('canvas');
                ecoBase.width = canvas.width;
                ecoBase.height = canvas.height;
                const ecoCtx = ecoBase.getContext('2d');

                // Silueta oscura semitransparente
                ecoCtx.fillStyle = "rgba(0, 0, 0, 1)";
                ecoCtx.fillRect(0, 0, ecoBase.width, ecoBase.height);
                ecoCtx.globalCompositeOperation = "destination-in";
                ecoCtx.drawImage(mpResults.segmentationMask, 0, 0, ecoBase.width, ecoBase.height);

                // 3. Dibujar 4 ecos desplazados hacia atrás (con opacidad decreciente)
                const echoCount = 4;
                const stepX = 18; // desplazamiento horizontal por eco
                const stepY = 8;  // leve desplazamiento vertical
                for (let i = echoCount; i >= 1; i--) {
                    mpCanvasCtx.globalAlpha = 0.18 * (echoCount - i + 1) / echoCount;
                    mpCanvasCtx.drawImage(ecoBase, stepX * i, stepY * i);
                }

                // 4. Dibujar la silueta original del video (recortada) encima, nítida
                mpCanvasCtx.globalAlpha = 1.0;
                const personCanvas = document.createElement('canvas');
                personCanvas.width = canvas.width;
                personCanvas.height = canvas.height;
                const personCtx = personCanvas.getContext('2d');
                personCtx.drawImage(mpResults.image, 0, 0, personCanvas.width, personCanvas.height);
                personCtx.globalCompositeOperation = "destination-in";
                personCtx.drawImage(mpResults.segmentationMask, 0, 0, personCanvas.width, personCanvas.height);
                mpCanvasCtx.drawImage(personCanvas, 0, 0);
                break;
            }
        }
        mpCanvasCtx.globalCompositeOperation = "source-over"; // Resetear para futuros dibujos

        // Actualizar la textura WebGL con el contenido del canvas 2D de MediaPipe
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, mpOutputTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas); // Usa el canvas 2D como fuente
        gl.uniform1i(filterTypeLocation, postProcessFilterIndex); // Usa el filtro de post-proceso
    } else {
        // Renderizar con WebGL directamente usando la textura del video
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, videoTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video); // Usa el video como fuente

        if (selectedFilter === 'audio-color-shift') {
            const currentColor = palettes[paletteIndex];
            gl.uniform3fv(colorShiftUniformLocation, new Float32Array(currentColor));
        }

        if (selectedFilter === 'modular-color-shift') {
            const bassAmp = mapValue(Math.sin(currentTime * 0.8 + 0), -1, 1, 0.0, 2.0);
            const midAmp = mapValue(Math.sin(currentTime * 1.2 + Math.PI / 3), -1, 1, 0.0, 1.5);
            const highAmp = mapValue(Math.sin(currentTime * 1.5 + Math.PI * 2 / 3), -1, 1, 0.0, 2.5);

            gl.uniform1f(bassAmpUniformLocation, bassAmp);
            gl.uniform1f(midAmpUniformLocation, midAmp);
            gl.uniform1f(highAmpUniformLocation, highAmp);
        }

        let filterIndex = 0;
        switch (selectedFilter) {
            case 'grayscale': filterIndex = 1; break;
            case 'invert': filterIndex = 2; break;
            case 'sepia': filterIndex = 3; break;
            case 'eco-pink': filterIndex = 4; break;
            case 'weird': filterIndex = 5; break;
            case 'glow-outline': filterIndex = 6; break;
            case 'angelical-glitch': filterIndex = 7; break;
            case 'audio-color-shift': filterIndex = 8; break;
            case 'modular-color-shift': filterIndex = 9; break;
            case 'kaleidoscope': filterIndex = 10; break;
            case 'mirror': filterIndex = 11; break;
            case 'fisheye': filterIndex = 12; break;
            case 'recuerdo': filterIndex = 13; break;
            case 'glitch2': filterIndex = 14; break;
            default: filterIndex = 0; break;
        }
        gl.uniform1i(filterTypeLocation, filterIndex);
    }

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(drawVideoFrame);
}

function mapValue(value, inMin, inMax, outMin, outMax) {
    return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}


// --- MANEJADORES DE EVENTOS ---
captureBtn.addEventListener('click', () => {
    if (!gl || !glcanvas.width || !glcanvas.height) {
        console.error('WebGL no está inicializado o el canvas no tiene dimensiones para la captura.');
        return;
    }

    let img = new Image();
    img.src = glcanvas.toDataURL('image/png'); 
    
    img.onload = () => {
        addToGallery(img, 'img');
    };
    img.onerror = (e) => {
        console.error('Error al cargar la imagen para la galería:', e);
    };
});


recordBtn.addEventListener('click', () => {
  if (!isRecording) {
    chunks = [];
    console.log('Iniciando grabación desde glcanvas.captureStream().');
    let streamToRecord = glcanvas.captureStream(); // Capturar el stream del canvas con los filtros
    
    // Si hay audio en el currentStream de la cámara, añadirlo a la grabación
    const audioTracks = currentStream.getAudioTracks();
    if (audioTracks.length > 0) {
        let audioStream = new MediaStream();
        audioStream.addTrack(audioTracks[0]);
        streamToRecord.addTrack(audioTracks[0]); // Combina el video del canvas con el audio de la cámara
    }

    mediaRecorder = new MediaRecorder(streamToRecord, { mimeType: 'video/webm; codecs=vp8' });

    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) chunks.push(e.data);
      console.log('Datos de video disponibles, tamaño:', e.data.size);
    };
    mediaRecorder.onstop = () => {
      console.log('Grabación detenida. Chunks capturados:', chunks.length);
      const blob = new Blob(chunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      let vid = document.createElement('video');
      vid.src = url;
      vid.controls = true;
      addToGallery(vid, 'video');
    };
    mediaRecorder.start();
    isRecording = true;
    controls.style.display = 'none';
    recordingControls.style.display = 'flex';
    console.log('Grabación iniciada.');
  }
});

pauseBtn.addEventListener('click', () => {
  if (isPaused) {
    mediaRecorder.resume();
    pauseBtn.textContent = 'Pausa'; // Texto de pausa
    console.log('Grabación reanudada.');
  } else {
    mediaRecorder.pause();
    pauseBtn.textContent = 'Reanudar'; // Texto de reanudar
    console.log('Grabación pausada.');
  }
  isPaused = !isPaused;
});

stopBtn.addEventListener('click', () => {
  mediaRecorder.stop();
  isRecording = false;
  controls.style.display = 'flex';
  recordingControls.style.display = 'none';
  console.log('Grabación finalizada.');
});

filterBtn.addEventListener('click', () => {
  filtersDropdown.style.display = (filtersDropdown.style.display === 'block') ? 'none' : 'block';
  console.log('Toggle de dropdown de filtros.');
});

filterSelect.addEventListener('change', () => {
  selectedFilter = filterSelect.value;
  filtersDropdown.style.display = 'none';
  console.log('Filtro seleccionado manualmente:', selectedFilter);
});

fullscreenBtn.addEventListener('click', () => {
  if (!document.fullscreenElement) {
    cameraContainer.requestFullscreen();
    console.log('Solicitando fullscreen.');
  } else {
    document.exitFullscreen();
    console.log('Saliendo de fullscreen.');
  }
});

function addToGallery(element, type) {
    // 1. Incrementar contador y crear título enumerado
    mediaCounter++;
    const itemType = type === 'img' ? 'Foto' : 'Video';
    const itemTitle = `${itemType} ${mediaCounter}`;

    let container = document.createElement('div');
    container.className = 'gallery-item';
    container.dataset.title = itemTitle; // Añadir el título como data-attribute

    // 2. Añadir la etiqueta visual
    const label = document.createElement('div');
    label.classList.add('gallery-label');
    label.textContent = itemTitle;
    container.appendChild(label); // Añadir la etiqueta al contenedor

    container.appendChild(element);

    // Event listener para abrir la ventana de previsualización al hacer clic
    element.addEventListener('click', () => {
        console.log('Creando ventana de previsualización de', type);
        
        const previewWindow = document.createElement('div');
        previewWindow.className = 'preview-window';
        document.body.appendChild(previewWindow);

        const closeButton = document.createElement('span');
        closeButton.textContent = 'X'; 
        closeButton.className = 'close-preview-window-button';
        previewWindow.appendChild(closeButton);

        const clonedElement = element.cloneNode(true);
        if (type === 'video') {
            clonedElement.controls = true; // Mostrar controles para videos
            clonedElement.play(); // Reproducir al abrir
        }
        previewWindow.appendChild(clonedElement);

        // --- Botones de acción en la previsualización ---
        let previewActions = document.createElement('div');
        previewActions.className = 'preview-actions';

        let downloadBtn = document.createElement('button');
        downloadBtn.textContent = 'Descargar'; 
        downloadBtn.onclick = () => {
            const a = document.createElement('a');
            a.href = clonedElement.src;
            // 3. Usar el título enumerado para el nombre del archivo
            const extension = type === 'img' ? '.png' : '.webm';
            a.download = itemTitle.replace(/\s/g, '_') + extension; 
            a.click();
            console.log('Descargando desde previsualización', type);
        };

        let shareBtn = document.createElement('button');
        shareBtn.textContent = 'Compartir'; 
        shareBtn.onclick = async () => {
            if (navigator.share) {
                try {
                    const file = await fetch(clonedElement.src).then(res => res.blob());
                    const fileName = itemTitle.replace(/\s/g, '_') + (type === 'img' ? '.png' : '.webm');
                    const fileType = type === 'img' ? 'image/png' : 'video/webm';
                    const shareData = {
                        files: [new File([file], fileName, { type: fileType })],
                        title: 'Mi creación desde Experimental Camera',
                        text: '¡Echa un vistazo a lo que hice con Experimental Camera!'
                    };
                    await navigator.share(shareData);
                    console.log('Contenido compartido exitosamente desde previsualización');
                } catch (error) {
                    console.error('Error al compartir desde previsualización:', error);
                }
            } else {
                alert('La API Web Share no es compatible con este navegador.');
                console.warn('La API Web Share no es compatible.');
            }
        };

        let deleteBtn = document.createElement('button');
        deleteBtn.textContent = 'Eliminar'; 
        deleteBtn.onclick = () => {
            if (type === 'video' && clonedElement.src.startsWith('blob:')) {
                URL.revokeObjectURL(clonedElement.src); // Libera la URL del blob para videos
            }
            previewWindow.remove(); // Cierra la ventana de previsualización
            container.remove(); // Elimina el elemento de la galería
            console.log('Elemento de galería y previsualización eliminados.');
        };
        
        previewActions.appendChild(downloadBtn);
        if (navigator.share) {
            previewActions.appendChild(shareBtn);
        }
        previewActions.appendChild(deleteBtn);
        previewWindow.appendChild(previewActions);
        // --- Fin Botones de acción en la previsualización ---

        // Event listener para cerrar la ventana
        closeButton.addEventListener('click', () => {
            if (type === 'video' && clonedElement) {
                clonedElement.pause();
            }
            previewWindow.remove();
            console.log('Ventana de previsualización cerrada.');
        });

        // Hacer la ventana arrastrable
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        previewWindow.onmousedown = dragMouseDown;

        function dragMouseDown(e) {
            e = e || window.event;
            e.preventDefault();
            // obtener la posición del cursor en el inicio:
            pos3 = e.clientX;
            pos4 = e.clientY;
            document.onmouseup = closeDragElement;
            // llamar a una función cada vez que el cursor se mueve:
            document.onmousemove = elementDrag;
        }

        function elementDrag(e) {
            e = e || window.event;
            e.preventDefault();
            // calcular la nueva posición del cursor:
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;
            // establecer la nueva posición del elemento:
            previewWindow.style.top = (previewWindow.offsetTop - pos2) + "px";
            previewWindow.style.left = (previewWindow.offsetLeft - pos1) + "px";
        }

        function closeDragElement() {
            /* dejar de moverse cuando se suelta el botón del ratón: */
            document.onmouseup = null;
            document.onmousemove = null;
        }
    });

    // 4. Actualizar botones de acción de la galería
    let actions = document.createElement('div');
    actions.className = 'gallery-actions';

    let downloadBtn = document.createElement('button');
    downloadBtn.textContent = 'Descargar';
    downloadBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = element.src;
        // Usar el título enumerado para el nombre del archivo
        const extension = type === 'img' ? '.png' : '.webm';
        a.download = itemTitle.replace(/\s/g, '_') + extension;
        a.click();
        console.log('Descargando', type);
    };

    let shareBtn = document.createElement('button');
    shareBtn.textContent = 'Compartir';
    shareBtn.onclick = async () => {
        if (navigator.share) {
        try {
            const file = await fetch(element.src).then(res => res.blob());
            const fileName = itemTitle.replace(/\s/g, '_') + (type === 'img' ? '.png' : '.webm');
            const fileType = type === 'img' ? 'image/png' : 'video/webm';
            const shareData = {
            files: [new File([file], fileName, { type: fileType })],
            title: 'Mi creación desde Experimental Camera',
            text: '¡Echa un vistazo a lo que hice con Experimental Camera!'
            };
            await navigator.share(shareData);
            console.log('Contenido compartido exitosamente');
        } catch (error) {
            console.error('Error al compartir:', error);
        }
        } else {
        alert('La API Web Share no es compatible con este navegador.');
        console.warn('La API Web Share no es compatible.');
        }
    };

    let deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Eliminar';
    deleteBtn.onclick = () => {
        if (type === 'video' && element.src.startsWith('blob:')) {
        URL.revokeObjectURL(element.src);
        }
        container.remove();
        console.log('Elemento de galería eliminado.');
    };

    actions.appendChild(downloadBtn);
    if (navigator.share) {
        actions.appendChild(shareBtn);
    }
    actions.appendChild(deleteBtn);
    container.appendChild(actions);

    gallery.prepend(container);
}


// --- LÓGICA DE DOBLE TAP/CLICK PARA CAMBIAR DE CÁMARA ---
let lastTap = 0;
const DBL_TAP_THRESHOLD = 300;

glcanvas.addEventListener('touchend', (event) => {
    const currentTime = new Date().getTime();
    const tapLength = currentTime - lastTap;

    if (tapLength < DBL_TAP_THRESHOLD && tapLength > 0) {
        event.preventDefault();
        toggleCamera();
    }
    lastTap = currentTime;
}, { passive: false });

glcanvas.addEventListener('dblclick', () => {
    toggleCamera();
});

function toggleCamera() {
    if (availableCameraDevices.length > 1) {
        const currentIdx = availableCameraDevices.findIndex(
            device => device.deviceId === currentCameraDeviceId
        );
        const nextIdx = (currentIdx + 1) % availableCameraDevices.length;
        const nextDeviceId = availableCameraDevices[nextIdx].deviceId;
        startCamera(nextDeviceId);
    } else {
        alert("Solo hay una cámara disponible.");
    }
}

function changePaletteIndex() {
    paletteIndex = (paletteIndex + 1) % palettes.length;
}

// --- LÓGICA DE SWIPE PARA CAMBIAR FILTROS ---
let touchStartX = 0;
let touchEndX = 0;
const SWIPE_THRESHOLD = 50; // Pixeles mínimos para considerar un swipe

glcanvas.addEventListener('touchstart', (e) => {
    touchStartX = e.touches[0].clientX;
});

glcanvas.addEventListener('touchmove', (e) => {
    touchEndX = e.touches[0].clientX;
});

glcanvas.addEventListener('touchend', () => {
    const diffX = touchEndX - touchStartX;

    if (Math.abs(diffX) > SWIPE_THRESHOLD) {
        // Obtener todas las opciones, incluyendo las de los optgroups
        const options = Array.from(filterSelect.querySelectorAll('option')).map(option => option.value);
        let currentIndex = options.indexOf(selectedFilter);

        if (diffX > 0) { // Swipe a la derecha (filtro anterior)
            currentIndex = (currentIndex > 0) ? currentIndex - 1 : options.length - 1;
        } else { // Swipe a la izquierda (filtro siguiente)
            currentIndex = (currentIndex < options.length - 1) ? currentIndex + 1 : 0;
        }
        
        selectedFilter = options[currentIndex];
        filterSelect.value = selectedFilter; // Sincroniza el select
    }
    // Reiniciar valores de touch
    touchStartX = 0;
    touchEndX = 0;
});

// --- CORRECCIÓN DE GESTO: EVITAR CAMBIO DE FILTRO POR TAP/CLICK ---
glcanvas.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
});

listCameras();

// --- TUTORIAL DE INICIACIÓN ---
(function initTutorial() {
    const overlay = document.getElementById('tutorial-overlay');
    const skipBtn = document.getElementById('tutorial-skip');
    if (!overlay || !skipBtn) return;

    const TUTORIAL_KEY = 'exp_cam_tutorial_seen';
    if (localStorage.getItem(TUTORIAL_KEY)) {
        overlay.style.display = 'none';
        return;
    }

    skipBtn.addEventListener('click', () => {
        overlay.style.display = 'none';
        try { localStorage.setItem(TUTORIAL_KEY, '1'); } catch(e) {}
    });
})();
