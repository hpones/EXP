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
    const int FILTER_KALEIDOSCOPE = 10; // Changed from FRACTAL
    const int FILTER_MIRROR = 11;
    const int FILTER_FISHEYE = 12;
    // Los filtros de silueta serán manejados por MediaPipe en el canvas 2D

    // Función para generar ruido básico
    float random(vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
    }
    
    // Función de brillo para detectar luz
    float brightness(vec3 color) {
        return dot(color, vec3(0.299, 0.587, 0.114));
    }

    // Kaleidoscope effect (replaces Fractal)
    vec4 kaleidoscope(sampler2D image, vec2 uv, float angle, float segments) {
        const float PI = 3.14159265359;
        vec2 center = vec2(0.5, 0.5);
        vec2 p = uv - center;
        float r = length(p);
        float a = atan(p.y, p.x);
        a = mod(a - angle, 2.0 * PI / segments);
        if (mod(floor(a * segments / PI), 2.0) == 0.0) {
            a = 2.0 * PI / segments - a;
        }
        p = vec2(r * cos(a), r * sin(a));
        return texture2D(image, p + center);
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
        } else if (u_filterType == FILTER_KALEIDOSCOPE) { // Kaleidoscope filter
            finalColor = kaleidoscope(u_image, texCoord, u_time * 0.2, 6.0).rgb; // 6 segments, rotates slowly
        } else if (u_filterType == FILTER_MIRROR) {
            vec2 p = texCoord;
            if (p.x > 0.5) p.x = 1.0 - p.x;
            finalColor = texture2D(u_image, p).rgb;
        } else if (u_filterType == FILTER_FISHEYE) {
            vec2 tc = texCoord - 0.5;
            float dist = dot(tc, tc);
            float blur = 1.0 - dist * 0.8; // Adjust 0.8 for desired fisheye strength
            finalColor = texture2D(u_image, tc * blur + 0.5).rgb;
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

// Configura los buffers de un cuadrado que llena el canvas
function setupQuadBuffers(gl) {
    const positions = new Float32Array([
        -1, -1,
         1, -1,
        -1,  1,
        -1,  1,
         1, -1,
         1,  1,
    ]);
    positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    // Corrected texture coordinates for non-inverted video
    const texCoords = new Float32Array([
        0, 1,
        1, 1,
        0, 0,
        0, 0,
        1, 1,
        1, 0,
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

    bassAmpUniformLocation = gl.getUniformLocation(program, 'u_bassAmp');
    midAmpUniformLocation = gl.getUniformLocation(program, 'u_midAmp');
    highAmpUniformLocation = gl.getUniformLocation(program, 'u_highAmp');

    gl.enableVertexAttribArray(program.positionLocation);
    gl.enableVertexAttribArray(program.texCoordLocation);

    setupQuadBuffers(gl);
    setupTextures(gl); // Prepara ambas texturas aquí

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.vertexAttribPointer(program.positionLocation, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.vertexAttribPointer(program.texCoordLocation, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(program.imageLocation, 0); // La textura activa 0 será la que se use

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
      width: { ideal: 1920 }, // Request higher resolution for better quality
      height: { ideal: 1080 }
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
            dataArray = new Uint8Array(analyser.frequencyBinBinCount);
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
    // --- Fin detección de nivel de audio ---

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);

    gl.uniform2f(program.resolutionLocation, glcanvas.width, glcanvas.height);
    const currentTime = performance.now() / 1000.0;
    gl.uniform1f(timeLocation, currentTime);

    const isMediaPipeFilter = ['whiteGlow', 'blackBg', 'whiteBg'].includes(selectedFilter);

    if (isMediaPipeFilter && mpResults && mpResults.segmentationMask && mpResults.image) {
        mpCanvasCtx.clearRect(0, 0, canvas.width, canvas.height); // Siempre limpia el canvas auxiliar primero

        switch (selectedFilter) {
            case "whiteGlow": // "Silueta roja" (person red, original background)
                // Dibuja la imagen original como fondo
                mpCanvasCtx.drawImage(mpResults.image, 0, 0, canvas.width, canvas.height);
                
                // Aplica el color rojo solo donde la máscara indica la persona
                mpCanvasCtx.save();
                mpCanvasCtx.globalCompositeOperation = 'source-atop'; // Dibuja nuevo contenido solo donde ya hay contenido (la persona del original)
                mpCanvasCtx.fillStyle = 'red';
                mpCanvasCtx.fillRect(0, 0, canvas.width, canvas.height);
                mpCanvasCtx.restore();
                break;

            case "blackBg":
            case "whiteBg":
                // 1. Dibuja el color de fondo deseado (negro o blanco)
                mpCanvasCtx.fillStyle = selectedFilter === "blackBg" ? "black" : "white";
                mpCanvasCtx.fillRect(0, 0, canvas.width, canvas.height);

                // 2. Dibuja la imagen original, recortada por la máscara de segmentación, sobre el fondo
                mpCanvasCtx.globalCompositeOperation = 'destination-atop'; // Dibuja nuevo contenido solo donde el canvas ya tiene píxeles no transparentes
                mpCanvasCtx.drawImage(mpResults.image, 0, 0, canvas.width, canvas.height);
                break;
        }
        mpCanvasCtx.globalCompositeOperation = "source-over"; // Reinicia a la operación por defecto para futuros dibujos

        // Actualizar la textura WebGL con el contenido del canvas 2D de MediaPipe
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, mpOutputTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas); // Usa el canvas 2D como fuente
        gl.uniform1i(filterTypeLocation, 0); // Desactiva los filtros WebGL cuando se usa MediaPipe
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
            case 'kaleidoscope': filterIndex = 10; break; // Changed from fractal to kaleidoscope
            case 'mirror': filterIndex = 11; break;
            case 'fisheye': filterIndex = 12; break;
            default: filterIndex = 0; break;
        }
        gl.uniform1i(filterTypeLocation, filterIndex);
    }

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(drawVideoFrame);
}

function mapValue(value, inMin, inMax, outMin, outMax) {
    if (inMax === inMin) {
        return outMin; 
    }
    return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}


// Check for MP4 support
const supportsMp4 = MediaRecorder.isTypeSupported('video/mp4; codecs=avc1.424028');
const mimeType = supportsMp4 ? 'video/mp4; codecs=avc1.424028' : 'video/webm; codecs=vp8';
console.log(`Video recording will use: ${mimeType}`);

// --- MANEJADORES DE EVENTOS ---
captureBtn.addEventListener('click', () => {
    if (!gl || !glcanvas.width || !glcanvas.height) {
        console.error('WebGL no está inicializado o el canvas no tiene dimensiones para la captura.');
        return;
    }

    let img = new Image();
    img.src = glcanvas.toDataURL('image/png'); 
    
    img.onload = () => {
        addToGallery(img, 'img', 'image/png'); // Pass image mime type
    };
    img.onerror = (e) => {
        console.error('Error al cargar la imagen para la galería:', e);
    };
});


recordBtn.addEventListener('click', () => {
  if (!isRecording) {
    chunks = [];
    console.log('Iniciando grabación desde glcanvas.captureStream().');
    let streamToRecord = glcanvas.captureStream(60); // Request 60 FPS for fluidity
    
    // If there's audio in the currentStream from the camera, add it to the recording
    const audioTracks = currentStream.getAudioTracks();
    if (audioTracks.length > 0) {
        streamToRecord.addTrack(audioTracks[0]); // Combine video from canvas with audio from camera
    }

    // Set a higher bitrate for better quality
    const options = { mimeType: mimeType, videoBitsPerSecond: 8 * 1024 * 1024 }; // 8 Mbps

    mediaRecorder = new MediaRecorder(streamToRecord, options);

    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) chunks.push(e.data);
      console.log('Datos de video disponibles, tamaño:', e.data.size);
    };
    mediaRecorder.onstop = () => {
      console.log('Grabación detenida. Chunks capturados:', chunks.length);
      const blob = new Blob(chunks, { type: mimeType });
      const url = URL.createObjectURL(blob);
      let vid = document.createElement('video');
      vid.src = url;
      vid.controls = true;
      vid.autoplay = true; // Autoplay when added to gallery
      vid.onloadedmetadata = () => {
        vid.play();
        console.log('Video grabado cargado y reproduciendo.');
      };
      addToGallery(vid, 'video', mimeType.split(';')[0]); // Pass mimeType to gallery
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

function addToGallery(element, type, fileMimeType = '') {
  let container = document.createElement('div');
  container.className = 'gallery-item';
  
  // Clone the element for the gallery display (thumbnail)
  const galleryThumbnail = element.cloneNode(true);
  if (type === 'video') {
    galleryThumbnail.controls = false; // No controls for thumbnail
    galleryThumbnail.muted = true; // Mute thumbnail
    galleryThumbnail.loop = true; // Loop thumbnail
    galleryThumbnail.play(); // Autoplay thumbnail
  }
  container.appendChild(galleryThumbnail);

  // Event listener para abrir la ventana de previsualización al hacer clic en el thumbnail
  galleryThumbnail.addEventListener('click', () => {
        console.log('Creando ventana de previsualización de', type);
        
        const previewWindow = document.createElement('div');
        previewWindow.className = 'preview-window';
        document.body.appendChild(previewWindow);

        const closeButton = document.createElement('span');
        closeButton.textContent = '✖'; 
        closeButton.className = 'close-preview-window-button';
        previewWindow.appendChild(closeButton);

        const clonedElementForPreview = element.cloneNode(true); // Clone original element again for preview
        if (type === 'video') {
            clonedElementForPreview.controls = true; // Show controls for videos
            clonedElementForPreview.autoplay = true; // Autoplay on open
            clonedElementForPreview.muted = false; // Unmute for preview
            clonedElementForPreview.loop = false; // Don't loop in preview unless intended
            clonedElementForPreview.play(); // Ensure it plays when opened
        }
        previewWindow.appendChild(clonedElementForPreview);

        // --- Botones de acción en la previsualización ---
        let previewActions = document.createElement('div');
        previewActions.className = 'preview-actions';

        let downloadBtn = document.createElement('button');
        downloadBtn.textContent = '⬇'; 
        downloadBtn.onclick = () => {
            const a = document.createElement('a');
            a.href = clonedElementForPreview.src;
            const extension = type === 'img' ? 'png' : (fileMimeType === 'video/mp4' ? 'mp4' : 'webm');
            a.download = `${type}_${Date.now()}.${extension}`; 
            a.click();
            console.log('Descargando desde previsualización', type);
        };

        let shareBtn = document.createElement('button');
        shareBtn.textContent = '✉︎'; 
        shareBtn.onclick = async () => {
            if (navigator.share) {
                try {
                    const file = await fetch(clonedElementForPreview.src).then(res => res.blob());
                    const extension = type === 'img' ? 'png' : (fileMimeType === 'video/mp4' ? 'mp4' : 'webm');
                    const fileName = `${type}_${Date.now()}.${extension}`;
                    const shareData = {
                        files: [new File([file], fileName, { type: fileMimeType || (type === 'img' ? 'image/png' : 'video/webm') })],
                        // Removed title and text properties
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
        deleteBtn.textContent = '✖'; 
        deleteBtn.onclick = () => {
            if (type === 'video' && clonedElementForPreview.src.startsWith('blob:')) {
                URL.revokeObjectURL(clonedElementForPreview.src); // Revoke blob URL for videos
            }
            previewWindow.remove(); // Close preview window
            container.remove(); // Remove element from gallery
            console.log('Elemento de galería y previsualización eliminados.');
        };
        
        previewActions.appendChild(downloadBtn);
        if (navigator.share) {
            previewActions.appendChild(shareBtn);
        }
        previewActions.appendChild(deleteBtn);
        previewWindow.appendChild(previewActions);
        // --- End action buttons in preview ---

        // Event listener to close the window
        closeButton.addEventListener('click', () => {
            if (type === 'video' && clonedElementForPreview) {
                clonedElementForPreview.pause();
                // If it's a blob URL, revoke it when the preview is closed, not just when deleted
                if (clonedElementForPreview.src.startsWith('blob:')) {
                  URL.revokeObjectURL(clonedElementForPreview.src);
                }
            }
            previewWindow.remove();
            console.log('Ventana de previsualización cerrada.');
        });

        // Make the window draggable
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        previewWindow.onmousedown = dragMouseDown;

        function dragMouseDown(e) {
            e = e || window.event;
            e.preventDefault();
            // get the cursor position at startup:
            pos3 = e.clientX;
            pos4 = e.clientY;
            document.onmouseup = closeDragElement;
            // call a function whenever the cursor moves:
            document.onmousemove = elementDrag;
        }

        function elementDrag(e) {
            e = e || window.event;
            e.preventDefault();
            // calculate the new cursor position:
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;
            // set the element's new position:
            previewWindow.style.top = (previewWindow.offsetTop - pos2) + "px";
            previewWindow.style.left = (previewWindow.offsetLeft - pos1) + "px";
        }

        function closeDragElement() {
            /* stop moving when mouse button is released: */
            document.onmouseup = null;
            document.onmousemove = null;
        }
    });

  let actions = document.createElement('div');
  actions.className = 'gallery-actions';

  let downloadBtn = document.createElement('button');
  downloadBtn.textContent = '⬇';
  downloadBtn.onclick = () => {
    const a = document.createElement('a');
    a.href = element.src; // Use the original element's src
    const extension = type === 'img' ? 'png' : (fileMimeType === 'video/mp4' ? 'mp4' : 'webm');
    a.download = `${type}_${Date.now()}.${extension}`;
    a.click();
    console.log('Descargando', type);
  };

  let shareBtn = document.createElement('button');
  shareBtn.textContent = '✉︎';
  shareBtn.onclick = async () => {
    if (navigator.share) {
      try {
        const file = await fetch(element.src).then(res => res.blob()); // Use the original element's src
        const extension = type === 'img' ? 'png' : (fileMimeType === 'video/mp4' ? 'mp4' : 'webm');
        const fileName = `${type}_${Date.now()}.${extension}`;
        const shareData = {
          files: [new File([file], fileName, { type: fileMimeType || (type === 'img' ? 'image/png' : 'video/webm') })],
          // Removed title and text properties
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
  deleteBtn.textContent = '✖';
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
let touchStartY = 0; // Added for vertical swipe
let touchEndX = 0;
let touchEndY = 0; // Added for vertical swipe
const SWIPE_THRESHOLD = 50; // Minimum pixels to consider a swipe

glcanvas.addEventListener('touchstart', (e) => {
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY; // Capture Y start
});

glcanvas.addEventListener('touchmove', (e) => {
    touchEndX = e.touches[0].clientX;
    touchEndY = e.touches[0].clientY; // Capture Y end
});

glcanvas.addEventListener('touchend', () => {
    const diffX = touchEndX - touchStartX;
    const diffY = touchEndY - touchStartY; // Calculate Y difference

    // Check for horizontal swipe for filter change
    if (Math.abs(diffX) > SWIPE_THRESHOLD && Math.abs(diffX) > Math.abs(diffY)) { // Prioritize horizontal
        // Get all options, including those from optgroups
        const options = Array.from(filterSelect.querySelectorAll('option')).map(option => option.value);
        let currentIndex = options.indexOf(selectedFilter);

        if (diffX > 0) { // Swipe right (previous filter)
            currentIndex = (currentIndex > 0) ? currentIndex - 1 : options.length - 1;
        } else { // Swipe left (next filter)
            currentIndex = (currentIndex < options.length - 1) ? currentIndex + 1 : 0;
        }
        
        selectedFilter = options[currentIndex];
        filterSelect.value = selectedFilter; // Synchronize the select
        console.log('Filtro cambiado por swipe:', selectedFilter);
    } 
    // Check for vertical swipe to scroll to gallery in fullscreen
    else if (document.fullscreenElement && document.fullscreenElement.id === 'camera-container' && diffY > SWIPE_THRESHOLD && Math.abs(diffY) > Math.abs(diffX)) {
        // Swipe down to scroll to gallery
        window.scroll({
            top: document.body.scrollHeight, // Scroll to bottom of the page
            behavior: 'smooth'
        });
        console.log('Swipe down detected in fullscreen, scrolling to gallery.');
    }

    // Reset touch values
    touchStartX = 0;
    touchStartY = 0;
    touchEndX = 0;
    touchEndY = 0;
});

listCameras();
