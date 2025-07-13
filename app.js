let video = document.getElementById('video');
let glcanvas = document.getElementById('glcanvas');
let canvas = document.getElementById('canvas'); // Para la captura de imágenes estáticas
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
let cameraContainer = document.getElementById('camera-container'); // Necesario para fullscreen

let currentStream;
let mediaRecorder;
let chunks = [];
let isRecording = false;
let isPaused = false;
let selectedFilter = 'none'; // Este se usará para seleccionar el filtro en GLSL
let currentCameraDeviceId = null;
let currentFacingMode = null; // 'user' (frontal) o 'environment' (trasera)

// --- VARIABLES Y CONFIGURACIÓN DE WEBG L ---
let gl; // Contexto WebGL
let program; // Programa de shaders
let positionBuffer; // Buffer para las posiciones de los vértices
let texCoordBuffer; // Buffer para las coordenadas de textura
let videoTexture; // Textura donde se cargará el fotograma del video
let filterTypeLocation; // Ubicación del uniform para el tipo de filtro
let timeLocation; // Ubicación del uniform para el tiempo (para efectos dinámicos)

// --- VARIABLES Y CONFIGURACIÓN DE AUDIO (existente del filtro anterior) ---
let audioContext;
let analyser;
let microphone;
let dataArray; // Para almacenar los datos de frecuencia del audio
const AUDIO_THRESHOLD = 0.15; // Umbral de sonido para cambiar la paleta

let paletteIndex = 0;
const palettes = [
    [1.0, 0.0, 0.0],    // Rojo (valores normalizados 0-1)
    [0.0, 1.0, 0.0],    // Verde
    [0.0, 0.0, 1.0],    // Azul
    [1.0, 1.0, 0.0],    // Amarillo
    [1.0, 0.0, 1.0],    // Magenta
    [0.0, 1.0, 1.0],    // Cyan
];
let colorShiftUniformLocation; // Ubicación del uniform para el color de la paleta

// --- NUEVOS UNIFORMS PARA EL FILTRO MODULAR COLOR SHIFT ---
let bassAmpUniformLocation;
let midAmpUniformLocation;
let highAmpUniformLocation;

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
    uniform bool u_flipX;
    uniform int u_filterType; // Nuevo uniform para seleccionar el filtro
    uniform vec2 u_resolution; // Nuevo uniform para la resolución del canvas
    uniform float u_time; // Nuevo uniform para el tiempo, para efectos dinámicos
    uniform vec3 u_colorShift; // Uniform para el filtro de cambio de color por audio

    // NUEVOS UNIFORMS PARA EL FILTRO MODULAR COLOR SHIFT
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
    const int FILTER_MODULAR_COLOR_SHIFT = 9; // Nuevo filtro

    // Función para generar ruido básico (copiada de tu fragShader anterior)
    float random(vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
    }
    
    // Función de brillo para detectar luz (copiada de tu fragShader anterior)
    float brightness(vec3 color) {
        return dot(color, vec3(0.299, 0.587, 0.114));
    }

    void main() {
        vec2 texCoord = v_texCoord;
        if (u_flipX) {
            texCoord.x = 1.0 - texCoord.x; // Voltear horizontalmente
        }
        vec4 color = texture2D(u_image, texCoord); // Color original del píxel
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
        } else if (u_filterType == FILTER_MODULAR_COLOR_SHIFT) { // Lógica del nuevo filtro "Modular Color Shift"
            // Paletas de color (normalizadas de 0-255 a 0-1)
            const vec3 palette0 = vec3(80.0/255.0, 120.0/255.0, 180.0/255.0); // Graves
            const vec3 palette1 = vec3(100.0/255.0, 180.0/255.0, 200.0/255.0); // Medios
            const vec3 palette2 = vec3(120.0/255.0, 150.0/255.0, 255.0/255.0); // Agudos

            float brightness_val = (color.r + color.g + color.b) / 3.0; // Brillo normalizado del píxel

            // Umbrales de brillo (normalizados de 0-255 a 0-1)
            if (brightness_val > (170.0/255.0)) { // Si es muy brillante (corresponde a "agudos")
                finalColor.rgb = mix(color.rgb, palette2, u_highAmp);
            } else if (brightness_val > (100.0/255.0)) { // Si es de brillo medio (corresponde a "medios")
                finalColor.rgb = mix(color.rgb, palette1, u_midAmp);
            } else { // Si es de brillo bajo (corresponde a "graves")
                finalColor.rgb = mix(color.rgb, palette0, u_bassAmp);
            }
            finalColor.rgb = clamp(finalColor.rgb, 0.0, 1.0); // Asegurarse de que los colores estén en el rango 0-1
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

// Inicializa la textura para el video
function setupVideoTexture(gl) {
    videoTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
}

// Actualiza la textura con el fotograma actual del video
function updateVideoTexture(gl, video) {
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
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
    program.flipXLocation = gl.getUniformLocation(program, 'u_flipX');
    filterTypeLocation = gl.getUniformLocation(program, 'u_filterType'); // Obtener ubicación del uniform del filtro
    program.resolutionLocation = gl.getUniformLocation(program, 'u_resolution'); // Obtener ubicación del uniform de resolución
    timeLocation = gl.getUniformLocation(program, 'u_time'); // Obtener ubicación del uniform de tiempo
    colorShiftUniformLocation = gl.getUniformLocation(program, 'u_colorShift'); // Obtener ubicación del uniform para el cambio de color

    // Obtener ubicaciones para los nuevos uniforms del filtro modular
    bassAmpUniformLocation = gl.getUniformLocation(program, 'u_bassAmp');
    midAmpUniformLocation = gl.getUniformLocation(program, 'u_midAmp');
    highAmpUniformLocation = gl.getUniformLocation(program, 'u_highAmp');

    gl.enableVertexAttribArray(program.positionLocation);
    gl.enableVertexAttribArray(program.texCoordLocation);

    setupQuadBuffers(gl);
    setupVideoTexture(gl);

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.vertexAttribPointer(program.positionLocation, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.vertexAttribPointer(program.texCoordLocation, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(program.imageLocation, 0);

    // Establecer el filtro inicial (sin filtro)
    gl.uniform1i(filterTypeLocation, 0); // 0 = FILTER_NONE
    console.log('WebGL inicialización completa.');
}


// --- LÓGICA DE CÁMARA Y STREAMING ---
let availableCameraDevices = [];

async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');

    availableCameraDevices = videoDevices;
    console.log('Cámaras disponibles:', availableCameraDevices);
    console.log('Número de cámaras disponibles:', availableCameraDevices.length);
    
    if (availableCameraDevices.length > 0) {
      // Si no hay una cámara actual o la cámara actual ya no está disponible, selecciona la primera
      if (!currentCameraDeviceId || !availableCameraDevices.some(d => d.deviceId === currentCameraDeviceId)) {
        currentCameraDeviceId = availableCameraDevices[0].deviceId;
        console.log('Cámara inicial seleccionada:', currentCameraDeviceId);
      }
      startCamera(currentCameraDeviceId);
    } else {
      alert('No se encontraron dispositivos de cámara.');
      console.warn('No se encontraron dispositivos de cámara.');
    }
  } catch (err) {
    console.error('Error al listar dispositivos de cámara:', err);
    alert('Error al listar dispositivos de cámara. Revisa los permisos.');
  }
}

async function startCamera(deviceId) {
  console.log('Intentando iniciar cámara con Device ID:', deviceId);
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
    console.log('Stream anterior detenido.');
  }

  const constraints = {
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width: { ideal: 1280 },
      height: { ideal: 720 }
    },
    audio: true // Solicitamos acceso al micrófono aquí
  };

  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = currentStream;
    currentCameraDeviceId = deviceId || currentStream.getVideoTracks()[0].getSettings().deviceId;

    const videoTrack = currentStream.getVideoTracks()[0];
    const settings = videoTrack.getSettings();
    currentFacingMode = settings.facingMode || 'unknown';
    console.log('Cámara actual - Device ID:', currentCameraDeviceId, 'Facing Mode:', currentFacingMode);

    // --- Web Audio API setup ---
    if (currentStream.getAudioTracks().length > 0) {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256; // Un tamaño de FFT más pequeño para una respuesta más rápida
            dataArray = new Uint8Array(analyser.frequencyBinCount);
            console.log('AudioContext y Analyser inicializados.');
        }

        // Desconectar el micrófono anterior si existe
        if (microphone) {
            microphone.disconnect();
        }

        // Conectar la nueva fuente de audio
        const audioSource = audioContext.createMediaStreamSource(currentStream);
        audioSource.connect(analyser);
        // analyser.connect(audioContext.destination); // Opcional: para escuchar el audio del micrófono
        microphone = audioSource; // Guardar la referencia
        console.log('Micrófono conectado al Analyser.');
    } else {
        console.warn('No se encontró pista de audio en el stream de la cámara.');
        // Limpiar recursos de audio si no hay pista de audio disponible
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
      console.log('Video metadata cargada y reproduciendo.');
      if (glcanvas.width !== video.videoWidth || glcanvas.height !== video.videoHeight) {
        glcanvas.width = video.videoWidth;
        glcanvas.height = video.videoHeight;
        console.log('Canvas WebGL redimensionado a:', glcanvas.width, 'x', glcanvas.height);
        if (gl) {
          gl.viewport(0, 0, glcanvas.width, glcanvas.height);
          console.log('Viewport de WebGL actualizado.');
        }
      }
      // Inicializar WebGL solo una vez que el video esté listo y si no se ha inicializado ya
      if (!gl) {
        initWebGL();
        console.log('WebGL inicializado tras cargar video.');
      }
      drawVideoFrame();
      console.log('Bucle de renderizado WebGL iniciado.');
    };
  } catch (err) {
    console.error('No se pudo acceder a la cámara/micrófono:', err);
    alert('No se pudo acceder a la cámara/micrófono. Revisa los permisos. Error: ' + err.name);
  }
}

// --- BUCLE PRINCIPAL DE RENDERIZADO WEBG L ---
function drawVideoFrame() {
    // Asegurarse de que gl y program estén disponibles
    if (!gl || !program || !video.srcObject || video.readyState !== video.HAVE_ENOUGH_DATA) {
        requestAnimationFrame(drawVideoFrame);
        return;
    }

    // --- Detección de nivel de audio (solo si el filtro de audio está seleccionado) ---
    if (analyser && dataArray && selectedFilter === 'audio-color-shift') {
        analyser.getByteFrequencyData(dataArray); // O `getByteTimeDomainData` para la forma de onda
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
        }
        let average = sum / dataArray.length;
        let normalizedLevel = average / 255.0; // Normalizar a un rango de 0-1

        if (normalizedLevel > AUDIO_THRESHOLD) {
            changePaletteIndex(); // Cambiar la paleta si el sonido es fuerte
        }
    }
    // --- Fin detección de nivel de audio ---

    updateVideoTexture(gl, video);

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(program);

    const isFrontFacing = currentFacingMode === 'user';
    gl.uniform1i(program.flipXLocation, isFrontFacing ? 1 : 0);
    
    // Pasar la resolución al shader
    gl.uniform2f(program.resolutionLocation, glcanvas.width, glcanvas.height);
    
    // Pasar el tiempo al shader (en segundos)
    const currentTime = performance.now() / 1000.0;
    gl.uniform1f(timeLocation, currentTime);

    // Pasar el color de la paleta actual al shader si el filtro activo es 'audio-color-shift'
    if (selectedFilter === 'audio-color-shift') {
        const currentColor = palettes[paletteIndex];
        gl.uniform3fv(colorShiftUniformLocation, new Float32Array(currentColor));
    }

    // --- Pasar "amplitudes" basadas en tiempo para el filtro Modular Color Shift ---
    if (selectedFilter === 'modular-color-shift') {
        // Generar valores pulsantes usando seno y mapearlos a los rangos deseados
        const bassAmp = mapValue(Math.sin(currentTime * 0.8 + 0), -1, 1, 0.0, 2.0); // 0-2.0
        const midAmp = mapValue(Math.sin(currentTime * 1.2 + Math.PI / 3), -1, 1, 0.0, 1.5); // 0-1.5
        const highAmp = mapValue(Math.sin(currentTime * 1.5 + Math.PI * 2 / 3), -1, 1, 0.0, 2.5); // 0-2.5

        gl.uniform1f(bassAmpUniformLocation, bassAmp);
        gl.uniform1f(midAmpUniformLocation, midAmp);
        gl.uniform1f(highAmpUniformLocation, highAmp);
    }

    let filterIndex = 0; // FILTER_NONE por defecto
    switch (selectedFilter) {
        case 'grayscale': filterIndex = 1; break; // FILTER_GRAYSCALE
        case 'invert': filterIndex = 2; break;    // FILTER_INVERT
        case 'sepia': filterIndex = 3; break;     // FILTER_SEPIA
        case 'eco-pink': filterIndex = 4; break;  // FILTER_ECO_PINK
        case 'weird': filterIndex = 5; break;     // FILTER_WEIRD
        case 'glow-outline': filterIndex = 6; break; // Filtro Glow con contorno
        case 'angelical-glitch': filterIndex = 7; break; // Filtro Angelical Glitch
        case 'audio-color-shift': filterIndex = 8; break; // Filtro Audio Color Shift
        case 'modular-color-shift': filterIndex = 9; break; // Nuevo filtro Modular Color Shift
        default: filterIndex = 0; break;
    }
    gl.uniform1i(filterTypeLocation, filterIndex);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(drawVideoFrame);
}

// Función auxiliar para mapear un valor de un rango a otro
function mapValue(value, inMin, inMax, outMin, outMax) {
    return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}


// --- MANEJADORES DE EVENTOS ---
captureBtn.addEventListener('click', () => {
    console.log('Botón de captura clickeado.');
    if (!gl || !glcanvas.width || !glcanvas.height) {
        console.error('WebGL no está inicializado o el canvas no tiene dimensiones para la captura.');
        return;
    }

    let img = new Image();
    img.src = glcanvas.toDataURL('image/png'); 
    
    img.onload = () => {
        console.log('Imagen cargada para la galería desde glcanvas.toDataURL().');
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
      vid.onloadedmetadata = () => {
        vid.play();
        console.log('Video grabado cargado y reproduciendo.');
      };
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
    pauseBtn.textContent = '⏸️';
    console.log('Grabación reanudada.');
  } else {
    mediaRecorder.pause();
    pauseBtn.textContent = '▶️';
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
  console.log('Filtro seleccionado:', selectedFilter);
});

fullscreenBtn.addEventListener('click', () => {
  if (!document.fullscreenElement) {
    cameraContainer.requestFullscreen(); // cameraContainer es el elemento principal para fullscreen
    console.log('Solicitando fullscreen.');
  } else {
    document.exitFullscreen();
    console.log('Saliendo de fullscreen.');
  }
});

function addToGallery(element, type) {
  let container = document.createElement('div');
  container.className = 'gallery-item';
  container.appendChild(element);

  let actions = document.createElement('div');
  actions.className = 'gallery-actions';

  let downloadBtn = document.createElement('button');
  downloadBtn.textContent = 'Descargar';
  downloadBtn.onclick = () => {
    const a = document.createElement('a');
    a.href = element.src;
    a.download = type === 'img' ? 'foto.png' : 'video.webm';
    a.click();
    console.log('Descargando', type);
  };

  let shareBtn = document.createElement('button');
  shareBtn.textContent = 'Compartir';
  shareBtn.onclick = async () => {
    if (navigator.share) {
      try {
        const file = await fetch(element.src).then(res => res.blob());
        const fileName = type === 'img' ? 'foto.png' : 'video.webm';
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
  if (navigator.share) { // Solo añadir el botón de compartir si la API está disponible
    actions.appendChild(shareBtn);
  }
  actions.appendChild(deleteBtn);
  container.appendChild(actions);

  gallery.prepend(container); // Añadir al principio de la galería
}

// --- LÓGICA DE DOBLE TAP/CLICK PARA CAMBIAR DE CÁMARA ---
let lastTap = 0;
const DBL_TAP_THRESHOLD = 300;

glcanvas.addEventListener('touchend', (event) => {
    console.log("Evento 'touchend' en glcanvas.");
    const currentTime = new Date().getTime();
    const tapLength = currentTime - lastTap;

    if (tapLength < DBL_TAP_THRESHOLD && tapLength > 0) {
        console.log("¡Doble tap detectado!");
        event.preventDefault(); // Prevenir el zoom por doble tap predeterminado
        toggleCamera();
    }
    lastTap = currentTime;
}, { passive: false }); // Usar { passive: false } para permitir preventDefault

glcanvas.addEventListener('dblclick', () => {
    console.log("Evento 'dblclick' en glcanvas.");
    toggleCamera();
});

// Función centralizada para cambiar de cámara
function toggleCamera() {
    if (availableCameraDevices.length > 1) {
        const currentIdx = availableCameraDevices.findIndex(
            device => device.deviceId === currentCameraDeviceId
        );
        const nextIdx = (currentIdx + 1) % availableCameraDevices.length;
        const nextDeviceId = availableCameraDevices[nextIdx].deviceId;
        console.log('Cambiando de cámara. Actual:', currentCameraDeviceId, 'Siguiente:', nextDeviceId);
        startCamera(nextDeviceId);
    } else {
        console.log("Solo hay una cámara disponible para cambiar.");
        alert("Solo hay una cámara disponible.");
    }
}

// Función para cambiar el índice de la paleta (para filtro de audio)
function changePaletteIndex() {
    paletteIndex = (paletteIndex + 1) % palettes.length;
    console.log('Paleta cambiada a índice:', paletteIndex, ' Color:', palettes[paletteIndex]);
}

// Iniciar el proceso de listar cámaras y obtener el stream
listCameras();
