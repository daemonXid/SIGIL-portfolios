/**
 * 🌊 DAEMON Fluid Simulation Engine v2.0
 * 
 * WebGL2 GPU-based Navier-Stokes fluid dynamics simulation
 * Optimized for DAEMON-SIGIL
 * 
 * @author DAEMON Architect
 * @license MIT
 * 
 * Features:
 * - WebGL2 with fallback to WebGL1
 * - Float textures with linear filtering
 * - 8 Color modes
 * - GPU particle system
 * - Automatic quality adjustment
 */

class DaemonFluidEngine {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) throw new Error(`Canvas "${canvasId}" not found`);

        // Try WebGL2, fallback to WebGL1
        this.gl = this.canvas.getContext('webgl2', {
            alpha: false,
            depth: false,
            stencil: false,
            antialias: false,
            preserveDrawingBuffer: false,
            powerPreference: 'high-performance',
        });

        this.isWebGL2 = !!this.gl;

        if (!this.gl) {
            this.gl = this.canvas.getContext('webgl', {
                alpha: false,
                depth: false,
                stencil: false,
                antialias: false,
                preserveDrawingBuffer: false,
            });
        }

        if (!this.gl) throw new Error('WebGL not supported');

        console.log(`🌊 DAEMON Fluid Engine: ${this.isWebGL2 ? 'WebGL2' : 'WebGL1'}`);

        // Extensions
        this._initExtensions();

        // Settings with defaults
        this.settings = {
            quality: options.quality ?? 2,
            solverIterations: options.solverIterations ?? 20,
            dyeIntensity: options.dyeIntensity ?? 1.2,
            forceRadius: options.forceRadius ?? 0.018,
            colorMode: options.colorMode ?? 0,
            cellSize: 32,
            particleSpeed: options.particleSpeed ?? 1.0,
            dyeDecay: options.dyeDecay ?? 0.98,
            velocityDissipation: options.velocityDissipation ?? 0.999,
        };

        // Quality presets (optimized)
        this.qualityPresets = [
            { particles: 16384, scale: 1 / 8, iterations: 10 }, // Ultra Low
            { particles: 65536, scale: 1 / 6, iterations: 14 }, // Low
            { particles: 262144, scale: 1 / 4, iterations: 20 }, // Medium
            { particles: 524288, scale: 1 / 3, iterations: 25 }, // High
            { particles: 1048576, scale: 1 / 2, iterations: 32 }, // Ultra High
        ];

        // Color mode names (8 modes)
        this.colorModeNames = [
            'Plasma', 'Ocean', 'Fire', 'Neon',
            'Aurora', 'Sunset', 'Cosmic', 'Matrix'
        ];

        // State
        this.simWidth = 0;
        this.simHeight = 0;
        this.aspectRatio = 1;

        // Mouse
        this.pointer = { x: 0, y: 0, dx: 0, dy: 0, isDown: false, moved: false };
        this.lastPointer = { x: 0, y: 0 };

        // Multi-touch support
        this.pointers = [];

        // Performance
        this.frameCount = 0;
        this.lastFpsTime = performance.now();
        this.currentFps = 60;
        this.lastFrameTime = performance.now();
        this.autoQualityEnabled = options.autoQuality !== false;
        this.lowFpsCount = 0;

        // Initialize
        this._initShaders();
        this._initBuffers();
        this._initFramebuffers();
        this._initEventListeners();
        this._resize();

        // Start
        this._animate();
    }

    // ========== EXTENSIONS ==========

    _initExtensions() {
        const gl = this.gl;

        if (this.isWebGL2) {
            // WebGL2 - float textures built-in
            this.floatTexSupport = true;
            this.floatLinearSupport = !!gl.getExtension('EXT_color_buffer_float');
            gl.getExtension('EXT_color_buffer_float');
        } else {
            // WebGL1 - need extensions
            this.floatTexSupport = !!gl.getExtension('OES_texture_float');
            this.floatLinearSupport = !!gl.getExtension('OES_texture_float_linear');
            this.halfFloatExt = gl.getExtension('OES_texture_half_float');
            gl.getExtension('OES_texture_half_float_linear');
        }
    }

    // ========== SHADERS ==========

    _getShaderHeader() {
        if (this.isWebGL2) {
            return `#version 300 es
precision highp float;
precision highp sampler2D;
`;
        }
        return `precision highp float;
`;
    }

    _getVertexSource() {
        if (this.isWebGL2) {
            return `#version 300 es
in vec2 a_position;
out vec2 v_texCoord;
out vec2 v_simPos;
uniform float u_aspectRatio;

void main() {
    v_texCoord = a_position;
    vec2 clipSpace = a_position * 2.0 - 1.0;
    v_simPos = vec2(clipSpace.x * u_aspectRatio, clipSpace.y);
    gl_Position = vec4(clipSpace, 0.0, 1.0);
}`;
        }
        return `
attribute vec2 a_position;
varying vec2 v_texCoord;
varying vec2 v_simPos;
uniform float u_aspectRatio;

void main() {
    v_texCoord = a_position;
    vec2 clipSpace = a_position * 2.0 - 1.0;
    v_simPos = vec2(clipSpace.x * u_aspectRatio, clipSpace.y);
    gl_Position = vec4(clipSpace, 0.0, 1.0);
}`;
    }

    _getAdvectSource() {
        const prefix = this.isWebGL2 ? `#version 300 es
precision highp float;
in vec2 v_texCoord;
in vec2 v_simPos;
out vec4 fragColor;
#define texture2D texture
` : `precision highp float;
varying vec2 v_texCoord;
varying vec2 v_simPos;
`;
        const suffix = this.isWebGL2 ? 'fragColor' : 'gl_FragColor';

        return `${prefix}
uniform sampler2D u_velocity;
uniform sampler2D u_target;
uniform float u_dt;
uniform float u_rdx;
uniform vec2 u_invRes;
uniform float u_aspectRatio;
uniform float u_dissipation;

vec2 simToTexel(vec2 simPos) {
    return vec2(simPos.x / u_aspectRatio + 1.0, simPos.y + 1.0) * 0.5;
}

void main() {
    vec2 vel = texture2D(u_velocity, v_texCoord).xy;
    vec2 tracedPos = v_simPos - u_dt * u_rdx * vel;
    vec2 texelPos = simToTexel(tracedPos) / u_invRes;
    
    vec2 st = floor(texelPos - 0.5) + 0.5;
    vec2 t = texelPos - st;
    st *= u_invRes;
    vec2 st2 = st + u_invRes;
    
    vec4 t11 = texture2D(u_target, st);
    vec4 t21 = texture2D(u_target, vec2(st2.x, st.y));
    vec4 t12 = texture2D(u_target, vec2(st.x, st2.y));
    vec4 t22 = texture2D(u_target, st2);
    
    ${suffix} = mix(mix(t11, t21, t.x), mix(t12, t22, t.x), t.y) * u_dissipation;
}`;
    }

    _getDivergenceSource() {
        const prefix = this.isWebGL2 ? `#version 300 es
precision highp float;
in vec2 v_texCoord;
out vec4 fragColor;
#define texture2D texture
` : `precision highp float;
varying vec2 v_texCoord;
`;
        const suffix = this.isWebGL2 ? 'fragColor' : 'gl_FragColor';

        return `${prefix}
uniform sampler2D u_velocity;
uniform float u_halfRdx;
uniform vec2 u_invRes;

vec2 sampleVel(vec2 coord) {
    vec2 offset = vec2(0.0);
    vec2 mult = vec2(1.0);
    if (coord.x < 0.0) { offset.x = 1.0; mult.x = -1.0; }
    else if (coord.x > 1.0) { offset.x = -1.0; mult.x = -1.0; }
    if (coord.y < 0.0) { offset.y = 1.0; mult.y = -1.0; }
    else if (coord.y > 1.0) { offset.y = -1.0; mult.y = -1.0; }
    return mult * texture2D(u_velocity, coord + offset * u_invRes).xy;
}

void main() {
    vec2 L = sampleVel(v_texCoord - vec2(u_invRes.x, 0.0));
    vec2 R = sampleVel(v_texCoord + vec2(u_invRes.x, 0.0));
    vec2 B = sampleVel(v_texCoord - vec2(0.0, u_invRes.y));
    vec2 T = sampleVel(v_texCoord + vec2(0.0, u_invRes.y));
    ${suffix} = vec4(u_halfRdx * ((R.x - L.x) + (T.y - B.y)), 0.0, 0.0, 1.0);
}`;
    }

    _getPressureSource() {
        const prefix = this.isWebGL2 ? `#version 300 es
precision highp float;
in vec2 v_texCoord;
out vec4 fragColor;
#define texture2D texture
` : `precision highp float;
varying vec2 v_texCoord;
`;
        const suffix = this.isWebGL2 ? 'fragColor' : 'gl_FragColor';

        return `${prefix}
uniform sampler2D u_pressure;
uniform sampler2D u_divergence;
uniform float u_alpha;
uniform vec2 u_invRes;

float sampleP(vec2 coord) {
    vec2 offset = vec2(0.0);
    if (coord.x < 0.0) offset.x = 1.0;
    else if (coord.x > 1.0) offset.x = -1.0;
    if (coord.y < 0.0) offset.y = 1.0;
    else if (coord.y > 1.0) offset.y = -1.0;
    return texture2D(u_pressure, coord + offset * u_invRes).x;
}

void main() {
    float L = sampleP(v_texCoord - vec2(u_invRes.x, 0.0));
    float R = sampleP(v_texCoord + vec2(u_invRes.x, 0.0));
    float B = sampleP(v_texCoord - vec2(0.0, u_invRes.y));
    float T = sampleP(v_texCoord + vec2(0.0, u_invRes.y));
    float div = texture2D(u_divergence, v_texCoord).x;
    ${suffix} = vec4((L + R + B + T + u_alpha * div) * 0.25, 0.0, 0.0, 1.0);
}`;
    }

    _getGradientSource() {
        const prefix = this.isWebGL2 ? `#version 300 es
precision highp float;
in vec2 v_texCoord;
out vec4 fragColor;
#define texture2D texture
` : `precision highp float;
varying vec2 v_texCoord;
`;
        const suffix = this.isWebGL2 ? 'fragColor' : 'gl_FragColor';

        return `${prefix}
uniform sampler2D u_pressure;
uniform sampler2D u_velocity;
uniform float u_halfRdx;
uniform vec2 u_invRes;

float sampleP(vec2 coord) {
    vec2 offset = vec2(0.0);
    if (coord.x < 0.0) offset.x = 1.0;
    else if (coord.x > 1.0) offset.x = -1.0;
    if (coord.y < 0.0) offset.y = 1.0;
    else if (coord.y > 1.0) offset.y = -1.0;
    return texture2D(u_pressure, coord + offset * u_invRes).x;
}

void main() {
    float L = sampleP(v_texCoord - vec2(u_invRes.x, 0.0));
    float R = sampleP(v_texCoord + vec2(u_invRes.x, 0.0));
    float B = sampleP(v_texCoord - vec2(0.0, u_invRes.y));
    float T = sampleP(v_texCoord + vec2(0.0, u_invRes.y));
    vec2 vel = texture2D(u_velocity, v_texCoord).xy;
    ${suffix} = vec4(vel - u_halfRdx * vec2(R - L, T - B), 0.0, 1.0);
}`;
    }

    _getForceSource() {
        const prefix = this.isWebGL2 ? `#version 300 es
precision highp float;
in vec2 v_texCoord;
in vec2 v_simPos;
out vec4 fragColor;
#define texture2D texture
` : `precision highp float;
varying vec2 v_texCoord;
varying vec2 v_simPos;
`;
        const suffix = this.isWebGL2 ? 'fragColor' : 'gl_FragColor';

        return `${prefix}
uniform sampler2D u_velocity;
uniform float u_dt;
uniform float u_dx;
uniform vec2 u_pointer;
uniform vec2 u_lastPointer;
uniform bool u_isDown;
uniform float u_radius;
uniform float u_aspectRatio;

vec2 clipToSim(vec2 c) { return vec2(c.x * u_aspectRatio, c.y); }

float distToSeg(vec2 a, vec2 b, vec2 p, out float fp) {
    vec2 d = p - a;
    vec2 x = b - a;
    float lx = length(x);
    if (lx <= 0.0001) return length(d);
    float proj = dot(d, x / lx);
    fp = proj / lx;
    if (proj < 0.0) return length(d);
    else if (proj > lx) return length(p - b);
    return sqrt(abs(dot(d, d) - proj * proj));
}

void main() {
    vec2 vel = texture2D(u_velocity, v_texCoord).xy;
    
    if (u_isDown) {
        vec2 ptr = clipToSim(u_pointer);
        vec2 lastPtr = clipToSim(u_lastPointer);
        vec2 ptrVel = -(lastPtr - ptr) / max(u_dt, 0.001);
        
        float fp;
        float dist = distToSeg(ptr, lastPtr, v_simPos, fp);
        float tapering = 1.0 - clamp(fp, 0.0, 1.0) * 0.6;
        
        float m = exp(-dist / u_radius);
        m *= tapering * tapering;
        
        vec2 targetVel = ptrVel * u_dx;
        vel += (targetVel - vel) * m;
    }
    
    ${suffix} = vec4(vel, 0.0, 1.0);
}`;
    }

    _getDyeSource() {
        const prefix = this.isWebGL2 ? `#version 300 es
precision highp float;
in vec2 v_texCoord;
in vec2 v_simPos;
out vec4 fragColor;
#define texture2D texture
` : `precision highp float;
varying vec2 v_texCoord;
varying vec2 v_simPos;
`;
        const suffix = this.isWebGL2 ? 'fragColor' : 'gl_FragColor';

        return `${prefix}
uniform sampler2D u_dye;
uniform float u_dt;
uniform vec2 u_pointer;
uniform vec2 u_lastPointer;
uniform bool u_isDown;
uniform float u_radius;
uniform float u_aspectRatio;
uniform float u_intensity;
uniform int u_colorMode;
uniform float u_decay;

vec2 clipToSim(vec2 c) { return vec2(c.x * u_aspectRatio, c.y); }

float distToSeg(vec2 a, vec2 b, vec2 p, out float fp) {
    vec2 d = p - a;
    vec2 x = b - a;
    float lx = length(x);
    if (lx <= 0.0001) return length(d);
    float proj = dot(d, x / lx);
    fp = proj / lx;
    if (proj < 0.0) return length(d);
    else if (proj > lx) return length(p - b);
    return sqrt(abs(dot(d, d) - proj * proj));
}

// 8 Color palettes
vec3 plasma(float x) {
    return mix(vec3(0.134, 0.0, 0.117), vec3(0.0, 0.478, 1.0), x) + vec3(0.631, 0.925, 1.0) * pow(x, 9.0) * 0.1;
}

vec3 ocean(float x) {
    return mix(vec3(0.0, 0.02, 0.1), vec3(0.0, 0.5, 0.7), x) + vec3(0.7, 1.0, 1.0) * pow(x, 5.0) * 0.25;
}

vec3 fire(float x) {
    return mix(vec3(0.15, 0.0, 0.0), vec3(1.0, 0.35, 0.0), x) + vec3(1.0, 1.0, 0.4) * pow(x, 3.0) * 0.35;
}

vec3 neon(float x) {
    return mix(vec3(0.4, 0.0, 0.5), vec3(0.0, 1.0, 0.4), x) + vec3(1.0, 1.0, 1.0) * pow(x, 7.0) * 0.2;
}

vec3 aurora(float x) {
    vec3 c1 = vec3(0.1, 0.0, 0.2);
    vec3 c2 = vec3(0.0, 0.8, 0.4);
    vec3 c3 = vec3(0.3, 1.0, 0.9);
    return mix(mix(c1, c2, x), c3, x * x) + vec3(0.8, 1.0, 0.9) * pow(x, 6.0) * 0.15;
}

vec3 sunset(float x) {
    vec3 c1 = vec3(0.15, 0.0, 0.1);
    vec3 c2 = vec3(0.9, 0.3, 0.1);
    vec3 c3 = vec3(1.0, 0.8, 0.3);
    return mix(mix(c1, c2, x), c3, x * x) + vec3(1.0, 0.9, 0.7) * pow(x, 4.0) * 0.2;
}

vec3 cosmic(float x) {
    vec3 c1 = vec3(0.05, 0.0, 0.15);
    vec3 c2 = vec3(0.5, 0.1, 0.7);
    vec3 c3 = vec3(0.9, 0.6, 1.0);
    return mix(mix(c1, c2, x), c3, x * x) + vec3(1.0, 0.8, 1.0) * pow(x, 5.0) * 0.25;
}

vec3 matrix(float x) {
    return vec3(0.0, x * 0.8, 0.0) + vec3(0.2, 1.0, 0.3) * pow(x, 4.0) * 0.4;
}

vec3 getColor(int mode, float x) {
    if (mode == 0) return plasma(x);
    if (mode == 1) return ocean(x);
    if (mode == 2) return fire(x);
    if (mode == 3) return neon(x);
    if (mode == 4) return aurora(x);
    if (mode == 5) return sunset(x);
    if (mode == 6) return cosmic(x);
    return matrix(x);
}

void main() {
    vec4 color = texture2D(u_dye, v_texCoord);
    color.rgb *= u_decay;
    
    if (u_isDown) {
        vec2 ptr = clipToSim(u_pointer);
        vec2 lastPtr = clipToSim(u_lastPointer);
        vec2 ptrVel = -(lastPtr - ptr) / max(u_dt, 0.001);
        
        float fp;
        float dist = distToSeg(ptr, lastPtr, v_simPos, fp);
        float tapering = 1.0 - clamp(fp, 0.0, 1.0) * 0.6;
        
        float R = 0.03;
        float m = exp(-dist / R);
        
        float speed = length(ptrVel);
        float x = clamp((speed * speed * 0.015 - dist * 4.0) * tapering, 0.0, 1.0);
        
        color.rgb += m * getColor(u_colorMode, x) * u_intensity;
    }
    
    ${suffix} = color;
}`;
    }

    _getDisplaySource() {
        const prefix = this.isWebGL2 ? `#version 300 es
precision highp float;
in vec2 v_texCoord;
out vec4 fragColor;
#define texture2D texture
` : `precision highp float;
varying vec2 v_texCoord;
`;
        const suffix = this.isWebGL2 ? 'fragColor' : 'gl_FragColor';

        return `${prefix}
uniform sampler2D u_texture;
void main() { ${suffix} = texture2D(u_texture, v_texCoord); }`;
    }

    _getParticleVertSource() {
        if (this.isWebGL2) {
            return `#version 300 es
in vec2 a_particleUV;
uniform sampler2D u_particleData;
out vec4 v_color;

void main() {
    vec4 data = texture(u_particleData, a_particleUV);
    vec2 pos = data.xy;
    vec2 vel = data.zw;
    
    gl_PointSize = 1.0;
    gl_Position = vec4(pos, 0.0, 1.0);
    
    float speed = length(vel);
    float x = clamp(speed * 4.0, 0.0, 1.0);
    v_color.rgb = mix(vec3(0.134, 0.0, 0.117), vec3(0.0, 0.478, 1.0), x) + vec3(0.631, 0.925, 1.0) * x * x * x * 0.1;
    v_color.a = 1.0;
}`;
        }
        return `
attribute vec2 a_particleUV;
uniform sampler2D u_particleData;
varying vec4 v_color;

void main() {
    vec4 data = texture2D(u_particleData, a_particleUV);
    vec2 pos = data.xy;
    vec2 vel = data.zw;
    
    gl_PointSize = 1.0;
    gl_Position = vec4(pos, 0.0, 1.0);
    
    float speed = length(vel);
    float x = clamp(speed * 4.0, 0.0, 1.0);
    v_color.rgb = mix(vec3(0.134, 0.0, 0.117), vec3(0.0, 0.478, 1.0), x) + vec3(0.631, 0.925, 1.0) * x * x * x * 0.1;
    v_color.a = 1.0;
}`;
    }

    _getParticleFragSource() {
        if (this.isWebGL2) {
            return `#version 300 es
precision highp float;
in vec4 v_color;
out vec4 fragColor;
void main() { fragColor = v_color; }`;
        }
        return `precision highp float;
varying vec4 v_color;
void main() { gl_FragColor = v_color; }`;
    }

    _getParticleUpdateSource() {
        const prefix = this.isWebGL2 ? `#version 300 es
precision highp float;
in vec2 v_texCoord;
out vec4 fragColor;
#define texture2D texture
` : `precision highp float;
varying vec2 v_texCoord;
`;
        const suffix = this.isWebGL2 ? 'fragColor' : 'gl_FragColor';

        return `${prefix}
uniform sampler2D u_particleData;
uniform sampler2D u_velocity;
uniform float u_dt;
uniform vec2 u_flowScale;
uniform float u_drag;
uniform float u_speed;

void main() {
    vec4 data = texture2D(u_particleData, v_texCoord);
    vec2 pos = data.xy;
    vec2 vel = data.zw;
    
    vec2 texCoord = (pos + 1.0) * 0.5;
    vec2 flowVel = texture2D(u_velocity, texCoord).xy * u_flowScale;
    
    vel += (flowVel - vel) * u_drag;
    pos += u_dt * vel * u_speed;
    
    // Wrap
    pos = mod(pos + 1.0, 2.0) - 1.0;
    
    ${suffix} = vec4(pos, vel);
}`;
    }

    _getParticleInitSource() {
        const prefix = this.isWebGL2 ? `#version 300 es
precision highp float;
in vec2 v_texCoord;
out vec4 fragColor;
` : `precision highp float;
varying vec2 v_texCoord;
`;
        const suffix = this.isWebGL2 ? 'fragColor' : 'gl_FragColor';

        return `${prefix}
void main() {
    vec2 pos = v_texCoord * 2.0 - 1.0;
    ${suffix} = vec4(pos, 0.0, 0.0);
}`;
    }

    // ========== COMPILATION ==========

    _compileShader(source, type) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader error:', gl.getShaderInfoLog(shader));
            console.error('Source:', source);
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    _createProgram(vertSrc, fragSrc) {
        const gl = this.gl;
        const vert = this._compileShader(vertSrc, gl.VERTEX_SHADER);
        const frag = this._compileShader(fragSrc, gl.FRAGMENT_SHADER);
        if (!vert || !frag) return null;

        const prog = gl.createProgram();
        gl.attachShader(prog, vert);
        gl.attachShader(prog, frag);
        gl.linkProgram(prog);

        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            console.error('Link error:', gl.getProgramInfoLog(prog));
            return null;
        }

        prog.uniforms = {};
        const numUniforms = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < numUniforms; i++) {
            const info = gl.getActiveUniform(prog, i);
            prog.uniforms[info.name] = gl.getUniformLocation(prog, info.name);
        }

        prog.attributes = {};
        const numAttribs = gl.getProgramParameter(prog, gl.ACTIVE_ATTRIBUTES);
        for (let i = 0; i < numAttribs; i++) {
            const info = gl.getActiveAttrib(prog, i);
            prog.attributes[info.name] = gl.getAttribLocation(prog, info.name);
        }

        return prog;
    }

    _initShaders() {
        const vertSrc = this._getVertexSource();

        this.programs = {
            advect: this._createProgram(vertSrc, this._getAdvectSource()),
            divergence: this._createProgram(vertSrc, this._getDivergenceSource()),
            pressure: this._createProgram(vertSrc, this._getPressureSource()),
            gradient: this._createProgram(vertSrc, this._getGradientSource()),
            force: this._createProgram(vertSrc, this._getForceSource()),
            dye: this._createProgram(vertSrc, this._getDyeSource()),
            display: this._createProgram(vertSrc, this._getDisplaySource()),
            particleRender: this._createProgram(this._getParticleVertSource(), this._getParticleFragSource()),
            particleUpdate: this._createProgram(vertSrc, this._getParticleUpdateSource()),
            particleInit: this._createProgram(vertSrc, this._getParticleInitSource()),
        };
    }

    // ========== BUFFERS ==========

    _initBuffers() {
        const gl = this.gl;
        this.quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    }

    _initParticleBuffers() {
        const gl = this.gl;
        const preset = this.qualityPresets[this.settings.quality];
        const count = preset.particles;

        const dataSize = Math.ceil(Math.sqrt(count));
        this.particleDataSize = dataSize;
        this.particleCount = dataSize * dataSize;

        const uvs = new Float32Array(this.particleCount * 2);
        for (let i = 0; i < dataSize; i++) {
            for (let j = 0; j < dataSize; j++) {
                const idx = (i * dataSize + j) * 2;
                uvs[idx] = (j + 0.5) / dataSize;
                uvs[idx + 1] = (i + 0.5) / dataSize;
            }
        }

        if (this.particleUVBuffer) gl.deleteBuffer(this.particleUVBuffer);
        this.particleUVBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.particleUVBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);

        if (this.particleData) {
            this.particleData.forEach(t => {
                gl.deleteTexture(t.texture);
                gl.deleteFramebuffer(t.fbo);
            });
        }

        this.particleData = [
            this._createRenderTarget(dataSize, dataSize, true),
            this._createRenderTarget(dataSize, dataSize, true),
        ];
        this.particleReadIdx = 0;
        this._initParticles();
    }

    _initParticles() {
        const gl = this.gl;
        const prog = this.programs.particleInit;

        gl.useProgram(prog);
        gl.viewport(0, 0, this.particleDataSize, this.particleDataSize);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        const posLoc = prog.attributes.a_position;
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

        for (let i = 0; i < 2; i++) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.particleData[i].fbo);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }
    }

    // ========== FRAMEBUFFERS ==========

    _createRenderTarget(w, h, isFloat = false) {
        const gl = this.gl;
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        if (this.isWebGL2) {
            if (isFloat) {
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
            } else {
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            }
        } else {
            if (isFloat && this.floatTexSupport) {
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.FLOAT, null);
            } else {
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            }
        }

        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

        return { texture: tex, fbo, width: w, height: h };
    }

    _createDoubleFBO(w, h, isFloat = false) {
        return [this._createRenderTarget(w, h, isFloat), this._createRenderTarget(w, h, isFloat)];
    }

    _initFramebuffers() {
        const preset = this.qualityPresets[this.settings.quality];

        this.simWidth = Math.max(32, Math.round(window.innerWidth * preset.scale));
        this.simHeight = Math.max(32, Math.round(window.innerHeight * preset.scale));
        this.aspectRatio = this.simWidth / this.simHeight;

        this.settings.solverIterations = preset.iterations;

        this.velocityFBO = this._createDoubleFBO(this.simWidth, this.simHeight, true);
        this.pressureFBO = this._createDoubleFBO(this.simWidth, this.simHeight, true);
        this.divergenceFBO = this._createRenderTarget(this.simWidth, this.simHeight, true);
        this.dyeFBO = this._createDoubleFBO(this.simWidth, this.simHeight, false);

        this.velIdx = 0;
        this.pressIdx = 0;
        this.dyeIdx = 0;

        this._initParticleBuffers();
    }

    // ========== EVENTS ==========

    _initEventListeners() {
        window.addEventListener('resize', () => this._resize());

        // Mouse
        this.canvas.addEventListener('mousedown', e => {
            this.pointer.isDown = true;
            this._updatePointer(e);
            this.lastPointer.x = this.pointer.x;
            this.lastPointer.y = this.pointer.y;
        });
        window.addEventListener('mouseup', () => { this.pointer.isDown = false; });
        this.canvas.addEventListener('mousemove', e => {
            this.pointer.moved = true;
            this._updatePointer(e);
        });

        // Touch
        this.canvas.addEventListener('touchstart', e => {
            e.preventDefault();
            this.pointer.isDown = true;
            this._updateTouch(e);
            this.lastPointer.x = this.pointer.x;
            this.lastPointer.y = this.pointer.y;
        }, { passive: false });
        this.canvas.addEventListener('touchend', () => { this.pointer.isDown = false; });
        this.canvas.addEventListener('touchmove', e => {
            e.preventDefault();
            this.pointer.moved = true;
            this._updateTouch(e);
        }, { passive: false });
    }

    _updatePointer(e) {
        const rect = this.canvas.getBoundingClientRect();
        this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    }

    _updateTouch(e) {
        if (e.touches.length > 0) {
            const t = e.touches[0];
            const rect = this.canvas.getBoundingClientRect();
            this.pointer.x = ((t.clientX - rect.left) / rect.width) * 2 - 1;
            this.pointer.y = -((t.clientY - rect.top) / rect.height) * 2 + 1;
        }
    }

    _resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;

        const preset = this.qualityPresets[this.settings.quality];
        const newW = Math.max(32, Math.round(window.innerWidth * preset.scale));
        const newH = Math.max(32, Math.round(window.innerHeight * preset.scale));

        if (newW !== this.simWidth || newH !== this.simHeight) {
            this._initFramebuffers();
        }
    }

    // ========== SIMULATION ==========

    _step(dt) {
        const gl = this.gl;
        gl.viewport(0, 0, this.simWidth, this.simHeight);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);

        this._advect(this.velocityFBO, 'velIdx', dt, this.settings.velocityDissipation);
        this._applyForces(dt);
        this._computeDivergence();
        this._solvePressure();
        this._subtractGradient();
        this._updateDye(dt);
        this._advect(this.dyeFBO, 'dyeIdx', dt, this.settings.dyeDecay);
        this._updateParticles(dt);

        this.lastPointer.x = this.pointer.x;
        this.lastPointer.y = this.pointer.y;
    }

    _advect(target, idxName, dt, dissipation) {
        const gl = this.gl;
        const prog = this.programs.advect;
        gl.useProgram(prog);

        const readIdx = this[idxName];
        const writeIdx = 1 - readIdx;

        gl.bindFramebuffer(gl.FRAMEBUFFER, target[writeIdx].fbo);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocityFBO[this.velIdx].texture);
        gl.uniform1i(prog.uniforms.u_velocity, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, target[readIdx].texture);
        gl.uniform1i(prog.uniforms.u_target, 1);

        gl.uniform1f(prog.uniforms.u_dt, dt);
        gl.uniform1f(prog.uniforms.u_rdx, 1 / this.settings.cellSize);
        gl.uniform2f(prog.uniforms.u_invRes, 1 / this.simWidth, 1 / this.simHeight);
        gl.uniform1f(prog.uniforms.u_aspectRatio, this.aspectRatio);
        gl.uniform1f(prog.uniforms.u_dissipation, dissipation);

        const posLoc = prog.attributes.a_position;
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        this[idxName] = writeIdx;
    }

    _applyForces(dt) {
        const gl = this.gl;
        const prog = this.programs.force;
        gl.useProgram(prog);

        const writeIdx = 1 - this.velIdx;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocityFBO[writeIdx].fbo);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocityFBO[this.velIdx].texture);
        gl.uniform1i(prog.uniforms.u_velocity, 0);

        gl.uniform1f(prog.uniforms.u_dt, dt);
        gl.uniform1f(prog.uniforms.u_dx, this.settings.cellSize);
        gl.uniform2f(prog.uniforms.u_pointer, this.pointer.x, this.pointer.y);
        gl.uniform2f(prog.uniforms.u_lastPointer, this.lastPointer.x, this.lastPointer.y);
        gl.uniform1i(prog.uniforms.u_isDown, this.pointer.isDown ? 1 : 0);
        gl.uniform1f(prog.uniforms.u_radius, this.settings.forceRadius);
        gl.uniform1f(prog.uniforms.u_aspectRatio, this.aspectRatio);

        const posLoc = prog.attributes.a_position;
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        this.velIdx = writeIdx;
    }

    _computeDivergence() {
        const gl = this.gl;
        const prog = this.programs.divergence;
        gl.useProgram(prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.divergenceFBO.fbo);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocityFBO[this.velIdx].texture);
        gl.uniform1i(prog.uniforms.u_velocity, 0);

        gl.uniform1f(prog.uniforms.u_halfRdx, 0.5 / this.settings.cellSize);
        gl.uniform2f(prog.uniforms.u_invRes, 1 / this.simWidth, 1 / this.simHeight);

        const posLoc = prog.attributes.a_position;
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    _solvePressure() {
        const gl = this.gl;
        const prog = this.programs.pressure;
        gl.useProgram(prog);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.divergenceFBO.texture);
        gl.uniform1i(prog.uniforms.u_divergence, 1);

        gl.uniform1f(prog.uniforms.u_alpha, -this.settings.cellSize * this.settings.cellSize);
        gl.uniform2f(prog.uniforms.u_invRes, 1 / this.simWidth, 1 / this.simHeight);

        const posLoc = prog.attributes.a_position;
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

        for (let i = 0; i < this.settings.solverIterations; i++) {
            const writeIdx = 1 - this.pressIdx;
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.pressureFBO[writeIdx].fbo);

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.pressureFBO[this.pressIdx].texture);
            gl.uniform1i(prog.uniforms.u_pressure, 0);

            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            this.pressIdx = writeIdx;
        }
    }

    _subtractGradient() {
        const gl = this.gl;
        const prog = this.programs.gradient;
        gl.useProgram(prog);

        const writeIdx = 1 - this.velIdx;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocityFBO[writeIdx].fbo);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.pressureFBO[this.pressIdx].texture);
        gl.uniform1i(prog.uniforms.u_pressure, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.velocityFBO[this.velIdx].texture);
        gl.uniform1i(prog.uniforms.u_velocity, 1);

        gl.uniform1f(prog.uniforms.u_halfRdx, 0.5 / this.settings.cellSize);
        gl.uniform2f(prog.uniforms.u_invRes, 1 / this.simWidth, 1 / this.simHeight);

        const posLoc = prog.attributes.a_position;
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        this.velIdx = writeIdx;
    }

    _updateDye(dt) {
        const gl = this.gl;
        const prog = this.programs.dye;
        gl.useProgram(prog);

        const writeIdx = 1 - this.dyeIdx;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.dyeFBO[writeIdx].fbo);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.dyeFBO[this.dyeIdx].texture);
        gl.uniform1i(prog.uniforms.u_dye, 0);

        gl.uniform1f(prog.uniforms.u_dt, dt);
        gl.uniform2f(prog.uniforms.u_pointer, this.pointer.x, this.pointer.y);
        gl.uniform2f(prog.uniforms.u_lastPointer, this.lastPointer.x, this.lastPointer.y);
        gl.uniform1i(prog.uniforms.u_isDown, this.pointer.isDown ? 1 : 0);
        gl.uniform1f(prog.uniforms.u_radius, this.settings.forceRadius);
        gl.uniform1f(prog.uniforms.u_aspectRatio, this.aspectRatio);
        gl.uniform1f(prog.uniforms.u_intensity, this.settings.dyeIntensity);
        gl.uniform1i(prog.uniforms.u_colorMode, this.settings.colorMode);
        gl.uniform1f(prog.uniforms.u_decay, this.settings.dyeDecay);

        const posLoc = prog.attributes.a_position;
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        this.dyeIdx = writeIdx;
    }

    _updateParticles(dt) {
        const gl = this.gl;
        const prog = this.programs.particleUpdate;
        gl.useProgram(prog);
        gl.viewport(0, 0, this.particleDataSize, this.particleDataSize);

        const writeIdx = 1 - this.particleReadIdx;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.particleData[writeIdx].fbo);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.particleData[this.particleReadIdx].texture);
        gl.uniform1i(prog.uniforms.u_particleData, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.velocityFBO[this.velIdx].texture);
        gl.uniform1i(prog.uniforms.u_velocity, 1);

        gl.uniform1f(prog.uniforms.u_dt, dt);
        gl.uniform2f(prog.uniforms.u_flowScale,
            1 / (this.settings.cellSize * this.aspectRatio),
            1 / this.settings.cellSize
        );
        gl.uniform1f(prog.uniforms.u_drag, 1.0);
        gl.uniform1f(prog.uniforms.u_speed, this.settings.particleSpeed);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        const posLoc = prog.attributes.a_position;
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        this.particleReadIdx = writeIdx;
    }

    // ========== RENDER ==========

    _render() {
        const gl = this.gl;
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.SRC_ALPHA);
        gl.blendEquation(gl.FUNC_ADD);

        this._renderParticles();
        this._renderDye();

        gl.disable(gl.BLEND);
    }

    _renderParticles() {
        const gl = this.gl;
        const prog = this.programs.particleRender;
        gl.useProgram(prog);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.particleData[this.particleReadIdx].texture);
        gl.uniform1i(prog.uniforms.u_particleData, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.particleUVBuffer);
        const uvLoc = prog.attributes.a_particleUV;
        gl.enableVertexAttribArray(uvLoc);
        gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, this.particleCount);
    }

    _renderDye() {
        const gl = this.gl;
        const prog = this.programs.display;
        gl.useProgram(prog);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.dyeFBO[this.dyeIdx].texture);
        gl.uniform1i(prog.uniforms.u_texture, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        const posLoc = prog.attributes.a_position;
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    // ========== ANIMATION ==========

    _animate() {
        const now = performance.now();
        const dt = Math.min((now - this.lastFrameTime) / 1000, 0.02);
        this.lastFrameTime = now;

        this.frameCount++;
        if (now - this.lastFpsTime >= 500) {
            this.currentFps = Math.round(this.frameCount * 1000 / (now - this.lastFpsTime));
            this.frameCount = 0;
            this.lastFpsTime = now;

            // Auto quality adjustment (less aggressive)
            if (this.autoQualityEnabled && this.currentFps < 20 && this.settings.quality > 0) {
                this.lowFpsCount++;
                if (this.lowFpsCount >= 5) {
                    this.setQuality(this.settings.quality - 1);
                    this.lowFpsCount = 0;
                    console.log(`⚠️ Auto-lowering quality to ${this.settings.quality}`);
                }
            } else {
                this.lowFpsCount = 0;
            }
        }

        this._step(dt);
        this._render();

        requestAnimationFrame(() => this._animate());
    }

    // ========== PUBLIC API ==========

    getFPS() { return this.currentFps; }
    getParticleCount() { return this.particleCount; }
    getColorModeNames() { return this.colorModeNames; }

    setQuality(q) {
        this.settings.quality = Math.max(0, Math.min(4, q));
        this._initFramebuffers();
    }

    updateSettings(s) { Object.assign(this.settings, s); }

    reset() {
        const gl = this.gl;
        gl.clearColor(0, 0, 0, 1);
        [this.velocityFBO, this.pressureFBO, this.dyeFBO].forEach(arr => {
            arr.forEach(t => {
                gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
                gl.clear(gl.COLOR_BUFFER_BIT);
            });
        });
        this._initParticles();
    }
}

window.DaemonFluidEngine = DaemonFluidEngine;
