// The shared deep-space backdrop: a tinted nebula dome + twinkling stars that follow the camera,
// so every chapter feels like a place in the same universe.
import * as THREE from 'three';
import { rng } from './geom.js';

export class Background {
  constructor(scene, { stars = 5000, nebula = true } = {}) {
    this.group = new THREE.Group();
    this.group.renderOrder = -20;
    scene.add(this.group);
    this.tint = new THREE.Color('#2b4fd6');
    this.tintTarget = this.tint.clone();

    this.nebulaMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uTint: { value: this.tint },
        uTint2: { value: new THREE.Color('#7a3cc4') },
        uDetail: { value: nebula ? 1 : 0 },
      },
      vertexShader: /* glsl */ `
        varying vec3 vDir;
        void main() {
          vDir = normalize(position);
          vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_Position = p.xyww;
        }`,
      fragmentShader: /* glsl */ `
        varying vec3 vDir;
        uniform float uTime;
        uniform vec3 uTint;
        uniform vec3 uTint2;
        uniform float uDetail;
        float hash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
        float noise(vec3 x) {
          vec3 i = floor(x); vec3 f = fract(x); f = f * f * (3.0 - 2.0 * f);
          return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
                     mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
        }
        float fbm(vec3 p) { float a = 0.5; float s = 0.0; for (int i = 0; i < 4; i++) { s += a * noise(p); p = p * 2.03 + 11.7; a *= 0.5; } return s; }
        void main() {
          vec3 d = normalize(vDir);
          vec3 base = mix(vec3(0.010, 0.014, 0.030), vec3(0.004, 0.006, 0.014), smoothstep(-0.6, 0.9, d.y));
          vec3 col = base;
          if (uDetail > 0.5) {
            float n = fbm(d * 2.2 + vec3(0.0, uTime * 0.004, 0.0));
            float m = fbm(d * 4.5 - 3.1);
            float cloud = smoothstep(0.42, 0.85, n) * (0.55 + 0.45 * m);
            float cloud2 = smoothstep(0.5, 0.9, fbm(d.zxy * 2.6 + 7.3));
            col += uTint * cloud * 0.085 + uTint2 * cloud2 * 0.05;
          } else {
            col += uTint * 0.02 * smoothstep(0.2, -0.8, d.y);
          }
          gl_FragColor = vec4(col, 1.0);
        }`,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: true,
      fog: false,
    });
    const dome = new THREE.Mesh(new THREE.SphereGeometry(50, 24, 16), this.nebulaMat);
    dome.frustumCulled = false;
    dome.renderOrder = -21;
    this.group.add(dome);

    const rnd = rng(20261112);
    const pos = new Float32Array(stars * 3);
    const col = new Float32Array(stars * 3);
    const size = new Float32Array(stars);
    const phase = new Float32Array(stars);
    const c = new THREE.Color();
    const tones = ['#ffffff', '#cfe0ff', '#a9c7ff', '#ffe9c9', '#ffd2f0'];
    for (let i = 0; i < stars; i++) {
      const u = rnd() * 2 - 1;
      const a = rnd() * Math.PI * 2;
      const r = Math.sqrt(1 - u * u);
      pos[i * 3] = Math.cos(a) * r * 40;
      pos[i * 3 + 1] = u * 40;
      pos[i * 3 + 2] = Math.sin(a) * r * 40;
      const bright = Math.pow(rnd(), 6);
      c.set(tones[Math.floor(rnd() * tones.length)]).multiplyScalar(0.35 + bright * 1.6);
      col[i * 3] = c.r;
      col[i * 3 + 1] = c.g;
      col[i * 3 + 2] = c.b;
      size[i] = 1.0 + bright * 2.6 + rnd() * 0.8;
      phase[i] = rnd() * 6.283;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(size, 1));
    geo.setAttribute('phase', new THREE.BufferAttribute(phase, 1));
    this.starMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uPx: { value: 1 } },
      vertexShader: /* glsl */ `
        attribute float size; attribute float phase; attribute vec3 color;
        varying vec3 vColor;
        uniform float uTime; uniform float uPx;
        void main() {
          float tw = 0.78 + 0.22 * sin(uTime * (0.6 + fract(phase) * 1.4) + phase);
          vColor = color * tw;
          vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_Position = p.xyww;
          gl_PointSize = size * uPx;
        }`,
      fragmentShader: /* glsl */ `
        varying vec3 vColor;
        void main() {
          float d = length(gl_PointCoord - 0.5) * 2.0;
          float a = smoothstep(1.0, 0.0, d);
          gl_FragColor = vec4(vColor * a, a);
        }`,
      transparent: true,
      depthWrite: false,
      depthTest: true, // drawn at the far plane (z = w), so anything in the scene covers the stars
      blending: THREE.AdditiveBlending,
      fog: false,
    });
    const starPts = new THREE.Points(geo, this.starMat);
    starPts.frustumCulled = false;
    starPts.renderOrder = -20;
    this.group.add(starPts);
  }

  /** Chapters tint the nebula with their accent colour. */
  setTint(color) {
    this.tintTarget.set(color);
  }

  setDetail(on) {
    this.nebulaMat.uniforms.uDetail.value = on ? 1 : 0;
  }

  update(dt, t, camera, pixelRatio) {
    this.group.position.copy(camera.position);
    this.tint.lerp(this.tintTarget, 1 - Math.exp(-dt * 1.5));
    this.nebulaMat.uniforms.uTime.value = t;
    this.starMat.uniforms.uTime.value = t;
    this.starMat.uniforms.uPx.value = pixelRatio;
  }
}
