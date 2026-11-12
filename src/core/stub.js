// Placeholder chapter used while a real module is being built.
import * as THREE from 'three';
import { header, button } from './ui.js';
import { edges } from './geom.js';

export async function enter(ctx) {
  const mesh = edges(new THREE.IcosahedronGeometry(3, 1), '#5ad7ff', { k: 1.6 });
  ctx.root.add(mesh);
  ctx.ui.left(header({ kicker: 'Under construction', title: ctx.route.path, lead: 'This chapter is being assembled.' }), button('Back to the universe', { kind: 'primary', onClick: () => ctx.router.go('') }));
  return { update: (dt) => (mesh.rotation.y += dt * 0.3) };
}
